"""ERP agent client: Claude  <->  erp_backend_tools.py (MCP)  <->  ERP backend.

* MCP se saare ERP tools auto-discover hote hain (erp_orders, erp_inventory, ...).
* Claude ka jawab token-by-token stream hota hai.
* HITL: koi bhi write action (add / update / delete / approve ...) chalne se pehle
  insaan se approve / reject / edit poocha jata hai. Read actions seedhe chalte hain.

Usage:
    python client.py                       # interactive chat
    python client.py "pending orders dikhao"   # one-shot

.env (client + server dono padhte hain):
    ANTHROPIC_API_KEY=...
    ERP_BASE_URL=https://acerbiomedical.in
    ERP_USERNAME=...   ERP_PASSWORD=...    # service account (server use karta hai)
    MCP_SERVER=erp_backend_tools.py        # ya http://127.0.0.1:8766/mcp
    CLAUDE_MODEL=claude-haiku-5-5
"""
import argparse
import asyncio
import json
import os
import sys
from typing import Any

import anthropic
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from typing_extensions import NotRequired, TypedDict

load_dotenv()

MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
MCP_SERVER = os.getenv("MCP_SERVER", "erp_backend_tools.py")
MAX_TOOL_CHARS = int(os.getenv("MAX_TOOL_CHARS", "12000"))  # bade tool output se context na bhare
MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "12"))         # tool-loop ki upper limit

READ_ACTIONS = {
    "help", "filter", "list", "view", "summary",
    "serial_history", "serial_search", "available_serials", "invoice_lookup",
    "available_parts", "available_hologram_parts", "fx_rate",
    "damage_image", "media", "proof",
}

SYSTEM_PROMPT = """You are the assistant for Acer Biomedical's ERP. You work through the erp_* tools.
- Reply in the same language the user writes in (Hindi/Hinglish/English).
- Unsure about a resource's filters or data fields? Call the tool with action='help' first.
- Never invent ids, serial numbers or amounts; look them up with filter/view/summary.
- For add/update/delete/approve etc. a human reviews and approves each call before it runs.
  Say briefly what you are about to change. If a call is rejected, do not retry it unchanged;
  ask the user what to do instead.
- Keep answers short and factual; quote ids and numbers exactly as the tools return them."""


# --------------------------------------------------------------------------- types
class ToolSpec(TypedDict):
    name: str
    description: str
    input_schema: dict[str, Any]
    cache_control: NotRequired[dict[str, str]]


class PendingCall(TypedDict):
    id: str
    name: str
    input: dict[str, Any]


class Approval(TypedDict):
    approved: bool
    reason: NotRequired[str]
    edited_input: NotRequired[dict[str, Any]]


class ToolOutcome(TypedDict):
    tool_use_id: str
    content: str
    is_error: bool


# --------------------------------------------------------------------------- MCP
def build_mcp() -> Client:
    if MCP_SERVER.startswith(("http://", "https://")):
        return Client(MCP_SERVER)
    # stdio: server ko hum spawn karte hain. Auth token LLM ko kabhi nahi dikhana, isliye
    # `auth_token` param band (server service-account se login karta hai).
    env = {**os.environ, "ERP_EXPOSE_AUTH_TOKEN": "0", "MCP_TRANSPORT": "stdio"}
    return Client(PythonStdioTransport(script_path=MCP_SERVER, env=env, python_cmd=sys.executable))


async def load_tools(mcp: Client) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for t in await mcp.list_tools():
        raw = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
        schema = json.loads(json.dumps(raw or {"type": "object", "properties": {}}))
        schema.setdefault("type", "object")
        schema.get("properties", {}).pop("auth_token", None)  # http mode me bhi chhupao
        if "required" in schema:
            schema["required"] = [r for r in schema["required"] if r != "auth_token"]
        specs.append({"name": t.name, "description": t.description or "", "input_schema": schema})
    if specs:
        specs[-1]["cache_control"] = {"type": "ephemeral"}  # tools list cache => sasta + tez
    return specs


def result_text(res: Any) -> str:
    parts = [c.text for c in (getattr(res, "content", None) or []) if getattr(c, "text", None)]
    text = "\n".join(parts)
    if not text:
        text = json.dumps(getattr(res, "structured_content", None) or getattr(res, "data", None),
                          default=str, ensure_ascii=False)
    try:  # compact JSON = kam tokens
        text = json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)
    except ValueError:
        pass
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + f"... [truncated {len(text) - MAX_TOOL_CHARS} chars; narrow filters/limit]"
    return text


def looks_failed(text: str) -> bool:
    try:
        data = json.loads(text.split("... [truncated")[0])
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("ok") is False


# --------------------------------------------------------------------------- HITL
def needs_approval(call: PendingCall) -> bool:
    args = call["input"]
    if "action" not in args:  # erp_overview jaise read-only tools
        return False
    return str(args.get("action", "")).strip().lower() not in READ_ACTIONS


def _ask(prompt: str) -> str:
    return input(prompt).strip()


async def ask_approval(call: PendingCall) -> Approval:
    args = call["input"]
    print("\n\n+-- APPROVAL NEEDED " + "-" * 40)
    print(f"| tool   : {call['name']}")
    print(f"| action : {args.get('action')}   id: {args.get('id') or '-'}   id2: {args.get('id2') or '-'}")
    print("| data   :")
    for line in json.dumps(args.get("data"), indent=2, ensure_ascii=False, default=str).splitlines():
        print("|   " + line)
    print("+" + "-" * 58)
    while True:
        choice = (await asyncio.to_thread(_ask, "[y] approve   [n] reject   [e] edit data  > ")).lower()
        if choice in ("y", "yes", "h", "haan"):
            return {"approved": True}
        if choice in ("n", "no", "nahi"):
            why = await asyncio.to_thread(_ask, "reason (optional) > ")
            return {"approved": False, "reason": why}
        if choice == "e":
            raw = await asyncio.to_thread(_ask, "new `data` JSON (single line) > ")
            try:
                edited = {**args, "data": json.loads(raw)}
            except ValueError as e:
                print(f"invalid JSON: {e}")
                continue
            return {"approved": True, "edited_input": edited}


async def execute(mcp: Client, call: PendingCall) -> ToolOutcome:
    if needs_approval(call):
        decision = await ask_approval(call)
        if not decision["approved"]:
            why = decision.get("reason") or "no reason given"
            return {"tool_use_id": call["id"], "is_error": True,
                    "content": f"REJECTED by the human reviewer ({why}). Nothing was changed."}
        call["input"] = decision.get("edited_input", call["input"])
    args = {k: v for k, v in call["input"].items() if k != "auth_token"}
    try:
        res = await mcp.call_tool(call["name"], args, raise_on_error=False)
        text = result_text(res)
        return {"tool_use_id": call["id"], "content": text,
                "is_error": bool(getattr(res, "is_error", False)) or looks_failed(text)}
    except Exception as e:  # MCP / network error model ko wapas dikhao, crash nahi
        return {"tool_use_id": call["id"], "content": f"tool call failed: {e}", "is_error": True}


# --------------------------------------------------------------------------- agent loop (streaming)
async def run_turn(llm: anthropic.AsyncAnthropic, mcp: Client, tools: list[ToolSpec],
                   messages: list[dict[str, Any]]) -> None:
    for _ in range(MAX_STEPS):
        async with llm.messages.stream(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
                                       tools=tools, messages=messages) as stream:  # type: ignore[arg-type]
            async for text in stream.text_stream:
                print(text, end="", flush=True)
            final = await stream.get_final_message()
        messages.append({"role": "assistant", "content": final.content})

        if final.stop_reason != "tool_use":
            print()
            return

        calls: list[PendingCall] = [
            {"id": b.id, "name": b.name, "input": dict(b.input)}
            for b in final.content if b.type == "tool_use"
        ]
        outcomes: list[ToolOutcome] = []
        for call in calls:  # sequential: approval prompts ek-ek karke aayein
            if not needs_approval(call):
                print(f"\n  > {call['name']}({call['input'].get('action')})", flush=True)
            outcomes.append(await execute(mcp, call))
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": o["tool_use_id"], "content": o["content"],
             "is_error": o["is_error"]} for o in outcomes]})
    print(f"\n[stopped after {MAX_STEPS} tool steps - ask again to continue]")


async def amain(prompt: str | None) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY set nahi hai (.env me daalo).")
    llm = anthropic.AsyncAnthropic()
    async with build_mcp() as mcp:
        tools = await load_tools(mcp)
        print(f"connected to ERP MCP: {len(tools)} tools ({', '.join(t['name'] for t in tools)})")
        messages: list[dict[str, Any]] = []

        if prompt:
            messages.append({"role": "user", "content": prompt})
            await run_turn(llm, mcp, tools, messages)
            return

        print("type your question. /reset = new chat, /tools = list tools, exit = quit\n")
        while True:
            try:
                text = (await asyncio.to_thread(_ask, "you > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                continue
            if text.lower() in ("exit", "quit", "/exit"):
                return
            if text == "/reset":
                messages.clear()
                print("chat reset.")
                continue
            if text == "/tools":
                for t in tools:
                    print(f"- {t['name']}: {t['description'][:90]}")
                continue
            messages.append({"role": "user", "content": text})
            try:
                await run_turn(llm, mcp, tools, messages)
            except anthropic.APIError as e:
                print(f"\n[model error] {e}")
                messages.pop()  # adhura turn hatao taaki history valid rahe
            except KeyboardInterrupt:
                print("\n[interrupted]")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Claude agent for the ERP MCP server")
    ap.add_argument("prompt", nargs="?", help="one-shot question (khali = interactive)")
    asyncio.run(amain(ap.parse_args().prompt))


if __name__ == "__main__":
    main()