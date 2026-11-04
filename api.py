import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator, Optional

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from pydantic import BaseModel, Field

load_dotenv()

from configuration import get_params
from data_extraction import DocumentProcessor, UnsupportedFormatError
from logger import logger
from model_call import get_model
from vector_store import get_store

ERP_BASE = os.getenv("ERP_BASE_URL", "https://acerbiomedical.in").rstrip("/")
MCP_SERVER = os.getenv("MCP_SERVER", "erp_backend_tools.py")
CHAT_CHARS = int(os.getenv("CHAT_DOC_CHARS", "60000"))
SPEC_CHARS = int(os.getenv("SPEC_CHARS", "30000"))
SRC_RE = re.compile(r"(?m)^## \[(catalogue|manual)\] (.+)$")
SPLIT_RE = re.compile(r"(?m)^(?=## \[(?:catalogue|manual)\] )")

SYSTEM_PROMPT = (
    "You are Kiwi, an AI assistant developed by Acer Biomedicals LLP, a medical equipment manufacturer in Noida, "
    "Uttar Pradesh, India. Never mention Claude or Anthropic. Answer only from the product documents provided; "
    "never invent specs and say clearly when data is missing. Reply in the user's language."
)

processor = DocumentProcessor()
SPECS: dict[str, tuple[float, str]] = {}  # spec_id -> (time, text); chhota aur short-lived
SPEC_TTL = 900  # 15 min
ATTACH_RE = re.compile(r"<attachment\b.*?</attachment>", re.S)


def _put_spec(spec_id: str, text: str) -> None:
    now = time.time()
    for k in [k for k, (t, _) in SPECS.items() if now - t > SPEC_TTL]:
        SPECS.pop(k, None)
    while len(SPECS) >= 5:
        SPECS.pop(next(iter(SPECS)))
    SPECS[spec_id] = (now, text)


def _scrub_attachments(messages: list[dict[str, Any]]) -> None:
    """Turn poora hone ke baad uploaded file ka text history se hata do; chat baaki rehti hai."""
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and "<attachment" in m["content"]:
            m["content"] = ATTACH_RE.sub("<attachment removed after use>", m["content"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MCP_SERVER.startswith(("http://", "https://")):
        mcp = Client(MCP_SERVER)
    else:
        env = {**os.environ, "ERP_EXPOSE_AUTH_TOKEN": "1", "MCP_TRANSPORT": "stdio"}
        mcp = Client(PythonStdioTransport(script_path=MCP_SERVER, env=env, python_cmd=sys.executable))
    async with mcp:
        app.state.mcp = mcp
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def user_token(authorization: str = Header(default="")) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="login required")
    return token


def _extract(file: UploadFile) -> str:
    content = file.file.read()
    try:
        return processor.extract(file.filename or "", content)
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"extraction failed: {type(e).__name__}: {e}")


def _sse(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class LoginReq(BaseModel):
    username: str
    password: str


APP_CLIENT_KEY = os.getenv("APP_CLIENT_KEY", "").strip()


@app.post("/api/login")
async def login(req: LoginReq) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if APP_CLIENT_KEY:
        headers["User-Agent"] = f"erp-mcp key={APP_CLIENT_KEY}"
    url = f"{ERP_BASE}/login/"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.post(url, json={"username": req.username, "password": req.password}, headers=headers)
    except Exception as e:
        logger.exception("erp login request failed")
        raise HTTPException(status_code=502, detail=f"cannot reach {url} -> {type(e).__name__}: {e}")
    body = r.text[:300]
    logger.info("erp login: POST %s -> %s | client_key_set=%s | body=%s", url, r.status_code, bool(APP_CLIENT_KEY), body)
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code != 200 or not data.get("access_token"):
        detail = data.get("detail") if isinstance(data, dict) and data.get("detail") else body
        raise HTTPException(status_code=401, detail=f"POST {url} -> {r.status_code}: {detail}")
    return {"access_token": data["access_token"], "username": req.username, "role": data.get("role", "")}


def _products() -> list[dict[str, Any]]:
    store = get_store()  # sirf naam + chhoti info; text disk se nahi padhta
    return [{"name": n, **store.info(n)} for n in store.list_documents()]


@app.get("/api/products")
def products(_: str = Depends(user_token)) -> list[dict[str, Any]]:
    return _products()


@app.post("/api/products/upload")
def upload_product_file(
    product: str = Form(...),
    kind: str = Form(...),
    file: UploadFile = File(...),
    _: str = Depends(user_token),
) -> dict[str, Any]:
    if kind not in ("catalogue", "manual"):
        raise HTTPException(status_code=422, detail="kind must be catalogue or manual")
    product = product.strip()
    if not product:
        raise HTTPException(status_code=422, detail="product name required")
    text = _extract(file)
    header = f"## [{kind}] {file.filename}"
    block = f"{header}\n{text}"
    store = get_store()
    if store.has(product):
        parts = [p for p in SPLIT_RE.split(store.get_text(product)) if p.strip()]
        parts = [p for p in parts if not p.startswith(header + "\n")]
        merged = "\n\n".join([*(p.strip() for p in parts), block])
    else:
        merged = f"# {product}\n\n{block}"
    chunks = processor.chunk(merged, product)
    try:
        n = store.add(product, merged, chunks)
    except Exception:
        logger.exception("save failed")
        raise HTTPException(status_code=500, detail="vector store save failed")
    return {"product": product, "chunks": n, "chars": len(merged)}


@app.get("/api/products/text")
def product_text(name: str, _: str = Depends(user_token)) -> dict[str, str]:
    store = get_store()
    if not store.has(name):
        raise HTTPException(status_code=404, detail="not found")
    return {"name": name, "text": store.get_text(name)}


@app.delete("/api/products")
def delete_product(name: str, _: str = Depends(user_token)) -> dict[str, str]:
    store = get_store()
    if not store.has(name):
        raise HTTPException(status_code=404, detail="not found")
    with store._lock:
        store._remove(name)
        if store.index is not None:
            store._persist()
    return {"deleted": name}


class ChatReq(BaseModel):
    product: Optional[str] = None
    messages: list[dict[str, Any]] = Field(min_length=1)


@app.post("/api/chat")
def chat(req: ChatReq, _: str = Depends(user_token)) -> StreamingResponse:
    system: list[dict[str, Any]] = [{"type": "text", "text": SYSTEM_PROMPT}]
    if req.product:
        store = get_store()
        if not store.has(req.product):
            raise HTTPException(status_code=404, detail="product not found")
        system.append({
            "type": "text",
            "text": f'<product name="{req.product}">\n{store.get_text(req.product)[:CHAT_CHARS]}\n</product>',
            "cache_control": {"type": "ephemeral"},
        })
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in req.messages
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip()
    ]
    model = get_model()

    def gen() -> Iterator[str]:
        try:
            with model.client.messages.stream(
                model=model.model, max_tokens=model.max_tokens, system=system, messages=messages
            ) as s:
                for t in s.text_stream:
                    yield _sse("token", t)
            yield _sse("done", "[DONE]")
        except Exception:
            logger.exception("chat stream failed")
            yield _sse("error", "model call failed")

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/erp/tools")
async def erp_tools(_: str = Depends(user_token)) -> list[dict[str, Any]]:
    out = []
    for t in await app.state.mcp.list_tools():
        raw = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
        schema = json.loads(json.dumps(raw))
        schema.get("properties", {}).pop("auth_token", None)
        out.append({"name": t.name, "description": t.description or "", "schema": schema})
    return out


class ErpCall(BaseModel):
    tool: str
    args: dict[str, Any] = {}


@app.post("/api/erp/call")
async def erp_call(req: ErpCall, token: str = Depends(user_token)) -> dict[str, Any]:
    args = {**req.args, "auth_token": token}
    try:
        res = await app.state.mcp.call_tool(req.tool, args, raise_on_error=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"tool call failed: {e}")
    text = "\n".join(c.text for c in (res.content or []) if getattr(c, "text", None))
    ok = not getattr(res, "is_error", False)
    try:
        return {"ok": ok, "result": json.loads(text)}
    except ValueError:
        return {"ok": ok, "result": text}


READ_ACTIONS = {
    "help", "filter", "list", "view", "summary", "serial_history", "serial_search", "available_serials",
    "invoice_lookup", "available_parts", "available_hologram_parts", "fx_rate", "damage_image", "media", "proof",
}
MAX_TOOL_CHARS = int(os.getenv("MAX_TOOL_CHARS", "12000"))
MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "12"))
ERP_SYSTEM = (
    "You are Kiwi, the assistant for Acer Biomedical's ERP (developed by Acer Biomedicals LLP, India). "
    "Never mention Claude or Anthropic. You work through the erp_* tools.\n"
    "- Reply in the same language the user writes in (Hindi/Hinglish/English).\n"
    "- Unsure about a resource's filters or data fields? Call the tool with action='help' first.\n"
    "- Never invent ids, serial numbers or amounts; look them up with filter/view/summary.\n"
    "- For add/update/delete/approve etc. a human reviews and approves each call before it runs. "
    "Say briefly what you are about to change. If a call is rejected, do not retry it unchanged; ask the user.\n"
    "- When the user attaches a document (invoice, PO, etc., given in <attachment> tags) and asks for an entry: "
    "read it, call action='help' on the right tool to learn the data fields, look up needed ids with read actions, "
    "then make ONE write call with data filled as completely as possible. Do not ask the user to type details in chat: "
    "the write call is shown to them as an editable form where they review and correct everything. "
    "OCR text can have errors; leave a field empty instead of guessing.\n"
    "- Keep answers short and factual; quote ids and numbers exactly as the tools return them."
)
ERP_SESS: dict[str, dict[str, Any]] = {}
_llm: Optional[anthropic.AsyncAnthropic] = None


def _async_llm() -> anthropic.AsyncAnthropic:
    global _llm
    if _llm is None:
        _llm = anthropic.AsyncAnthropic()
    return _llm


async def _specs() -> list[dict[str, Any]]:
    cached = getattr(app.state, "specs", None)
    if cached:
        return cached
    specs = []
    for t in await app.state.mcp.list_tools():
        raw = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        schema = json.loads(json.dumps(raw))
        schema.setdefault("type", "object")
        schema.get("properties", {}).pop("auth_token", None)
        if "required" in schema:
            schema["required"] = [r for r in schema["required"] if r != "auth_token"]
        specs.append({"name": t.name, "description": t.description or "", "input_schema": schema})
    if specs:
        specs[-1]["cache_control"] = {"type": "ephemeral"}
    app.state.specs = specs
    return specs


def _needs_approval(call: dict[str, Any]) -> bool:
    args = call["input"]
    if "action" not in args:
        return False
    return str(args.get("action", "")).strip().lower() not in READ_ACTIONS


def _result_text(res: Any) -> str:
    text = "\n".join(c.text for c in (getattr(res, "content", None) or []) if getattr(c, "text", None))
    if not text:
        text = json.dumps(getattr(res, "structured_content", None) or getattr(res, "data", None), default=str, ensure_ascii=False)
    try:
        text = json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)
    except ValueError:
        pass
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + f"... [truncated {len(text) - MAX_TOOL_CHARS} chars; narrow filters/limit]"
    return text


async def _exec(call: dict[str, Any], token: str) -> dict[str, Any]:
    args = {**{k: v for k, v in call["input"].items() if k != "auth_token"}, "auth_token": token}
    try:
        res = await app.state.mcp.call_tool(call["name"], args, raise_on_error=False)
        text = _result_text(res)
        bad = bool(getattr(res, "is_error", False))
        if not bad:
            try:
                d = json.loads(text.split("... [truncated")[0])
                bad = isinstance(d, dict) and d.get("ok") is False
            except ValueError:
                pass
        return {"type": "tool_result", "tool_use_id": call["id"], "content": text, "is_error": bad}
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": call["id"], "content": f"tool call failed: {e}", "is_error": True}


async def _drive(sess: dict[str, Any], token: str):
    llm = _async_llm()
    specs = await _specs()
    model = get_params()["model"]
    while True:
        if sess["calls"] is None:
            if sess["steps"] >= MAX_STEPS:
                yield _sse("token", "\n[tool step limit reached - ask again to continue]")
                yield _sse("done", "[DONE]")
                return
            sess["steps"] += 1
            try:
                async with llm.messages.stream(
                    model=model, max_tokens=4096, system=ERP_SYSTEM, tools=specs, messages=sess["messages"]
                ) as st:
                    async for t in st.text_stream:
                        yield _sse("token", t)
                    final = await st.get_final_message()
            except Exception as e:
                logger.exception("erp agent model call failed")
                yield _sse("error", f"model call failed: {e}")
                return
            sess["messages"].append({"role": "assistant", "content": final.content})
            if final.stop_reason != "tool_use":
                _scrub_attachments(sess["messages"])
                yield _sse("done", "[DONE]")
                return
            sess["calls"] = [{"id": b.id, "name": b.name, "input": dict(b.input)} for b in final.content if b.type == "tool_use"]
            sess["idx"] = 0
            sess["outs"] = []
        calls = sess["calls"]
        while sess["idx"] < len(calls):
            call = calls[sess["idx"]]
            if _needs_approval(call):
                d = sess.pop("decision", None)
                if d is None:
                    yield _sse("approval", {"sid": sess["sid"], "name": call["name"], "input": call["input"]})
                    return
                if not d["approved"]:
                    why = d.get("reason") or "no reason given"
                    sess["outs"].append({
                        "type": "tool_result", "tool_use_id": call["id"], "is_error": True,
                        "content": f"REJECTED by the human reviewer ({why}). Nothing was changed.",
                    })
                    sess["idx"] += 1
                    continue
                if d.get("edited_data") is not None:
                    call["input"] = {**call["input"], "data": d["edited_data"]}
            yield _sse("tool", {"name": call["name"], "action": call["input"].get("action")})
            sess["outs"].append(await _exec(call, token))
            sess["idx"] += 1
        sess["messages"].append({"role": "user", "content": sess["outs"]})
        sess["calls"] = None


@app.post("/api/erp/extract")
def erp_extract(file: UploadFile = File(...), _: str = Depends(user_token)) -> dict[str, str]:
    return {"name": file.filename or "file", "text": _extract(file)[:SPEC_CHARS]}


class ErpChatReq(BaseModel):
    sid: Optional[str] = None
    message: str


@app.post("/api/erp/chat")
async def erp_chat(req: ErpChatReq, token: str = Depends(user_token)) -> StreamingResponse:
    sess = ERP_SESS.get(req.sid) if req.sid else None
    if sess is None:
        while len(ERP_SESS) >= 20:
            ERP_SESS.pop(next(iter(ERP_SESS)))
        sess = {"sid": uuid.uuid4().hex, "messages": [], "calls": None, "idx": 0, "outs": [], "steps": 0}
        ERP_SESS[sess["sid"]] = sess
    elif sess["calls"] is not None:
        raise HTTPException(status_code=409, detail="approve or reject the pending action first")
    sess["messages"].append({"role": "user", "content": req.message})
    sess["steps"] = 0

    async def gen():
        yield _sse("session", sess["sid"])
        async for e in _drive(sess, token):
            yield e

    return StreamingResponse(gen(), media_type="text/event-stream")


class ErpApprove(BaseModel):
    sid: str
    approved: bool
    edited_data: Any = None
    reason: str = ""


@app.post("/api/erp/approve")
async def erp_approve(req: ErpApprove, token: str = Depends(user_token)) -> StreamingResponse:
    sess = ERP_SESS.get(req.sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    if sess["calls"] is None:
        raise HTTPException(status_code=409, detail="nothing pending")
    sess["decision"] = {"approved": req.approved, "edited_data": req.edited_data, "reason": req.reason}

    async def gen():
        async for e in _drive(sess, token):
            yield e

    return StreamingResponse(gen(), media_type="text/event-stream")



@app.post("/api/compliance/match")
def compliance_match(file: UploadFile = File(...), _: str = Depends(user_token)) -> dict[str, Any]:
    text = _extract(file)
    texts = [c["content"] for c in processor.chunk(text, "query")][: get_params()["max_query_chunks"]]
    try:
        hits = get_store().search(texts, k=5)
    except Exception:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail="similarity search failed")
    spec_id = uuid.uuid4().hex
    _put_spec(spec_id, text)
    return {"spec_id": spec_id, "chars": len(text), "candidates": hits}


class SheetReq(BaseModel):
    spec_id: str
    product: str


SHEET_PROMPT = (
    "Below are a tender technical specification and a product's documentation.\n"
    "Build a compliance sheet: one row per requirement in the specification.\n"
    "Return ONLY a JSON array, no markdown. Each item: "
    '{{"sr":int,"requirement":str,"required":str,"offered":str,'
    '"status":"Complies"|"Partially Complies"|"Not Complies"|"Not Found","remark":str}}.\n'
    "Use only the product documentation; 'offered' must quote the actual value from it, "
    "or be empty with status Not Found when absent. Keep remark short and mention the source file.\n\n"
    "<specification>\n{spec}\n</specification>\n\n<product name=\"{name}\">\n{doc}\n</product>"
)


def _parse_rows(raw: str) -> list[dict[str, Any]]:
    a, b = raw.find("["), raw.rfind("]")
    if a < 0 or b < a:
        raise ValueError("no json array")
    return json.loads(raw[a : b + 1])


@app.post("/api/compliance/sheet")
def compliance_sheet(req: SheetReq, _: str = Depends(user_token)) -> dict[str, Any]:
    entry = SPECS.get(req.spec_id)
    if entry is None or time.time() - entry[0] > SPEC_TTL:
        SPECS.pop(req.spec_id, None)
        raise HTTPException(status_code=404, detail="spec expired, file dobara upload karo")
    spec = entry[1]
    store = get_store()
    if not store.has(req.product):
        raise HTTPException(status_code=404, detail="product not found")
    prompt = SHEET_PROMPT.format(
        spec=spec[:SPEC_CHARS], name=req.product, doc=store.get_text(req.product)[:CHAT_CHARS]
    )
    try:
        rows = _parse_rows(get_model().ask(prompt, system=SYSTEM_PROMPT))
    except Exception:
        logger.exception("compliance sheet failed")
        raise HTTPException(status_code=502, detail="could not build compliance sheet, retry")
    SPECS.pop(req.spec_id, None)  # sheet ban gayi, spec text ab nahi chahiye
    return {"product": req.product, "rows": rows}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}