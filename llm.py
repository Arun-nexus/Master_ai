import json
import logging
import os
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("llm")
BASE = Path(__file__).resolve().parent
PARAMS = tomllib.loads(Path(os.getenv("LLM_PARAMS", BASE / "params.toml")).read_text("utf-8"))
MODELS = PARAMS["models"]
AGENTS = {k: os.getenv(f"{k.upper()}_MODEL") or v for k, v in PARAMS["agents"].items()}
ERP_MODEL = AGENTS["erp"]
FAST_MODEL = AGENTS["chat"]
EXTRACT_MODEL = AGENTS["extract"]
WORKERS = int(os.getenv("BATCH_WORKERS", "8"))
RETRIES = int(os.getenv("LLM_RETRIES", "3"))
LOG_FILE = BASE / os.getenv("LLM_LOG", "llm_log.jsonl")

_local = threading.local()
_log_lock = threading.Lock()
_clients = {}
_client_lock = threading.Lock()

for _agent, _name in AGENTS.items():
    if _name not in MODELS:
        raise KeyError(f"agent '{_agent}' uses unknown model '{_name}' (see params.toml)")
    log.info("agent %-8s -> %s (%s)", _agent, _name, MODELS[_name]["provider"])


def _client(provider):
    with _client_lock:
        if provider not in _clients:
            if provider == "anthropic":
                import anthropic
                _clients[provider] = anthropic.Anthropic(max_retries=4, timeout=120)
            elif provider == "openai":
                import openai
                _clients[provider] = openai.OpenAI(max_retries=4, timeout=120)
            elif provider == "gemini":
                from google import genai
                _clients[provider] = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            else:
                raise ValueError(f"unknown provider '{provider}'")
        return _clients[provider]


def _call_text(name, args):
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "</tool_call>"


def _flatten(messages):
    out = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "assistant":
            for c in m.get("tool_calls") or []:
                f = c["function"]
                content += f"\n[called {f['name']} {json.dumps(f.get('arguments') or {}, ensure_ascii=False)}]"
        elif role == "tool":
            role, content = "user", f"[result of {m.get('name', 'tool')}]\n{content}"
        content = content.strip()
        if not content:
            continue
        if out and out[-1]["role"] == role and role != "system":
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    return out


def _anthropic_convert(messages, with_tools):
    system, out, ids, n = "", [], [], 0

    def push(role, blocks):
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            system = content
        elif role == "user":
            if content.strip():
                push("user", [{"type": "text", "text": content}])
        elif role == "assistant":
            blocks = [{"type": "text", "text": content}] if content.strip() else []
            for c in m.get("tool_calls") or []:
                f = c["function"]
                args = f.get("arguments") or {}
                if with_tools:
                    n += 1
                    ids.append(f"toolu_{n}")
                    blocks.append({"type": "tool_use", "id": f"toolu_{n}", "name": f["name"], "input": args})
                else:
                    blocks.append({"type": "text", "text": f"[called {f['name']} {json.dumps(args, ensure_ascii=False)}]"})
            if blocks:
                push("assistant", blocks)
        elif role == "tool":
            if with_tools and ids:
                push("user", [{"type": "tool_result", "tool_use_id": ids.pop(0), "content": content or "ok"}])
            else:
                push("user", [{"type": "text", "text": f"[result of {m.get('name', 'tool')}]\n{content}"}])
    return system, out


def _anthropic(spec, messages, tools, max_tokens):
    system, msgs = _anthropic_convert(messages, bool(tools))
    kw = {"model": spec["id"], "max_tokens": max_tokens, "messages": msgs}
    if "temperature" in spec:
        kw["temperature"] = spec["temperature"]
    if system:
        kw["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    if tools:
        kw["tools"] = [{
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}},
        } for t in tools]
        kw["tools"][-1]["cache_control"] = {"type": "ephemeral"}
    r = _client("anthropic").messages.create(**kw)
    parts = []
    for b in r.content:
        if b.type == "text":
            parts.append(b.text)
        elif b.type == "tool_use":
            parts.append(_call_text(b.name, b.input))
    return "\n".join(parts).strip(), r.usage.input_tokens, r.usage.output_tokens


def _openai_messages(messages):
    out, ids, n = [], [], 0
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "assistant":
            msg = {"role": "assistant", "content": content or None}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = []
                for c in calls:
                    f = c["function"]
                    n += 1
                    ids.append(f"call_{n}")
                    msg["tool_calls"].append({
                        "id": f"call_{n}", "type": "function",
                        "function": {"name": f["name"], "arguments": json.dumps(f.get("arguments") or {}, ensure_ascii=False)},
                    })
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": ids.pop(0) if ids else "call_0", "content": content or "ok"})
        else:
            out.append({"role": role, "content": content})
    return out


def _openai(spec, messages, tools, max_tokens):
    kw = {
        "model": spec["id"],
        "messages": _openai_messages(messages) if tools else _flatten(messages),
        "max_completion_tokens": max_tokens,
    }
    if "temperature" in spec:
        kw["temperature"] = spec["temperature"]
    if "reasoning_effort" in spec:
        kw["reasoning_effort"] = spec["reasoning_effort"]
    if tools:
        kw["tools"] = tools
    r = _client("openai").chat.completions.create(**kw)
    m = r.choices[0].message
    parts = [m.content] if m.content else []
    for tc in m.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except ValueError:
            args = {}
        parts.append(_call_text(tc.function.name, args))
    return "\n".join(parts).strip(), r.usage.prompt_tokens, r.usage.completion_tokens


def _gemini(spec, messages, tools, max_tokens):
    from google.genai import types
    flat = _flatten(messages)
    system = "\n".join(m["content"] for m in flat if m["role"] == "system")
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in flat if m["role"] != "system"
    ]
    cfg = {"max_output_tokens": max_tokens}
    if system:
        cfg["system_instruction"] = system
    if "temperature" in spec:
        cfg["temperature"] = spec["temperature"]
    if tools:
        cfg["tools"] = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"].get("description", ""),
                parameters_json_schema=t["function"].get("parameters") or {"type": "object", "properties": {}},
            ) for t in tools
        ])]
    r = _client("gemini").models.generate_content(
        model=spec["id"], contents=contents, config=types.GenerateContentConfig(**cfg))
    parts = []
    cand = r.candidates[0] if r.candidates else None
    for p in (cand.content.parts or []) if cand and cand.content else []:
        if getattr(p, "function_call", None):
            parts.append(_call_text(p.function_call.name, dict(p.function_call.args or {})))
        elif p.text and not getattr(p, "thought", False):
            parts.append(p.text)
    u = r.usage_metadata
    tin = (u.prompt_token_count or 0) if u else 0
    tout = ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0
    return "\n".join(parts).strip(), tin, tout


ADAPTERS = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini}


def _record(name, spec, latency, tin, tout, err):
    cost = tin * spec.get("input", 0) / 1e6 + tout * spec.get("output", 0) / 1e6
    row = {"ts": round(time.time()), "model": name, "provider": spec["provider"], "latency": round(latency, 3),
           "in": tin, "out": tout, "cost": round(cost, 6), "error": err}
    _local.last = row
    if os.getenv("LLM_LOG", "llm_log.jsonl") != "0":
        with _log_lock, open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def last_usage():
    return getattr(_local, "last", {})


def _run(name, messages, tools, max_tokens):
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}' (see params.toml)")
    spec = MODELS[name]
    for attempt in range(RETRIES + 1):
        t0, err = time.time(), ""
        try:
            text, tin, tout = ADAPTERS[spec["provider"]](spec, messages, tools, max_tokens)
            _record(name, spec, time.time() - t0, tin, tout, "")
            return text
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            _record(name, spec, time.time() - t0, 0, 0, err)
            transient = any(s in err for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded"))
            if not transient or attempt == RETRIES:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))


def chat_generate(messages, tools=None, max_new_tokens=512, model=None):
    return _run(model or FAST_MODEL, messages, tools, max_new_tokens)


def generate(prompt, max_new_tokens=300, model=None):
    return _run(model or EXTRACT_MODEL, [{"role": "user", "content": prompt}], None, max_new_tokens)


def generate_batch(prompts, max_new_tokens=300, model=None):
    if not prompts:
        return []
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(lambda p: generate(p, max_new_tokens, model), prompts))


def stats():
    if not LOG_FILE.exists():
        return {}
    agg = {}
    for line in LOG_FILE.read_text("utf-8").splitlines():
        r = json.loads(line)
        a = agg.setdefault(r["model"], {"calls": 0, "errors": 0, "lat": [], "in": 0, "out": 0, "cost": 0.0})
        a["calls"] += 1
        a["errors"] += bool(r["error"])
        a["lat"].append(r["latency"])
        a["in"] += r["in"]
        a["out"] += r["out"]
        a["cost"] += r["cost"]
    return agg


if __name__ == "__main__":
    import sys
    keys = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
    print("AGENT      MODEL                 PROVIDER   KEY")
    for a, n in AGENTS.items():
        p = MODELS[n]["provider"]
        print(f"{a:10s} {n:21s} {p:10s} {'set' if os.getenv(keys[p]) else 'MISSING'}")
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        print("\nMODEL                 calls  err  avg_s  avg_in  avg_out  cost_usd")
        for n, a in stats().items():
            c = a["calls"]
            print(f"{n:21s} {c:5d} {a['errors']:4d} {sum(a['lat']) / c:6.2f} {a['in'] // c:7d} {a['out'] // c:8d} {a['cost']:9.4f}")