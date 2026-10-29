"""ERP agent with CARDS + human-in-the-loop (HITL).

Cards: the model first sees only ONE tool, open_card(card). Once it opens a card (orders, inventory, ...),
that card's full tool (erp_orders, erp_inventory, ...) is added to its schemas. tools.py is unchanged.

HITL: every action that is not a plain view (list / get / search / track ...) is NOT executed straight away.
It becomes a pending "approval" that the UI shows to the user. The user can edit id / id2 / data and then
approve (executes with the edited values) or cancel. View actions run immediately.
Unknown actions are treated as writes (fail-safe).

Shares the single Qwen model with the tender pipeline (rag.chat_generate).
"""
import asyncio
import copy
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from auth import AuthExpired
from extract import gpu_lock
from rag import chat_generate

log = logging.getLogger("erp_agent")

BASE_DIR = Path(__file__).resolve().parent
TOOLS_FILE = BASE_DIR / "tools.py"

MAX_TOOL_ROUNDS = 6
MAX_NEW_TOKENS = int(os.getenv("ERP_MAX_NEW_TOKENS", "3072"))
FINAL_NEW_TOKENS = 300
MAX_TOOL_OUTPUT_CHARS = 6000
KEEP_USER_TURNS = 6

SERVER = {
    "ABM_ERP": {
        "transport": "stdio",
        "command": sys.executable,          # same venv as the server, no dependency on `uv` being in PATH
        "args": [str(TOOLS_FILE)],
        "cwd": str(BASE_DIR),
        "env": {**os.environ},
    }
}

# card name -> (tool name in tools.py, one-line description shown to the model)
CARDS = {
    "orders": ("erp_orders", "list/track/create/update/delete orders, confirm delivery"),
    "customers": ("erp_customers", "companies (Users page): summary table, ledger/history, search, create, credit/debit, payments"),
    "requests": ("erp_requests", "approval requests: approve, reject, partial approve"),
    "allocations": ("erp_allocations", "stock allocated to users/distributors/services, demo units, damage"),
    "inventory": ("erp_inventory", "stock lots, serial numbers, invoice lookup, repair"),
    "shipments": ("erp_shipments", "incoming shipments from suppliers, mark received, fx rate"),
    "assembly": ("erp_assembly", "assembly batches: create, edit, complete, approve"),
    "dispatch": ("erp_dispatch", "dispatch queue and confirming order/service/spare part dispatch"),
    "users": ("erp_users", "user accounts, technicians, distributors"),
    "services": ("erp_services", "service tickets, installations, warranty"),
    "work": ("erp_work", "assign work and view work/replies"),
}
TOOL_TO_CARD = {tool: card for card, (tool, _) in CARDS.items()}

# Words (English + Hinglish) that tell us which card a message is about. The matching cards are opened for the
# model before it answers, because a small model often fails at the extra open_card step.
CARD_HINTS = {
    "orders": ("order",),
    "customers": ("customer", "company", "companies", "ledger", "debit", "credit"),
    "requests": ("request",),
    "allocations": ("allocat", "demo"),
    "inventory": ("inventory", "stock", "serial", "invoice"),
    "shipments": ("shipment", "supplier"),
    "assembly": ("assembly",),
    "dispatch": ("dispatch",),
    "users": ("user", "technician", "distributor"),
    "services": ("service", "installation", "warranty"),
    "work": ("work", "task"),
}
MAX_PREOPEN = 2
BAD_REPLY_RE = re.compile(r"^\s*(<error>|error\b)", re.I)


def guess_cards(text: str) -> list:
    t = text.lower()
    return [c for c, words in CARD_HINTS.items() if any(w in t for w in words)][:MAX_PREOPEN]


# Filler words (English + Hinglish) that a small model often copies into `search`. `search` must only hold a
# real name / number (company, product, invoice no ...), so these are stripped before the tool runs.
STOP_WORDS = set("""
show display provide give get fetch list find check see view tell details detail info information data record
records all every any latest recent new the of a an me my to for with their and or is are have has which that
whose than more less above below amount amounts please pls plz mujhe de do dena dedo kar karo kro krdo batao
bata dikhao dikha dikhana sare saare sab sabhi ki ka ke k ko se mein hai hain wale wali jo aaj kal yesterday
today kitne kitni kya status
credit debit order orders inventory customer customers company companies stock shipment shipments user users
request requests service services work dispatch assembly allocation allocations
""".split())

UNIT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7,
        "thousand": 1e3, "hazar": 1e3, "hazaar": 1e3}
AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)(?:\s*(lakhs?|lacs?|crores?|thousand|hazaar|hazar)|(k)\b)?", re.I)
MORE_RE = re.compile(r"\b(more|greater|above|over|exceed\w*|zyada|jyada|upar|higher|bigger|atleast)\b|>", re.I)
LESS_RE = re.compile(r"\b(less|below|under|kam|neeche|lower|smaller)\b|<", re.I)


def clean_search(term) -> str:
    words = re.findall(r"[\w\-/.&]+", str(term or ""))
    return " ".join(w for w in words if w.lower() not in STOP_WORDS).strip()


def parse_company_filter(text: str) -> dict:
    """'debit more than 2 lakh' -> {"field": "debit", "min": 200000}. Returns {} when nothing clear is found."""
    t = text.lower()
    out = {}
    if "overdue" in t:
        out["status"] = "overdue"
    elif re.search(r"over[\s_-]*limit|limit\s*(cross|exceed)", t):
        out["status"] = "over_limit"
    m = AMOUNT_RE.search(t)
    more, less = MORE_RE.search(t), LESS_RE.search(t)
    if not m or bool(more) == bool(less):
        return out
    if re.search(r"credit\s*limit|\blimit\b", t):
        field = "credit_limit"
    elif re.search(r"outstanding|baki|balance", t):
        field = "outstanding"
    elif "debit" in t:
        field = "debit"
    elif "credit" in t:
        field = "credit"
    elif "order" in t:
        field = "total_orders"
    else:
        return out
    value = float(m.group(1).replace(",", "")) * (UNIT.get((m.group(2) or "").lower(), 1) if m.group(2) else (1e3 if m.group(3) else 1))
    out.update({"field": field, ("min" if more else "max"): int(value) if value == int(value) else value})
    return out

# View-only actions (the GET routes in tools.py). They run without approval. Everything else needs the user's OK.
READ_ACTIONS = {
    "erp_orders": {"list", "track"},
    "erp_customers": {"list", "search", "company_list", "summary", "ledger"},
    "erp_requests": {"list", "mine", "proof"},
    "erp_allocations": {"list", "damage_image"},
    "erp_inventory": {"list", "detail", "serial_history", "serial_search", "available_serials", "invoice_lookup"},
    "erp_shipments": {"list", "get", "fx_rate"},
    "erp_assembly": {"list", "get", "available_parts"},
    "erp_dispatch": {"queue"},
    "erp_users": {"list", "list_all", "technicians", "distributors"},
    "erp_services": {"list", "my", "active", "media", "available_hologram_parts", "installations", "installation_media"},
    "erp_work": {"list_all", "list_mine", "view"},
}

OPEN_CARD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "open_card",
        "description": "Open an ERP section (card) before using it. Cards: "
        + "; ".join(f"{c} = {d}" for c, (_, d) in CARDS.items()),
        "parameters": {
            "type": "object",
            "properties": {"card": {"type": "string", "enum": list(CARDS)}},
            "required": ["card"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are Kiwi, the ERP assistant of Acer Biomedicals. Never mention Qwen or Alibaba; if asked "
    "who you are, say you are Kiwi developed by Acer Biomedicals. "
    "Acer Biomedicals is a medical equipment manufacturer which is located in india state noida operates in india"
    "Never invent data. To work with the ERP first call open_card with the right card "
    "(orders, inventory, ...), then call that card's tool with an `action`. "
    "Call a tool only with the arguments its schema asks for. "
    "Any action that changes data (create, update, delete, approve, ...) is shown to the user for review and "
    "approval automatically, so do NOT ask for confirmation in text: just call the tool with complete data. "
    "If the user attaches a document (PO, invoice, challan), read its text and fill the tool data only from it; "
    "never guess missing values. Customer PO -> orders/create (check the customer with customers/search first). "
    "Supplier invoice -> inventory/create with one call per product line, or shipments/create for an incoming shipment. "
    "If a required value is missing, ask the user. Leave out fields that are empty or unknown; never fill them with "
    "placeholders like \"pending\" and never send empty lists. Keep tool arguments compact. "
    "Lists can be long: if a result says has_more or gives a total, state how many are shown out of the total and never "
    "say that is everything. If a result says a table is shown to the user, do not repeat its rows. "
    "`search` is ONLY for a specific name or number the user typed (company, product, invoice no). Never put words "
    "like show, details, provide, kar, de, all in `search`: for 'show / list all' leave `search` empty. "
    "Amount questions about companies (debit / credit more than X) use customers + action summary with "
    "data={\"field\": \"debit\", \"min\": 200000}, never `search`. "
    "For any question about ERP data you MUST call a tool; never write an <error> message yourself. "
    "After you get the tool result, answer briefly and clearly in the same language the user writes in."
)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
OPEN_CALL_RE = re.compile(r"<tool_call>.*\Z", re.DOTALL)


def to_schema(tool) -> dict:
    params = tool.args_schema
    if params is None:
        params = {"type": "object", "properties": {}}
    elif not isinstance(params, dict):
        params = params.model_json_schema()
    params = copy.deepcopy(params)
    params.get("properties", {}).pop("auth_token", None)      # the user's ERP token is never shown to the model
    if "required" in params:
        params["required"] = [r for r in params["required"] if r != "auth_token"]
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description or "", "parameters": params},
    }


def to_text(result, trim: bool = True) -> str:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, str):
        text = result
    elif isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict):
                parts.append(block.get("text") or json.dumps(block, default=str))
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    else:
        text = json.dumps(result, default=str, ensure_ascii=False)
    if trim and len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n...[output truncated]"
    return text


def repair_json(text: str):
    stack, in_str, esc, best = [], False, False, None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if ch == "}":
                best = (i + 1, list(stack))
            if not stack:
                break
    if not best:
        return None
    end, rest = best
    try:
        return json.loads(text[:end].rstrip().rstrip(",") + "".join(reversed(rest)))
    except json.JSONDecodeError:
        return None


def parse_truncated(text: str) -> list:
    if "<tool_call>" not in text:
        return []
    tail = text.rsplit("<tool_call>", 1)[1].strip()
    data = repair_json(tail)
    if not isinstance(data, dict) or not data.get("name"):
        return []
    args = data.get("arguments", {})
    return [{"name": data["name"], "arguments": args if isinstance(args, dict) else {}, "truncated": True}]


def parse_tool_calls(text: str) -> list:
    calls = []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = data.get("name")
        args = data.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return calls or parse_truncated(text)


def clean_reply(text: str) -> str:
    return OPEN_CALL_RE.sub("", TOOL_CALL_RE.sub("", THINK_RE.sub("", text))).strip()


def is_write(call: dict) -> bool:
    """True for any ERP action that is not a plain view. Unknown actions count as writes."""
    if call["name"] not in TOOL_TO_CARD:
        return False
    action = str(call["arguments"].get("action", "")).lower()
    return action not in READ_ACTIONS.get(call["name"], set())


def as_data_dict(data) -> dict:
    if isinstance(data, dict):
        return data
    if isinstance(data, str) and data.strip():
        try:
            parsed = json.loads(data)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class ERPAgent:
    def __init__(self):
        self.tools: dict = {}
        self.schemas: dict = {}
        self.sessions: dict = {}

    async def start(self):
        client = MultiServerMCPClient(SERVER)
        tools = await client.get_tools()
        self.tools = {t.name: t for t in tools}
        self.schemas = {t.name: to_schema(t) for t in tools}
        log.info("ERP agent ready: %d tools, cards: %s", len(tools), ", ".join(
            c for c, (t, _) in CARDS.items() if t in self.tools))

    # ---- session helpers ----
    def _session(self, sid: str) -> dict:
        if sid not in self.sessions:
            self.sessions[sid] = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
                "active": [],       # opened cards (kept across turns)
                "pending": None,    # {"id", "call"} waiting for the user's approval
                "queue": [],
                "table": None,
                "total": 0,
                "done": 0,
                "token": "",        # this user's ERP token (empty = shared service account, terminal client)
            }
        return self.sessions[sid]

    @staticmethod
    def _trim(s: dict):
        msgs = s["messages"]
        users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if len(users) > KEEP_USER_TURNS:
            s["messages"] = msgs[:1] + msgs[users[-KEEP_USER_TURNS]:]

    def reset(self, sid: str):
        self.sessions.pop(sid, None)

    def _active_schemas(self, s: dict) -> list:
        out = [OPEN_CARD_SCHEMA]
        for card in s["active"]:
            tool = CARDS[card][0]
            if tool in self.schemas:
                out.append(self.schemas[tool])
        return out

    # ---- execution ----
    def _open_card(self, s: dict, args: dict) -> str:
        card = str(args.get("card", "")).strip().lower()
        if card not in CARDS or CARDS[card][0] not in self.tools:
            return f"Error: unknown card '{card}'. Valid cards: {', '.join(CARDS)}"
        if card not in s["active"]:
            s["active"].append(card)
        return f"Card '{card}' opened. Now call {CARDS[card][0]} with the `action` you need."

    @staticmethod
    def _tidy(s: dict, name: str, args: dict) -> dict:
        """Fix the typical small-model mistakes in READ calls only (writes are reviewed by the user anyway)."""
        args = dict(args)
        if name not in TOOL_TO_CARD or is_write({"name": name, "arguments": args}):
            return args
        args["search"] = clean_search(args.get("search"))
        if name == "erp_customers":
            said = s.get("last_user", "")
            action = str(args.get("action", "")).lower()
            wants_money = re.search(r"debit|credit|outstanding|ledger|balance|baki|overdue", said, re.I)
            if action in ("list", "search") and not args["search"] and wants_money:
                args["action"] = action = "summary"
            if action == "summary":
                data = as_data_dict(args.get("data"))
                keys = {str(k).lower() for k in data}
                has_amount = any(k in ("field", "sort_by", "min", "max") or k.startswith(("min_", "max_"))
                                 or k.endswith(("_min", "_max")) for k in keys)
                found = parse_company_filter(said)
                if found and not has_amount:
                    args["data"] = {**data, **found}
                    args["search"] = ""         # the amount phrase is not a company name
                elif found.get("status") and "status" not in data:
                    args["data"] = {**data, "status": found["status"]}
        return args

    async def _run_call(self, s: dict, call: dict) -> str:
        name, args = call["name"], call["arguments"]
        if name == "open_card":
            return self._open_card(s, args)
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: tool '{name}' does not exist. Use open_card first."
        card = TOOL_TO_CARD.get(name)
        if card and card not in s["active"]:
            s["active"].append(card)
        args = self._tidy(s, name, args)    # copy: the token must never end up in the chat history
        if s["token"]:
            args["auth_token"] = s["token"]
        else:
            args.pop("auth_token", None)
        try:
            text = to_text(await tool.ainvoke(args), trim=False)
        except Exception as e:
            return f"Error while running '{name}': {e}"
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        searched = str(args.get("search") or "").strip()
        if searched and card and not is_write(call) and isinstance(obj, dict) and obj.get("ok") and obj.get("total") == 0:
            try:        # nothing matched the search word: show the unfiltered result and say so
                obj2 = json.loads(to_text(await tool.ainvoke({**args, "search": ""}), trim=False))
            except Exception:
                obj2 = None
            if isinstance(obj2, dict) and obj2.get("ok") and obj2.get("total"):
                msg = (f"Nothing matched the search '{searched}', so unfiltered results are shown. "
                       "Tell the user that nothing matched and these are the latest records.")
                obj2["note"] = f"{obj2.get('note', '')} {msg}".strip()
                obj, text = obj2, json.dumps(obj2, ensure_ascii=False)
        if isinstance(obj, dict) and isinstance(obj.get("table"), dict):
            s["table"] = obj.pop("table")
            text = json.dumps(obj, ensure_ascii=False)
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n...[output truncated]"
        if s["token"] and re.search(r'"status"\s*:\s*401', text):
            raise AuthExpired()         # the ERP rejected this user's token
        return text

    @staticmethod
    def _generate(messages, schemas) -> str:
        with gpu_lock:
            return chat_generate(messages, schemas, MAX_NEW_TOKENS if schemas else FINAL_NEW_TOKENS)

    @staticmethod
    def _approval(call: dict) -> dict:
        """What the UI shows: fixed tool/action, editable id / id2 / data."""
        args = call["arguments"]
        return {
            "id": uuid.uuid4().hex,
            "tool": call["name"],
            "card": TOOL_TO_CARD.get(call["name"], ""),
            "action": str(args.get("action", "")),
            "arguments": {
                "id": str(args.get("id", "") or ""),
                "id2": str(args.get("id2", "") or ""),
                "data": as_data_dict(args.get("data")),
            },
        }

    def _next_approval(self, s: dict, lead: str = "") -> dict:
        call = s["queue"].pop(0)
        approval = self._approval(call)
        approval["step"] = [s["done"] + 1, s["total"]]
        s["pending"] = {"id": approval["id"], "call": call}
        log.info("approval needed: %s %s", call["name"], approval["action"])
        if s["total"] > 1:
            note = f"Change {s['done'] + 1} of {s['total']}. Review, edit if needed, then approve."
        else:
            note = "Review this change, edit if needed, then approve."
        return {"reply": f"{lead} {note}".strip(), "approval": approval}

    async def _loop(self, s: dict, final_only: bool = False) -> dict:
        nudged = False
        for _ in range(1 if final_only else MAX_TOOL_ROUNDS):
            schemas = None if final_only else self._active_schemas(s)
            raw = await asyncio.to_thread(self._generate, s["messages"], schemas)
            calls = [] if final_only else parse_tool_calls(raw)

            if not calls and not final_only and not nudged and BAD_REPLY_RE.search(clean_reply(raw)):
                nudged = True       # the model wrote an error instead of using a tool: ask once more
                s["messages"].append({"role": "assistant", "content": clean_reply(raw)})
                s["messages"].append({"role": "user", "content": "Call the right ERP tool now with the correct action. Do not reply with an error."})
                continue

            if not calls:
                reply = clean_reply(raw) or "Done."
                s["messages"].append({"role": "assistant", "content": reply})
                return self._finish(s, reply)

            cut = any(c.pop("truncated", False) for c in calls)
            writes, seen = [], set()
            for c in calls:
                if is_write(c):
                    key = json.dumps(c, sort_keys=True, default=str)
                    if key not in seen:
                        seen.add(key)
                        writes.append(c)
            if writes:
                s["queue"], s["total"], s["done"] = writes, len(writes), 0
                lead = "Warning: the model output was cut off, so the list may be incomplete. Check every row." if cut else ""
                return self._next_approval(s, lead)

            s["messages"].append({
                "role": "assistant",
                "content": clean_reply(raw),
                "tool_calls": [{"type": "function", "function": c} for c in calls],
            })
            for call in calls:
                log.info("tool: %s %s", call["name"], json.dumps(call["arguments"], ensure_ascii=False))
                result = await self._run_call(s, call)
                s["messages"].append({"role": "tool", "name": call["name"], "content": result})

        reply = "Too many steps and the task is not finished. Please ask a shorter or clearer request."
        s["messages"].append({"role": "assistant", "content": reply})
        return self._finish(s, reply)

    @staticmethod
    def _finish(s: dict, reply: str) -> dict:
        out = {"reply": reply, "approval": None}
        if s.get("table"):
            out["table"] = s["table"]
            s["table"] = None
        return out

    async def chat(self, sid: str, text: str, token: str = "", docs: list | None = None) -> dict:
        s = self._session(sid)
        s["token"] = token
        s["table"] = None
        self._trim(s)
        if s["pending"] or s["queue"]:
            s["pending"], s["queue"] = None, []
            s["messages"].append({"role": "assistant", "content": "(The previous change was not approved and was cancelled.)"})
        content = text + "".join(f"\n\n[Attached file: {n}]\n{t}" for n, t in (docs or []))
        msg = {"role": "user", "content": content}
        s["messages"].append(msg)
        s["last_user"] = text
        for card in guess_cards(text):
            if CARDS[card][0] in self.tools and card not in s["active"]:
                s["active"].append(card)
        out = await self._loop(s)
        if docs:
            msg["content"] = text + "\n[Attached: " + ", ".join(n for n, _ in docs) + "]"
        return out

    async def approve(self, sid: str, approval_id: str, approved: bool, edits: dict | None = None,
                      token: str = "") -> dict:
        s = self.sessions.get(sid)
        pending = s["pending"] if s else None
        if not pending or pending["id"] != approval_id:
            return {"reply": "This request has expired. Please ask again.", "approval": None, "applied": False}
        s["pending"] = None
        s["token"] = token

        if not approved:
            s["queue"] = []
            s["messages"].append({"role": "assistant", "content": "The user cancelled the change. Nothing was applied."})
            return {"reply": "Cancelled. Nothing was changed.", "approval": None, "applied": False}

        call = {"name": pending["call"]["name"], "arguments": dict(pending["call"]["arguments"])}
        edits = edits or {}
        for key in ("id", "id2"):
            if key in edits:
                call["arguments"][key] = str(edits[key] or "")
        if "data" in edits:
            call["arguments"]["data"] = edits["data"]

        log.info("approved: %s %s", call["name"], json.dumps(call["arguments"], ensure_ascii=False, default=str))
        s["messages"].append({
            "role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": call}],
        })
        result = await self._run_call(s, call)
        s["messages"].append({"role": "tool", "name": call["name"], "content": result})
        ok = bool(re.search(r'"ok"\s*:\s*true', result))
        s["done"] += 1

        if ok and s["queue"]:
            out = self._next_approval(s, f"Change {s['done']} applied.")
        else:
            s["queue"] = []
            out = await self._loop(s, final_only=True)
        out["applied"] = ok
        return out