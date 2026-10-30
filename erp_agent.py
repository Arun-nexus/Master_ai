"""ERP agent with CARDS + human-in-the-loop (HITL).

Cards: the model first sees only ONE tool, open_card(card). Once it opens a card (orders, inventory, ...),
that card's full tool (erp_orders, erp_inventory, ...) is added to its schemas.

The MCP server is now erp_backend_tools.py (filter / search / sort / cards are done on the server side).
Every tool has the actions: help, filter, view, summary + page specific add / update / delete / workflow actions.

HITL: every action that is not a plain view (list / get / search / track ...) is NOT executed straight away.
It becomes a pending "approval" that the UI shows to the user. The user can edit id / id2 / data and then
approve (executes with the edited values) or cancel. View actions run immediately.
Unknown actions are treated as writes (fail-safe).

Shares the single Qwen model with the tender pipeline (rag.chat_generate).
"""
import ast
import asyncio
import copy
import difflib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from auth import AuthExpired
from extract import gpu_lock
from rag import chat_generate

log = logging.getLogger("erp_agent")

BASE_DIR = Path(__file__).resolve().parent
TOOLS_FILE = Path(os.getenv("ERP_TOOLS_FILE", str(BASE_DIR / "erp_backend_tools.py")))

MAX_TOOL_ROUNDS = 6
MAX_NEW_TOKENS = int(os.getenv("ERP_MAX_NEW_TOKENS", "3072"))
FINAL_NEW_TOKENS = 300
MAX_TOOL_OUTPUT_CHARS = 6000
MODEL_ROWS = 8            # when a table is shown to the user, the model only sees this many rows
MAX_TABLE_ROWS = 500
KEEP_USER_TURNS = 6

# ERP_MCP_URL set  -> connect to an MCP server that is already running (deployed next to the website), nothing is
#                     started locally.  e.g. ERP_MCP_URL=https://acerbiomedical.in/mcp   (optional ERP_MCP_KEY)
# ERP_MCP_URL empty -> old behaviour: start erp_backend_tools.py as a local stdio subprocess.
MCP_URL = os.getenv("ERP_MCP_URL", "").strip()
MCP_KEY = os.getenv("ERP_MCP_KEY", "").strip()
if MCP_URL:
    SERVER = {"ABM_ERP": {
        "transport": os.getenv("ERP_MCP_TRANSPORT", "streamable_http"),     # or "sse"
        "url": MCP_URL,
        **({"headers": {"Authorization": f"Bearer {MCP_KEY}"}} if MCP_KEY else {}),
    }}
else:
    SERVER = {"ABM_ERP": {
        "transport": "stdio",
        "command": sys.executable,          # same venv as the server, no dependency on `uv` being in PATH
        "args": [str(TOOLS_FILE)],
        "cwd": str(BASE_DIR),
        "env": {**os.environ, "ERP_EXPOSE_AUTH_TOKEN": "1"},   # the agent injects the user's own ERP token
    }}

# card name -> (tool name in erp_backend_tools.py, one-line description shown to the model)
CARDS = {
    "orders": ("erp_orders", "orders: filter by status/company/date, view, add, update, delete, confirm delivery"),
    "customers": ("erp_customers", "companies (Users page): debit/credit/outstanding/overdue filters, ledger, add, credit/debit, payments"),
    "requests": ("erp_requests", "approval requests: approve, reject, partial approve, proof"),
    "allocations": ("erp_allocations", "stock allocated to users/distributors/services, demo units, damage, return"),
    "inventory": ("erp_inventory", "stock lots, serial numbers, invoice lookup, repair"),
    "shipments": ("erp_shipments", "incoming shipments from suppliers, mark received, fx rate"),
    "assembly": ("erp_assembly", "assembly batches: add, update, complete, approve"),
    "dispatch": ("erp_dispatch", "dispatch page (orders, spare parts, allocations, services) and confirming a dispatch"),
    "users": ("erp_users", "user accounts, technicians, distributors"),
    "services": ("erp_services", "service tickets, status flow, spare parts, warranty"),
    "installations": ("erp_installations", "installation requests: add, assign, schedule, status"),
    "work": ("erp_work", "assign work, view work, replies"),
    "overview": ("erp_overview", "top cards of ALL pages in one call (dashboard)"),
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
    "services": ("service", "warranty", "ticket"),
    "installations": ("installation",),
    "work": ("work", "task"),
    "overview": ("overview", "dashboard"),
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

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"


def parse_when(text: str, today: date | None = None) -> dict:
    """'today' / 'aaj' / 'yesterday' / 'kal' / '23 oct' / 'oct 23 2025' / '2026-10-23' / 'this month' -> filters."""
    today = today or today_date()
    t = (text or "").lower()
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        return {"date": f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"}
    day = mon = yr = None
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?{_MON}\b(?:[\s,]*(20\d{{2}}))?", t)
    if m:
        day, mon, yr = int(m[1]), MONTHS[m[2]], m[3]
    else:
        m = re.search(rf"\b{_MON}\s*(\d{{1,2}})(?:st|nd|rd|th)?\b(?:[\s,]*(20\d{{2}}))?", t)
        if m:
            mon, day, yr = MONTHS[m[1]], int(m[2]), m[3]
    if day:
        try:
            d = date(int(yr) if yr else today.year, mon, day)
            if not yr and d > today:        # '23 oct' said in January means last year
                d = date(today.year - 1, mon, day)
            return {"date": d.isoformat()}
        except ValueError:
            return {}
    if re.search(r"\b(today|aaj|aj)\b", t):
        return {"date": today.isoformat()}
    if re.search(r"\b(yesterday|kal)\b", t):
        return {"date": (today - timedelta(days=1)).isoformat()}
    if re.search(r"this month|is mahine|is month|current month", t):
        return {"from": today.replace(day=1).isoformat(), "to": today.isoformat()}
    return {}


DATE_TOOLS = {"erp_orders", "erp_inventory", "erp_services"}           # tools whose `filters` know `date`
NEWEST_TOOLS = {"erp_orders", "erp_inventory", "erp_shipments", "erp_assembly", "erp_allocations",
                "erp_dispatch", "erp_services", "erp_requests", "erp_work"}    # tools whose `sort` knows `newest`
SUMMARY_WORDS = re.compile(r"summary|total|cards?|overview|dashboard|count|how many|kitne|kitni|stats|statistics", re.I)
LATEST_WORDS = re.compile(r"\b(latest|recent|newest|new|last|naye|naya|aakhri)\b", re.I)
# the model wrote about the call instead of making it
NARRATE_RE = re.compile(r"\berp_\w+|\bI(?:'ll| will)(?: now)? (?:call|execute|use|run|fetch)|\blet me (?:call|use|run|execute|fetch)"
                        r"|\bstep\s*1\b|\baction\s*=|\bfilters?\s*=", re.I)


def fallback_call(said: str, active: list) -> dict | None:
    """Read-only call built from the user's words, used when the model twice failed to call a tool."""
    cards = guess_cards(said) or list(active)
    if not cards:
        return None
    card = cards[0]
    tool = CARDS[card][0]
    if card == "overview":
        return {"name": tool, "arguments": {}}
    t = said.lower()
    args = {"action": "filter", "limit": 20}
    if card == "customers":
        found = parse_company_filter(said)
        if found:
            args["filters"] = found
    elif card == "inventory":
        if re.search(r"out[\s-]*of[\s-]*stock|khatam|zero stock|finished|\bout\b", t):
            args["filters"] = {"stock": "out"}
        elif re.search(r"\blow\b|\bkam\b|shortage|running out|reorder", t):
            args["filters"] = {"stock": "low"}
    return {"name": tool, "arguments": args}


# View-only actions. They run without approval. Everything else needs the user's OK (unknown = write).
COMMON_READ = {"help", "filter", "list", "view", "summary"}
READ_ACTIONS = {
    "erp_inventory": {"serial_history", "serial_search", "available_serials", "invoice_lookup"},
    "erp_shipments": {"fx_rate"},
    "erp_assembly": {"available_parts"},
    "erp_allocations": {"damage_image"},
    "erp_requests": {"proof"},
    "erp_services": {"media", "available_hologram_parts"},
    "erp_installations": {"media"},
}
READ_TOOLS = {"erp_overview"}
NAME_TOOLS = {"erp_orders", "erp_customers", "erp_requests", "erp_services", "erp_allocations", "erp_dispatch"}          # tools without an `action` that only read

# Old action names (tools.py) -> new ones (erp_backend_tools.py). A small model often still uses the old names.
ACTION_ALIASES = {
    "list": "filter", "search": "filter", "queue": "filter", "company_list": "filter", "list_all": "filter",
    "list_mine": "filter", "mine": "filter", "my": "filter", "active": "filter", "technicians": "filter",
    "distributors": "filter", "get": "view", "detail": "view", "track": "view", "ledger": "view",
    "create": "add", "edit": "update", "update_status": "status", "assign": None,
    "confirm_order": "confirm", "confirm_service": "confirm", "confirm_spare_part": "confirm",
}
ACTION_ALIASES_PER_TOOL = {          # `assign` only means something else on work / installations
    "erp_work": {"assign": "add"},
}


def normalize_call(call: dict) -> dict:
    name, args = call.get("name", ""), dict(call.get("arguments") or {})
    action = str(args.get("action", "")).strip().lower()
    if name in TOOL_TO_CARD and name not in READ_TOOLS and action:
        mapped = ACTION_ALIASES_PER_TOOL.get(name, {}).get(action) or ACTION_ALIASES.get(action)
        if mapped:
            args["action"] = mapped
            if action in ("technicians", "distributors") and name == "erp_users":
                f = as_data_dict(args.get("filters"))
                args["filters"] = {"role": action[:-1], **f}
    return {**call, "arguments": args}


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
    "Reading: `filter` lists rows (optional `search`, `filters`, `sort`, `limit`), `view` (with `id`) shows one record, "
    "`summary` gives the page's top cards, `help` shows the exact filters and `data` fields of that card. "
    "Writing: `add`, `update`, `delete` and the card's workflow actions. Before the FIRST write on a card in a "
    "conversation, call that tool with action `help` to read the exact `data` fields. "
    "Call a tool only with the arguments its schema asks for. "
    "Any action that changes data (add, update, delete, approve, ...) is shown to the user for review and "
    "approval automatically, so do NOT ask for confirmation in text: just call the tool with complete data. "
    "If the user attaches a document (PO, invoice, challan), read its text and fill the tool data only from it; "
    "never guess missing values. Customer PO -> orders/add (check the customer with customers/filter first). "
    "Supplier invoice -> inventory/add with one call per product line, or shipments/add for an incoming shipment. "
    "If a required value is missing, ask the user. Leave out fields that are empty or unknown; never fill them with "
    "placeholders like \"pending\" and never send empty lists. Keep tool arguments compact. "
    "Lists can be long: if a result gives `matched` and `has_more`, state how many are shown out of the matched count and "
    "never say that is everything. If a result says a table is shown to the user, do not repeat its rows. "
    "`search` is ONLY for a specific name or number the user typed (company, product, invoice no). Never put words "
    "like show, details, provide, kar, de, all in `search`: for 'show / list all' leave `search` empty. "
    "Use `filters` for conditions (status, dates, amounts). Amount questions about companies (debit / credit more than X) use "
    "customers + action filter with filters={\"field\": \"debit\", \"min\": 200000}, never `search`. "
    "For counts or totals per group use group_by (and sum_field). "
    "For one company's ledger: find its customer_id with customers/filter + search, then customers/view with that id. "
    "For any question about ERP data you MUST call a tool; never write an <error> message yourself. "
    "If a result has ok=false, show the `error` text to the user plainly and fix the data before retrying. "
    "After you get the tool result, answer briefly and clearly in the same language the user writes in. "
    "NEVER describe or announce a tool call in text (no 'I will now call ...', no steps): output the call itself. "
    "Examples: 'latest / recent orders' -> erp_orders action=filter, sort=newest, limit=10 (NOT summary: summary is only "
    "for totals). 'orders of 23 Oct' -> erp_orders action=filter, filters={\"date\": \"<that date as YYYY-MM-DD>\"}. "
    "'low stock items' -> erp_inventory action=filter, filters={\"stock\": \"low\"}. "
    "'companies with debit above 5 lakh' -> erp_customers action=filter, filters={\"min_debit\": 500000}."
)


def today_date() -> date:
    """Today in the company's time zone (the server may run in UTC: 'today' must still be the Indian date)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(os.getenv("ERP_TZ", "Asia/Kolkata"))).date()
    except Exception:
        return date.today()


def prompt_with_date() -> str:
    d = today_date()
    return (SYSTEM_PROMPT + f" Today is {d:%A}, {d.isoformat()} ({d:%d %B %Y}). Work out words like today, yesterday, "
            "this month, or a date without a year (e.g. '23 oct') from this date, and write dates as YYYY-MM-DD.")

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
OPEN_CALL_RE = re.compile(r"<tool_call>.*\Z", re.DOTALL)


# Copy-this-shape examples per tool. A small model follows examples far better than a description.
# Every filter key below is one the server documents (see each tool's `help`).
QUERY_EXAMPLES = {
    "erp_orders": [
        ("latest 10 orders", '{"action":"filter","sort":"newest","limit":10}'),
        ("orders of one day", '{"action":"filter","filters":{"date":"YYYY-MM-DD"}}'),
        ("pending orders", '{"action":"filter","filters":{"status":"pending"}}'),
        ("orders of a company", '{"action":"filter","filters":{"company":"<name>"}}'),
        ("biggest orders", '{"action":"filter","sort":"amount_high","limit":10}'),
        ("count per status", '{"action":"filter","group_by":"status"}'),
        ("one order", '{"action":"view","id":"<order_id>"}')],
    "erp_customers": [
        ("debit above 5 lakh", '{"action":"filter","filters":{"min_debit":500000}}'),
        ("overdue companies", '{"action":"filter","filters":{"status":"overdue"}}'),
        ("highest outstanding", '{"action":"filter","sort":"outstanding","limit":10}'),
        ("find a company", '{"action":"filter","search":"<company name>"}'),
        ("one company ledger", '{"action":"view","id":"<customer_id>"}')],
    "erp_inventory": [
        ("low stock", '{"action":"filter","filters":{"stock":"low"}}'),
        ("out of stock", '{"action":"filter","filters":{"stock":"out"}}'),
        ("spare parts", '{"action":"filter","filters":{"category":"spare_parts"}}'),
        ("find a product", '{"action":"filter","search":"<product name>"}'),
        ("one lot / serials", '{"action":"view","id":"<lot id or product_id>"}'),
        ("find a serial", '{"action":"serial_search","id":"<3+ chars of serial>"}')],
    "erp_requests": [
        ("pending requests", '{"action":"filter","filters":{"status":"pending"}}'),
        ("one type", '{"action":"filter","filters":{"type":"spare_part","status":"pending"}}'),
        ("count per type", '{"action":"filter","group_by":"type"}')],
    "erp_allocations": [
        ("overdue allocations", '{"action":"filter","filters":{"status":"overdue"}}'),
        ("demo units", '{"action":"filter","filters":{"kind":"demo_unit"}}'),
        ("by company", '{"action":"filter","filters":{"company":"<name>"}}')],
    "erp_shipments": [
        ("pending shipments", '{"action":"filter","filters":{"status":"pending"}}'),
        ("unpaid / credit", '{"action":"filter","filters":{"payment":"credit"}}'),
        ("latest", '{"action":"filter","sort":"newest","limit":10}')],
    "erp_assembly": [
        ("waiting approval", '{"action":"filter","filters":{"status":"pending_approval"}}'),
        ("completed", '{"action":"filter","filters":{"status":"completed"}}')],
    "erp_dispatch": [
        ("pending dispatch", '{"action":"filter","filters":{"state":"pending"}}'),
        ("only orders", '{"action":"filter","filters":{"kind":"order","state":"pending"}}'),
        ("dispatched today", '{"action":"filter","filters":{"dispatched_today":true}}')],
    "erp_users": [
        ("all technicians", '{"action":"filter","filters":{"role":"technician"}}'),
        ("find a user", '{"action":"filter","search":"<name or username>"}'),
        ("users per role", '{"action":"filter","group_by":"role"}')],
    "erp_services": [
        ("in-progress services", '{"action":"filter","filters":{"status":"in_progress"}}'),
        ("one technician", '{"action":"filter","filters":{"technician":"<username>"}}'),
        ("warranty expired", '{"action":"filter","filters":{"warranty":"expired"}}'),
        ("latest", '{"action":"filter","sort":"newest","limit":10}')],
    "erp_installations": [
        ("by status", '{"action":"filter","filters":{"status":"<status>"}}'),
        ("of a technician", '{"action":"filter","filters":{"assigned_to":"<username>"}}')],
    "erp_work": [
        ("open work", '{"action":"filter","filters":{"status":"open"}}'),
        ("overdue work", '{"action":"filter","filters":{"overdue":true}}'),
        ("urgent, due soon first", '{"action":"filter","filters":{"priority":"urgent"},"sort":"due_soon"}')],
}


def read_guide(help_text: str, writes: list | None = None) -> str:
    """Keep only the reading part of a tool's `help` (filter / view / summary): small enough for a 4B model."""
    out, keep = [], False
    for line in help_text.splitlines()[1:]:         # line 0 is only the title
        m = re.match(r"\s*- (\w+)", line)
        if m:
            keep = m.group(1) in ("filter", "view", "summary")
        if keep:
            out.append(line.strip())
    guide = "\n".join(out)
    if writes:
        guide += "\nWrite actions (need the user's approval): " + ", ".join(writes) + ". Before the first write, call action='help' to see its data fields."
    return guide[:1800]


def with_guide(schema: dict, guide: str) -> dict:
    fn = schema["function"]
    name = fn["name"]
    ex = "\n".join(f"- {what}: {call}" for what, call in QUERY_EXAMPLES.get(name, []))
    extra = []
    if guide:
        extra.append("HOW TO QUERY (exact names; `filters` is a JSON object, dates are YYYY-MM-DD):\n" + guide)
    if ex:
        extra.append("EXAMPLES (copy this shape):\n" + ex)
    if extra:
        fn["description"] = (fn.get("description") or "").split("\n")[0] + "\n" + "\n".join(extra)
    return schema


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
    name = call["name"]
    if name in READ_TOOLS or name not in TOOL_TO_CARD:
        return False
    action = str(call["arguments"].get("action", "")).strip().lower()
    return action not in COMMON_READ and action not in READ_ACTIONS.get(name, set())


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


MONEY_KEY_RE = re.compile(r"amount|total|debit|credit|limit|outstanding|price|value|mrp|cost|balance|paid|due|charge|gst_amount", re.I)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(_cell(x)) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


def build_table(rows: list) -> dict | None:
    """Turn the compact `rows` of a filter result into the {columns, rows} table the UI draws."""
    rows = [r for r in rows if isinstance(r, dict)]
    if len(rows) < 2:
        return None
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    columns, money = [], []
    for k in keys:
        vals = [r.get(k) for r in rows if r.get(k) not in (None, "")]
        is_money = bool(vals) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) \
            and bool(MONEY_KEY_RE.search(k))
        money.append(is_money)
        columns.append({"label": k.replace("_", " ").strip().title(), "type": "money" if is_money else "text"})
    table_rows = []
    for r in rows[:MAX_TABLE_ROWS]:
        row = []
        for k, is_money in zip(keys, money):
            v = _cell(r.get(k))
            row.append(v if is_money or isinstance(v, str) else str(v))
        table_rows.append(row)
    return {"columns": columns, "rows": table_rows}


class ERPAgent:
    def __init__(self):
        self.tools: dict = {}
        self.schemas: dict = {}
        self.sessions: dict = {}

    async def start(self):
        client = MultiServerMCPClient(SERVER)
        try:
            tools = await client.get_tools()
        except BaseException as e:      # a TaskGroup error hides the real cause: unwrap it
            leaves = []

            def walk(x):
                subs = getattr(x, "exceptions", None)
                if subs:
                    for y in subs:
                        walk(y)
                else:
                    leaves.append(f"{type(x).__name__}: {x}")
            walk(e)
            where = SERVER["ABM_ERP"].get("url") or f"{sys.executable} {TOOLS_FILE}"
            msg = f"cannot reach the MCP server ({where}): " + " | ".join(leaves or [repr(e)])
            log.error(msg)
            raise RuntimeError(msg) from None
        self.tools = {t.name: t for t in tools}
        self.schemas = {t.name: to_schema(t) for t in tools}
        await self._load_guides()
        log.info("ERP agent ready: %d tools, cards: %s", len(tools), ", ".join(
            c for c, (t, _) in CARDS.items() if t in self.tools))

    # ---- session helpers ----
    def _session(self, sid: str) -> dict:
        if sid not in self.sessions:
            self.sessions[sid] = {
                "messages": [{"role": "system", "content": prompt_with_date()}],
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
        if name not in TOOL_TO_CARD or name in READ_TOOLS or is_write({"name": name, "arguments": args}):
            return args
        said = s.get("last_user", "")
        if str(args.get("action", "")).lower() == "summary" and not SUMMARY_WORDS.search(said):
            args["action"] = "filter"       # 'show latest orders' wants rows, not the page's total cards
        if str(args.get("action", "")).lower() == "filter":
            args["search"] = clean_search(args.get("search"))
            if name in NEWEST_TOOLS and not args.get("sort") and LATEST_WORDS.search(said):
                args["sort"] = "newest"
            if LATEST_WORDS.search(said) and not args.get("limit"):
                args["limit"] = 10
            when = parse_when(said) if name in DATE_TOOLS else {}
            if "from" in when and name != "erp_orders":
                when = {}
            if when:                        # the model has no clock: it wrote 2023 for '23 oct'. Use the real date.
                flt = as_data_dict(args.get("filters"))
                for k in ("date", "from", "to"):
                    flt.pop(k, None)
                args["filters"] = {**flt, **when}
                if parse_when(str(args.get("search") or "")):
                    args["search"] = ""
        if name == "erp_customers" and str(args.get("action", "")).lower() == "filter":
            said = s.get("last_user", "")
            flt = as_data_dict(args.get("filters"))
            keys = {str(k).lower() for k in flt}
            has_amount = any(k in ("field", "sort_by", "min", "max") or k.startswith(("min_", "max_"))
                             or k.endswith(("_min", "_max")) or "__" in k for k in keys)
            found = parse_company_filter(said)
            if found and not has_amount:
                args["filters"] = {**flt, **found}
                args["search"] = ""         # the amount phrase is not a company name
            elif found.get("status") and "status" not in flt:
                args["filters"] = {**flt, "status": found["status"]}
            elif flt != args.get("filters"):
                args["filters"] = flt       # a JSON string from the model -> a real object
        return args

    async def _load_guides(self):
        """Put each tool's own filter / sort / view rules and examples into the schema the model sees, so it knows
        what query to send (the website's JS knows this; the model only saw an untyped `filters`)."""
        for card, (name, _) in CARDS.items():
            if name not in self.tools or name not in self.schemas or name == "erp_overview":
                continue
            guide = ""
            try:
                obj = json.loads(to_text(await self.tools[name].ainvoke({"action": "help"}), trim=False))
                try:
                    valid = ast.literal_eval(str(obj.get("valid_actions", "[]")))
                except Exception:
                    valid = []
                writes = [a for a in valid if a not in COMMON_READ and a not in READ_ACTIONS.get(name, set())]
                guide = read_guide(str(obj.get("help", "")), writes)
            except Exception as e:
                log.warning("no help text for %s: %s", name, e)
            with_guide(self.schemas[name], guide)
        log.info("query guides loaded for %d tools", sum(1 for n in self.schemas if "HOW TO QUERY" in self.schemas[n]["function"]["description"]))

    async def _closest_companies(self, s: dict, typed: str) -> list:
        """Company names closest to a mistyped one ('Accure Medicals'), from the customers list (cached 5 min)."""
        tool = self.tools.get("erp_customers")
        if tool is None:
            return []
        cached = s.get("company_cache")
        if not cached or time.time() - cached[0] > 300:
            args = {"action": "filter", "limit": 500}
            if s["token"]:
                args["auth_token"] = s["token"]
            try:
                obj = json.loads(to_text(await tool.ainvoke(args), trim=False))
                names = [str(r.get("company_name")) for r in obj.get("rows", []) if isinstance(r, dict) and r.get("company_name")]
            except Exception:
                names = []
            cached = s["company_cache"] = (time.time(), names)
        low = {n.lower(): n for n in cached[1]}
        hits = difflib.get_close_matches(typed.lower(), list(low), n=3, cutoff=0.5)
        if not hits:     # 'accure' vs 'acer biomedicals': try word by word
            words = [w for w in re.findall(r"[a-z0-9]{4,}", typed.lower())]
            hits = [k for k in low if any(difflib.SequenceMatcher(None, w, p).ratio() >= 0.75 for w in words for p in re.findall(r"[a-z0-9]{3,}", k))][:3]
        return [low[h] for h in hits]

    async def _run_call(self, s: dict, call: dict) -> str:
        call = normalize_call(call)         # old action names (list/create/...) -> new ones; harmless if already done
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
        reading = bool(card) and not is_write(call)
        if searched and reading and isinstance(obj, dict) and obj.get("ok") and obj.get("matched") == 0 and obj.get("total"):
            try:        # nothing matched the search word: show the unfiltered result and say so
                obj2 = json.loads(to_text(await tool.ainvoke({**args, "search": ""}), trim=False))
            except Exception:
                obj2 = None
            if isinstance(obj2, dict) and obj2.get("ok") and obj2.get("matched"):
                msg = (f"Nothing matched the search '{searched}', so unfiltered results are shown. "
                       "Tell the user that nothing matched and these are the latest records.")
                obj2["note"] = f"{obj2.get('note', '')} {msg}".strip()
                obj, text = obj2, json.dumps(obj2, ensure_ascii=False)
        if reading and isinstance(obj, dict) and obj.get("ok") and obj.get("matched") == 0 and name in NAME_TOOLS:
            flt = as_data_dict(args.get("filters"))
            typed = searched or str(flt.get("company") or flt.get("name") or "").strip()
            close = await self._closest_companies(s, typed) if typed else []
            if close:
                obj["note"] = (f"{obj.get('note', '')} No company/name matches '{typed}'. Closest names in the ERP: "
                               f"{', '.join(close)}. Ask the user which one they mean.").strip()
                text = json.dumps(obj, ensure_ascii=False)
        if isinstance(obj, dict) and isinstance(obj.get("rows"), list) and not args.get("detail") \
                and not args.get("group_by") and str(args.get("action", "")).lower() in ("filter", "list"):
            table = build_table(obj["rows"])
            if table:       # the user sees the full table; the model only gets a short preview
                s["table"] = table
                shown = len(obj["rows"])
                obj["rows"] = obj["rows"][:MODEL_ROWS]
                obj["note"] = (f"The full table of {shown} rows is shown to the user automatically. Do not list the rows; "
                               f"reply with the count ({obj.get('matched', shown)} matched out of {obj.get('total', shown)}), "
                               "totals and at most the top few.")
                text = json.dumps(obj, ensure_ascii=False)
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
        nudged = used_fb = ran_tool = False
        for _ in range(1 if final_only else MAX_TOOL_ROUNDS):
            schemas = None if final_only else self._active_schemas(s)
            raw = await asyncio.to_thread(self._generate, s["messages"], schemas)
            calls = [] if final_only else [normalize_call(c) for c in parse_tool_calls(raw)]
            content = clean_reply(raw)

            failed = (not calls and not final_only and not ran_tool
                      and (BAD_REPLY_RE.search(content) or NARRATE_RE.search(content)))
            if failed and not nudged:
                nudged = True       # the model wrote an error / a description instead of calling a tool: ask once more
                s["messages"].append({"role": "assistant", "content": content})
                s["messages"].append({"role": "user", "content": "Do not describe the call. Output the ERP tool call itself now, with the correct action."})
                continue
            if failed and nudged and not used_fb:
                fb = fallback_call(s.get("last_user", ""), s["active"])
                if fb:                  # still nothing: run the plain read call for what the user asked
                    used_fb, calls, content = True, [fb], ""
                    log.info("model did not call a tool twice; fallback call %s", json.dumps(fb, ensure_ascii=False))

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
                "content": content,
                "tool_calls": [{"type": "function", "function": c} for c in calls],
            })
            ran_tool = True
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
        s["messages"][0]["content"] = prompt_with_date()      # long-lived sessions must still know today's date
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