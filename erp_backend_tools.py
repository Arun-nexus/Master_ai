"""
erp_backend_tools.py  -  BACKEND-HEAVY MCP tools for the Acer Biomedical ERP
===========================================================================

Kyun? ERP ka frontend (orders.js, inventory.js ...) filter / search / sort / cards / warranty / overdue /
ledger ka sara kaam browser me karta hai. MCP client (Claude etc.) ke paas browser nahi hota, isliye woh
raw routes se poora dataset kheench ke khud calculate karne me atak jata hai.

Ye file wahi sara kaam SERVER SIDE karti hai:
  filter  -> search + filters + sort + pagination + cards + group_by, sab ek call me, chhota compact output
  view    -> ek record (compact summary + poora record)
  summary -> sirf page ke cards (counts / totals)
  add / update / delete + page ke workflow actions -> app.py ke EXISTING routes ko hi call karti hai

Decoupling:
  * app.py me KOI change nahi. Sirf uske routes call hote hain.
  * Default = HTTP (ERP_BASE_URL). Chaho to ERP_INPROCESS=1 karke app.py ko import karke
    in-process call karo (network nahi lagta, phir bhi app.py edit nahi hota).
  * tools.py se koi import nahi - dono alag alag chal sakti hain.

.env:
  ERP_BASE_URL, ERP_USERNAME, ERP_PASSWORD, APP_CLIENT_KEY, ERP_TIMEOUT   (tools.py jaise hi)
  ERP_TOKEN        optional ready token
  ERP_CACHE_TTL    seconds (default 20) - list routes ka chhota cache, har write par clear ho jata hai
  ERP_INPROCESS    1 => app.py ko import karke call karo
  ERP_APP_MODULE   default "app"
  ERP_APP_PATH     ERP ke app.py ka full path (master AI ke apne app.py se naam clash ho to yahi use karo)
  ERP_TOOL_PREFIX  tool naam prefix (default "erp_")      ERP_TOOLS  sirf ye pages expose karo, e.g. "orders,dispatch"
  ERP_DOC_MODE     short (default, chhote local model ke liye) | full
  ERP_EXPOSE_AUTH_TOKEN  1 (default) = tools me auth_token param (web app inject karta hai) | 0 = model se chhupao, service account use
  MCP_TRANSPORT / MCP_HOST / MCP_PORT   (stdio | http | sse ...)

Run:  python erp_backend_tools.py
"""
import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

BASE_URL = os.getenv("ERP_BASE_URL", "https://acerbiomedical.in").rstrip("/")
ERP_USERNAME = os.getenv("ERP_USERNAME", "")
ERP_PASSWORD = os.getenv("ERP_PASSWORD", "")
APP_CLIENT_KEY = os.getenv("APP_CLIENT_KEY", "").strip()
TIMEOUT = float(os.getenv("ERP_TIMEOUT", "60"))
CACHE_TTL = float(os.getenv("ERP_CACHE_TTL", "20"))
INPROCESS = os.getenv("ERP_INPROCESS", "").strip().lower() in ("1", "true", "yes")
APP_MODULE = os.getenv("ERP_APP_MODULE", "app")
APP_PATH = os.getenv("ERP_APP_PATH", "").strip()        # ERP ke app.py ka full path (naam-clash se bachne ke liye)
TOOL_PREFIX = os.getenv("ERP_TOOL_PREFIX", "erp_")        # tools.py ke saath ek hi client me chalana ho to badlo, e.g. "erp2_"
ONLY_TOOLS = {x.strip().lower() for x in os.getenv("ERP_TOOLS", "").split(",") if x.strip()}   # e.g. "orders,inventory"
DOC_MODE = os.getenv("ERP_DOC_MODE", "short").lower()     # short = chhote model (4B, 6k ctx) ke liye; full = bade model ke liye
EXPOSE_TOKEN = os.getenv("ERP_EXPOSE_AUTH_TOKEN", "1").strip().lower() in ("1", "true", "yes")

mcp = FastMCP("Acer Biomedical ERP - backend tools")

_auth = {"token": os.getenv("ERP_TOKEN", "").strip() or None}
_auth_lock = asyncio.Lock()
_HIDDEN_KEYS = {"password", "confirm_password"}


# ======================================================================================
# 1. TRANSPORT  (HTTP ya in-process, dono me app.py ke routes hi call hote hain)
# ======================================================================================
class ErpError(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(str(detail))
        self.status, self.detail = status, detail


_asgi = {"app": None}


def _make_client() -> httpx.AsyncClient:
    if INPROCESS:
        if _asgi["app"] is None:
            if APP_PATH:
                folder = os.path.dirname(os.path.abspath(APP_PATH))
                if folder not in sys.path:
                    sys.path.append(folder)            # ERP ke apne imports (order.*, user.* ...) chalne ke liye
                spec = importlib.util.spec_from_file_location("erp_app_module", APP_PATH)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                mod = importlib.import_module(APP_MODULE)
            _asgi["app"] = getattr(mod, "app")
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=_asgi["app"]),
                                 base_url="http://erp.local", timeout=TIMEOUT)
    return httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT)


async def _login(client: httpx.AsyncClient) -> str | None:
    if not ERP_USERNAME or not ERP_PASSWORD:
        return None
    headers = {"User-Agent": f"erp-mcp key={APP_CLIENT_KEY}"} if APP_CLIENT_KEY else {}
    r = await client.post("/login/", json={"username": ERP_USERNAME, "password": ERP_PASSWORD}, headers=headers)
    return r.json().get("access_token") if r.status_code == 200 else None


async def _http(method: str, path: str, params: dict | None, body: Any, token: str | None = None):
    """(status, payload). token = user ka apna token, warna service account (lazy login + 1 retry)."""
    async with _make_client() as client:
        for attempt in range(2):
            if token:
                bearer = token
            else:
                async with _auth_lock:
                    if not _auth["token"]:
                        _auth["token"] = await _login(client)
                bearer = _auth["token"]
            if not bearer:
                return 401, {"detail": "could not log in to the ERP - check ERP_USERNAME / ERP_PASSWORD"}
            try:
                r = await client.request(method, path, params=params or None, json=body,
                                         headers={"Authorization": f"Bearer {bearer}"})
            except httpx.HTTPError as e:
                return 503, {"detail": f"ERP not reachable: {e.__class__.__name__}"}
            if r.status_code == 401 and attempt == 0 and not token:
                _auth["token"] = None
                continue
            break
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"detail": r.text[:500]}


# ---- chhota TTL cache (sirf bina-params GET lists), har write par clear ----------------
_cache: dict = {}


def _invalidate():
    _cache.clear()


def _err_text(detail: Any):
    if isinstance(detail, list):
        out = []
        for e in detail:
            if isinstance(e, dict) and "msg" in e:
                loc = ".".join(str(x) for x in e.get("loc", []) if x not in ("body", "query"))
                out.append(f"{loc}: {e['msg']}" if loc else str(e["msg"]))
            else:
                out.append(str(e))
        return "; ".join(out)
    return detail


async def _get_json(path: str, token: str, params: dict | None = None):
    key = (hashlib.sha1((token or "svc").encode()).hexdigest()[:10], path)
    if not params and CACHE_TTL > 0 and key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        return 200, _cache[key][1]
    status, payload = await _http("GET", path, params, None, token or None)
    if status < 400 and not params and CACHE_TTL > 0:
        _cache[key] = (time.time(), payload)
    return status, payload


async def _dataset(path: str, token: str, params: dict | None = None) -> list:
    status, payload = await _get_json(path, token, params)
    if status >= 400:
        raise ErpError(status, _err_text(payload.get("detail") if isinstance(payload, dict) else payload))
    if isinstance(payload, dict):
        ds = payload.get("dataset")
        return ds if isinstance(ds, list) else []
    return payload if isinstance(payload, list) else []


async def _dataset_any(paths: list, token: str) -> list:
    """pehla path jo permission/route deta hai (jaise /work/all -> /work/mine fallback)."""
    last = None
    for p in paths:
        try:
            return await _dataset(p, token)
        except ErpError as e:
            last = e
            if e.status not in (403, 404):
                raise
    raise last


async def _dataset_safe(path: str, token: str) -> list:
    try:
        return await _dataset(path, token)
    except ErpError:
        return []


# ======================================================================================
# 2. SMALL HELPERS
# ======================================================================================
def _shrink(o: Any, max_list: int = 100, max_str: int = 1500):
    if isinstance(o, str):
        return f"<{len(o)} chars omitted>" if (o.startswith("data:") or len(o) > max_str) else o
    if isinstance(o, list):
        out = [_shrink(i, max_list, max_str) for i in o[:max_list]]
        if len(o) > max_list:
            out.append(f"... +{len(o) - max_list} more")
        return out
    if isinstance(o, dict):
        return {k: _shrink(v, max_list, max_str) for k, v in o.items() if k not in _HIDDEN_KEYS}
    return o


def _clean(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if not str(k).startswith("__")}


def _as_dict(data: Any):
    if data is None or data == "":
        return {}
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            raise ErpError(400, "`data` / `filters` must be a JSON object")
    if not isinstance(data, dict):
        raise ErpError(400, "`data` / `filters` must be a JSON object")
    return data


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").replace("\u20b9", "").strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def _numf(x):
    try:
        return float(str(x).replace(",", "").replace("\u20b9", "").strip())
    except (TypeError, ValueError):
        return None


def _d10(v) -> str:
    return str(v or "")[:10]


def _day(v):
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _local_day(v):
    """ISO datetime (UTC / naive) -> server-local date."""
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return (dt.astimezone() if dt.tzinfo else dt).date()
    except ValueError:
        return _day(v)


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _low(v) -> str:
    return str(v if v is not None else "").strip().lower()


def _get(row: Any, path: str):
    cur = row
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _skey(v):
    if v is None:
        return (1, 0.0, "")
    n = _numf(v) if not isinstance(v, str) or re.fullmatch(r"\s*-?[\d,]+(\.\d+)?\s*", v) else None
    return (0, n, "") if n is not None else (0, 0.0, str(v).lower())


async def _maybe(x):
    return await x if inspect.isawaitable(x) else x


_OPS = ("__gte", "__lte", "__gt", "__lt", "__ne", "__in", "__contains")


def _generic_filter(raw: dict, brief: dict, key: str, val: Any) -> bool:
    """Koi bhi field: brief key ya raw dotted path.  amount__gte, status__in, name__contains, x__ne ..."""
    op = "eq"
    for s in _OPS:
        if key.endswith(s):
            key, op = key[:-len(s)], s[2:]
            break
    cur = brief[key] if key in brief else _get(raw, key)
    if op in ("gte", "lte", "gt", "lt"):
        a, b = _numf(cur), _numf(val)
        if a is None or b is None:
            a, b = str(cur or "")[:len(str(val))], str(val)
            if not a:
                return False
        return {"gte": a >= b, "lte": a <= b, "gt": a > b, "lt": a < b}[op]
    if op == "in":
        vals = val if isinstance(val, list) else str(val).split(",")
        return _low(cur) in [_low(x) for x in vals]
    if op == "contains":
        return _low(val) in _low(json.dumps(cur, default=str))
    if op == "ne":
        return _low(cur) != _low(val)
    if isinstance(cur, list):
        return any(_low(x) == _low(val) for x in cur)
    return _low(cur) == _low(val)


# ======================================================================================
# 3. RESOURCE ENGINE
# ======================================================================================
def R(method, path, wrap=None):
    return (method, path, wrap)


class Res:
    """Ek ERP page. loader = raw rows (derived data `__x` ke saath), brief = compact row,
    hay = search text, filters = named filters, cards = page ke top cards."""

    def __init__(self, name, tool_name, loader, ids, brief, hay, doc, *, filters=None, sorts=None,
                 default_sort="", cards=None, date_of=None, routes=None, extra=None, before=None,
                 search_fn=None, norm_filters=None, detail_fn=None, alt_ids=None):
        self.name, self.tool_name, self.loader, self.ids, self.brief, self.hay, self.doc = \
            name, tool_name, loader, ids, brief, hay, doc
        self.filters = filters or {}
        self.sorts = sorts or {}
        self.default_sort = default_sort
        self.cards = cards
        self.date_of = date_of
        self.routes = routes or {}
        self.extra = extra or {}
        self.before = before or {}
        self.search_fn = search_fn
        self.norm_filters = norm_filters
        self.detail_fn = detail_fn
        self.alt_ids = alt_ids

    def actions(self):
        return ["help", "filter", "view", "summary"] + list(self.routes) + list(self.extra)


def _pairs(res: Res, raws: list):
    out = []
    for r in raws:
        b = res.brief(r)
        if "date" in b and "month" not in b:
            b["month"] = str(b["date"] or "")[:7]
        out.append((r, b))
    return out


def _apply_named(res: Res, pair, key: str, val: Any) -> bool:
    raw, brief = pair
    k = key.lower()
    if k in ("from", "date_from", "to", "date_to"):
        d = _d10(res.date_of(raw)) if res.date_of else _d10(brief.get("date"))
        if not d:
            return False
        return d >= str(val)[:10] if k.endswith("from") else d <= str(val)[:10]
    if k in res.filters:
        return bool(res.filters[k](raw, brief, val))
    return _generic_filter(raw, brief, key, val)


def _sort_pairs(res: Res, pairs: list, sort: str, order: str):
    sort = sort or res.default_sort
    if not sort:
        return pairs
    flip = _low(order) in ("desc", "reverse")
    if sort in res.sorts:
        keyfn, rev = res.sorts[sort]
        if order:
            rev = flip
    else:
        keyfn = lambda raw, b, s=sort: (b[s] if s in b else _get(raw, s))
        rev = flip
    have = [p for p in pairs if keyfn(*p) is not None]
    none = [p for p in pairs if keyfn(*p) is None]
    have.sort(key=lambda p: _skey(keyfn(*p)), reverse=rev)
    return have + none


def _group(pairs: list, group_by: str, sum_field: str):
    groups: dict = {}
    for raw, b in pairs:
        v = b[group_by] if group_by in b else _get(raw, group_by)
        key = "(none)" if v in (None, "") else (", ".join(map(str, v)) if isinstance(v, list) else str(v))
        g = groups.setdefault(key, {"key": key, "count": 0, "sum": 0.0})
        g["count"] += 1
        if sum_field:
            g["sum"] += _num(b[sum_field] if sum_field in b else _get(raw, sum_field))
    out = sorted(groups.values(), key=lambda g: -g["count"])[:60]
    for g in out:
        g["sum"] = round(g["sum"], 2) if sum_field else None
    return [{k: v for k, v in g.items() if v is not None} for g in out]


async def _do_filter(res: Res, token, search, filters, sort, order, limit, offset, group_by, sum_field, detail):
    raws = await res.loader(token)
    pairs = _pairs(res, raws)
    flt = _as_dict(filters)
    if res.norm_filters:
        flt = res.norm_filters(dict(flt))
    ranked = False
    if search and str(search).strip():
        if res.search_fn:
            order_ = res.search_fn(raws, str(search))
            idx = {id(p[0]): p for p in pairs}
            pairs = [idx[id(r)] for r in order_ if id(r) in idx]
            ranked = True
        else:
            toks = str(search).lower().split()
            pairs = [p for p in pairs if all(t in res.hay(p[0]).lower() for t in toks)]
    for k, v in flt.items():
        if v in (None, "", []):
            continue
        pairs = [p for p in pairs if _apply_named(res, p, str(k), v)]
    if not (ranked and not sort):
        pairs = _sort_pairs(res, pairs, sort, order)

    out = {"ok": True, "resource": res.name, "total": len(raws), "matched": len(pairs)}
    if res.cards:
        out["cards"] = await _maybe(res.cards(raws, token))
    if group_by:
        out["group_by"] = group_by
        out["groups"] = _group(pairs, group_by, sum_field)
        out["note"] = "aggregate only - rows dekhne ke liye group_by khali rakh ke dobara call karo"
        return out
    limit = max(0, min(int(limit or 20), 50 if detail else 200))
    offset = max(0, int(offset or 0))
    page = pairs[offset:offset + limit]
    out.update({"returned": len(page), "offset": offset, "has_more": offset + len(page) < len(pairs)})
    out["rows"] = [_shrink(_clean(p[0]), 100, 600) if detail else p[1] for p in page]
    return out


async def _do_view(res: Res, token, id, detail):
    rid = str(id or "").strip()
    if not rid:
        raise ErpError(400, "action 'view' needs `id`")
    raws = await res.loader(token)
    hit = next((r for r in raws if rid.lower() in [str(i).lower() for i in res.ids(r)]), None)
    if hit is None and res.alt_ids:
        cands = [r for r in raws if rid.lower() in [str(i).lower() for i in res.alt_ids(r)]]
        if len(cands) == 1:
            hit = cands[0]
        elif cands:
            return {"ok": True, "resource": res.name, "multiple_matches": len(cands),
                    "note": "ek se zyada record mile - inme se kisi ka `id` do",
                    "candidates": [_pairs(res, [c])[0][1] for c in cands[:20]]}
    if hit is None:
        near = [_pairs(res, [r])[0][1] for r in raws if rid.lower() in res.hay(r).lower()][:5]
        raise ErpError(404, {"message": f"{res.name}: '{rid}' not found", "similar": near})
    out = {"ok": True, "resource": res.name, "summary": _pairs(res, [hit])[0][1],
           "record": _shrink(_clean(hit), 200, 1500)}
    if res.detail_fn:
        out["extra"] = await res.detail_fn(hit, token)
    return out


async def _do_write(res: Res, action, id, id2, data, token):
    method, path, wrap = res.routes[action]
    body_dict = _as_dict(data)
    if action in res.before:
        body_dict = await res.before[action](dict(body_dict), token)
    for name, value in (("id", id), ("id2", id2)):
        if "{" + name + "}" in path:
            if not str(value or "").strip():
                raise ErpError(400, f"action '{action}' needs `{name}`")
            path = path.replace("{" + name + "}", quote(str(value).strip(), safe=""))
    params, body = None, None
    if method == "GET":
        params = body_dict or None
    elif body_dict:
        body = {wrap: body_dict} if (wrap and wrap not in body_dict) else body_dict
    elif wrap:
        raise ErpError(400, f"action '{action}' needs `data`")
    status, payload = await _http(method, path, params, body, token or None)
    if status >= 400:
        raise ErpError(status, _err_text(payload.get("detail") if isinstance(payload, dict) else payload))
    if method != "GET":
        _invalidate()
    out = _shrink(payload, 200, 2500)
    return {"ok": True, **out} if isinstance(out, dict) else {"ok": True, "result": out}


async def _handle(res: Res, action, id, id2, search, filters, sort, order, limit, offset,
                  group_by, sum_field, detail, data, token):
    try:
        if action == "help":
            return {"ok": True, "resource": res.name, "valid_actions": res.actions(), "help": res.doc.strip() + "\n" + _COMMON_DOC}
        if action in ("filter", "list"):
            return await _do_filter(res, token, search, filters, sort, order, limit, offset, group_by, sum_field, detail)
        if action == "summary":
            raws = await res.loader(token)
            return {"ok": True, "resource": res.name, "total": len(raws),
                    "cards": await _maybe(res.cards(raws, token)) if res.cards else {}}
        if action == "view":
            return await _do_view(res, token, id, detail)
        if action in res.extra:
            return await res.extra[action](res, str(id or "").strip(), str(id2 or "").strip(), _as_dict(data), token)
        if action in res.routes:
            return await _do_write(res, action, id, id2, data, token)
        return {"ok": False, "error": f"unknown action '{action}'", "valid_actions": res.actions()}
    except ErpError as e:
        return {"ok": False, "status": e.status, "error": e.detail}


_COMMON_DOC = """
Common params: id/id2 = record id | search = text search (multi-word AND) | filters = JSON object |
sort (+order asc|desc) | limit/offset (default 20, max 200) | group_by (+sum_field) = aggregate only |
detail=true => rows ka poora record | data = JSON body (add/update/workflow).
`filters` me kisi bhi row-field par: {"field": v, "field__gte": n, "field__lte": n, "field__in": "a,b",
"field__contains": "x", "field__ne": v}; date range: {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}.
Field = rows me dikhne wala naam ya raw dotted path (jaise customer.company_name). Response me `cards`
= page ke top cards. Backend errors (`error`) seedha dikhao, retry se pehle data theek karo."""


def _tool_doc(res: Res) -> str:
    if DOC_MODE == "full":
        return res.doc.strip() + "\n" + _COMMON_DOC
    first = res.doc.strip().splitlines()[0].strip().split(" (")[0][:100]
    acts = ", ".join(a for a in res.actions() if a != "help")
    return (f"{first}. Actions: {acts}, help. Filters/data fields ke liye pehle action='help'. "
            "filters={field: value}, data={...} add/update ke liye.")


def _make_tool(res: Res):
    actions = tuple(res.actions())
    name = f"{TOOL_PREFIX}{res.name}"

    if EXPOSE_TOKEN:
        async def tool(
            action: Literal[actions], id: str = "", id2: str = "", search: str = "",
            filters: Any = None, sort: str = "", order: str = "", limit: int = 20, offset: int = 0,
            group_by: str = "", sum_field: str = "", detail: bool = False, data: Any = None,
            auth_token: str = "",
        ) -> dict:
            return await _handle(res, action, id, id2, search, filters, sort, order, limit, offset,
                                 group_by, sum_field, detail, data, auth_token)
    else:
        async def tool(
            action: Literal[actions], id: str = "", id2: str = "", search: str = "",
            filters: Any = None, sort: str = "", order: str = "", limit: int = 20, offset: int = 0,
            group_by: str = "", sum_field: str = "", detail: bool = False, data: Any = None,
        ) -> dict:
            return await _handle(res, action, id, id2, search, filters, sort, order, limit, offset,
                                 group_by, sum_field, detail, data, "")

    tool.__name__ = name
    tool.__doc__ = _tool_doc(res)
    mcp.tool(name=name)(tool)
    return tool


# ======================================================================================
# 4. PAGE DEFINITIONS
# ======================================================================================
TODAY = lambda: date.today()


# -------------------------------- ORDERS (orders.js) ---------------------------------
async def _orders_loader(token):
    return list(reversed(await _dataset("/order/", token)))


def _o_company(o):
    return (o.get("customer") or {}).get("company_name") or o.get("company_name") or ""


def _o_creator(o):
    c = o.get("creator")
    if not c:
        return ""
    return f"{c.get('raised_by') or '-'} -> {c.get('approved_by') or '-'}" if c.get("type") == "request" \
        else str(c.get("created_by") or "")


def _o_total(o):
    return _num(o.get("total_mrp") if o.get("total_mrp") is not None else o.get("price"))


_STATUS_LABEL = {"placed": "Pending", "processing": "Processing", "dispatched": "Dispatched",
                 "delivered": "Delivered", "cancelled": "Cancelled", "returned": "Returned"}


def _o_brief(o):
    items = o.get("items") or []
    return {
        "order_id": o.get("order_id"), "date": _d10(o.get("order_date")), "status": o.get("status"),
        "status_label": _STATUS_LABEL.get(o.get("status"), o.get("status")),
        "company": _o_company(o), "contact": (o.get("customer") or {}).get("contractor_person") or "",
        "items": "; ".join(f"{i.get('product_name')} x{i.get('quantity')}" for i in items)[:240]
        or o.get("product_name") or "",
        "qty": sum(_num(i.get("quantity")) for i in items) if items else o.get("quantity"),
        "payment_mode": o.get("payment_mode"), "total": _o_total(o), "created_by": _o_creator(o),
    }


def _o_serials(o):
    return [s for i in o.get("items") or [] for s in i.get("serial_numbers") or []]


def _o_cards(raws, token):
    t = TODAY()
    st = lambda s: sum(1 for o in raws if o.get("status") == s)
    return {"total": len(raws), "today": sum(1 for o in raws if _day(o.get("order_date")) == t),
            "pending(placed)": st("placed"), "processing": st("processing"), "dispatched": st("dispatched"),
            "delivered": st("delivered"), "cancelled": st("cancelled"), "returned": st("returned"),
            "total_value": round(sum(_o_total(o) for o in raws), 2),
            "credit_value": round(sum(_o_total(o) for o in raws if o.get("payment_mode") == "Credit"), 2)}


async def _prep_order(d: dict, token):
    """company_name se customer_id aur item.product_name se product_id/model_no/product_type khud nikalta hai."""
    if not d.get("customer_id") and not d.get("customer"):
        name = str(d.pop("company_name", None) or d.pop("customer_name", None) or "").strip()
        if name:
            rows = await _dataset("/customer/", token)
            hits = [c for c in rows if _low(c.get("company_name")) == name.lower()] \
                or [c for c in rows if name.lower() in _low(c.get("company_name"))]
            if not hits:
                raise ErpError(400, f"no customer named '{name}' - nayi company ke liye data.customer={{company_name, ...}} do")
            if len(hits) > 1:
                raise ErpError(400, {"message": "kai companies match hui, customer_id do",
                                     "candidates": [{"customer_id": c.get("customer_id"), "company_name": c.get("company_name"),
                                                     "gst_number": c.get("gst_number")} for c in hits[:10]]})
            d["customer_id"] = hits[0].get("customer_id")
    inv = None
    for it in d.get("items") or []:
        if not it.get("product_name") or all(it.get(k) not in (None, "") for k in ("product_id", "model_no", "product_type")):
            continue
        inv = inv if inv is not None else await _dataset("/inventory/", token)
        hits = [p for p in inv if _low(p.get("product_name")) == _low(it["product_name"])
                and (not it.get("model_no") or _low(p.get("model_no")) == _low(it["model_no"]))
                and (not it.get("product_type") or (p.get("product_type") or "product") == it["product_type"])]
        if not it.get("product_type"):
            hits = [p for p in hits if (p.get("product_type") or "product") == "product"] or hits
        combos = {(p.get("product_id"), p.get("model_no"), p.get("product_type") or "product") for p in hits}
        if len(combos) > 1:
            raise ErpError(400, {"message": f"'{it['product_name']}' ke kai variants hain - model_no / product_type do",
                                 "options": [{"product_id": a, "model_no": b, "product_type": c} for a, b, c in list(combos)[:10]]})
        if combos:
            pid, model, ptype = next(iter(combos))
            it.setdefault("product_id", pid)
            if it.get("model_no") in (None, ""):
                it["model_no"] = model
            if it.get("product_type") in (None, ""):
                it["product_type"] = ptype
    return d


ORDERS = Res(
    "orders", "erp_orders", _orders_loader, lambda o: [o.get("order_id")], _o_brief,
    lambda o: " ".join([str(o.get("order_id") or ""), _o_company(o), (o.get("customer") or {}).get("contractor_person") or "",
                        *[str(i.get("product_name") or "") for i in o.get("items") or []], *map(str, _o_serials(o))]),
    """Orders page (orders.js ka sara filter/search/sort/cards yahin server side).
    - filter: filters = {status: pending|placed|processing|dispatched|delivered|cancelled|returned, payment_mode,
      date: YYYY-MM-DD, from/to, company, product, serial, created_by, min_amount, max_amount}
      sort = newest|oldest|amount_high|amount_low|order_id.  group_by e.g. status | company | payment_mode | month (sum_field=total)
    - view: id = order_id.   summary: cards only
    - add: data = {customer_id | company_name (auto-resolve) | customer:{company_name, company_address, gst_number,
      contractor_person, contractor_number, contractor_email}, items:[{product_name (product_id/model_no/product_type auto-fill
      from inventory), quantity, price, tax_rate, serial_numbers?}], payment_mode, payment_details?, discount?,
      warranty_years?, warranty_charge?, reference (username, REQUIRED)}
    - update: id + data = fields to change | delete: id (admin) | confirm_delivery: id (admin/accounts)""",
    filters={
        "status": lambda o, b, v: o.get("status") == ("placed" if _low(v) == "pending" else _low(v)),
        "payment_mode": lambda o, b, v: _low(o.get("payment_mode")) == _low(v),
        "date": lambda o, b, v: _d10(o.get("order_date")) == str(v)[:10],
        "company": lambda o, b, v: _low(v) in _low(_o_company(o)),
        "product": lambda o, b, v: any(_low(v) in _low(i.get("product_name")) for i in o.get("items") or []),
        "serial": lambda o, b, v: any(_low(v) in _low(s) for s in _o_serials(o)),
        "created_by": lambda o, b, v: _low(v) in _low(_o_creator(o)),
        "min_amount": lambda o, b, v: _o_total(o) >= _num(v),
        "max_amount": lambda o, b, v: _o_total(o) <= _num(v),
    },
    sorts={"newest": (lambda o, b: o.get("order_date") or "", True), "oldest": (lambda o, b: o.get("order_date") or "", False),
           "amount_high": (lambda o, b: _o_total(o), True), "amount_low": (lambda o, b: _o_total(o), False),
           "order_id": (lambda o, b: o.get("order_id") or "", False)},
    default_sort="newest", cards=_o_cards, date_of=lambda o: o.get("order_date"),
    routes={"add": R("POST", "/order/create_order/"), "update": R("POST", "/order/update/{id}", "updated_order_value"),
            "delete": R("POST", "/order/delete/{id}"), "confirm_delivery": R("POST", "/order/confirm_delivery/{id}")},
    before={"add": _prep_order},
)


# ------------------------------ INVENTORY (inventory.js) -----------------------------
async def _inv_loader(token):
    return list(reversed(await _dataset("/inventory/", token)))


def _qty(p):
    return _num(p.get("quantity"))


def _stock_state(p):
    q = _qty(p)
    return "out" if q == 0 else "low" if q <= 10 else "ok"


def _is_part(p):
    return p.get("product_type") in ("spare_parts", "service_parts")


def _holo_count(p):
    if isinstance(p.get("hologram_count"), (int, float)):
        return int(p["hologram_count"])
    hn = p.get("hologram_numbers")
    return len(hn) if isinstance(hn, list) else (1 if p.get("hologram_no") else 0)


def _holo_status(p):
    if not _is_part(p):
        return ""
    c, q = _holo_count(p), _qty(p)
    return "Pending" if c == 0 else "Active" if (q and c >= q) else "Partial"


def _inv_brief(p):
    return {"id": p.get("_id"), "product_id": p.get("product_id"), "product_name": p.get("product_name"),
            "model_no": p.get("model_no"), "type": p.get("product_type") or "product", "qty": _qty(p),
            "stock": _stock_state(p), "supplier": p.get("supplier"), "price": _num(p.get("price")),
            "tax_rate": p.get("tax_rate"), "lot_no": p.get("lot_no"), "date": p.get("purchase_date"),
            "invoice_no": p.get("invoice_no"), "part_category": p.get("part_category"),
            "parent_product": p.get("parent_product_name"), "hologram": _holo_status(p) or None,
            "warranty": p.get("warranty_status"), "warranty_days_left": p.get("warranty_days_left")}


_INV_FIELDS = ["product_name", "product_id", "model_no", "supplier", "invoice_no", "lot_no", "parent_product_name", "part_category"]


def _ns(s):
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", str(s if s is not None else "").lower()).strip()


def _inv_rank(rows, q):
    tokens = _ns(q).split()
    if not tokens:
        return rows
    qq, scored = " ".join(tokens), []
    for i, p in enumerate(rows):
        hay = "|" + "|".join(_ns(p.get(f)) for f in _INV_FIELDS)
        hayc, name, ids = re.sub(r"[ |]", "", hay), _ns(p.get("product_name")), [_ns(p.get("product_id")), _ns(p.get("model_no"))]
        score, ok = 0.0, True
        for t in tokens:
            if t in hay:
                score += 3 if ("|" + t in hay or " " + t in hay) else 1
            elif t in hayc:
                score += 0.5
            else:
                ok = False
                break
        if not ok:
            continue
        score += 10 if name == qq else 5 if name.startswith(qq) else 0
        score += 8 if qq in ids else 0
        scored.append((score, i, p))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [p for _, _, p in scored]


def _inv_cards(raws, token):
    by_type, parts = {}, {}
    for p in raws:
        t = p.get("product_type") or "product"
        by_type[t] = by_type.get(t, 0) + 1
        if _is_part(p):
            s = parts.setdefault(t, {"active": 0, "partial": 0, "pending": 0})
            s[_holo_status(p).lower()] += 1
    return {"total_products": len(raws), "total_stock": sum(_qty(p) for p in raws),
            "low_stock(<=10)": sum(1 for p in raws if _qty(p) <= 10), "out_of_stock": sum(1 for p in raws if _qty(p) == 0),
            "suppliers": len({p.get("supplier") for p in raws if p.get("supplier")}),
            "inventory_value": round(sum(_num(p.get("price")) * _qty(p) for p in raws), 2),
            "by_type": by_type, "parts_hologram_status": parts}


async def _inv_detail(p, token):
    params = {k: p.get(k) for k in ("product_id", "model_no", "product_type") if p.get(k)}
    if p.get("lot_no"):
        params["lot_id"] = p["lot_no"]
    status, payload = await _http("GET", "/inventory/detail", params, None, token or None)
    return _shrink(payload, 300, 600) if status < 400 else {"detail_unavailable": _err_text(payload.get("detail") if isinstance(payload, dict) else payload)}


async def _inv_serial_history(res, id, id2, data, token):
    key = id or str(data.get("serial") or "")
    if not key:
        raise ErpError(400, "serial_history needs `id` = serial number")
    status, payload = await _http("GET", f"/inventory/serial_history/{quote(key.strip(), safe='')}", None, None, token or None)
    if status >= 400:
        raise ErpError(status, _err_text(payload.get("detail") if isinstance(payload, dict) else payload))
    return {"ok": True, **_shrink(payload, 200, 1200)} if isinstance(payload, dict) else {"ok": True, "result": _shrink(payload)}


async def _inv_get_route(path, params, token):
    status, payload = await _http("GET", path, params, None, token or None)
    if status >= 400:
        raise ErpError(status, _err_text(payload.get("detail") if isinstance(payload, dict) else payload))
    return {"ok": True, **_shrink(payload, 200, 1200)} if isinstance(payload, dict) else {"ok": True, "result": _shrink(payload)}


async def _inv_serial_search(res, id, id2, data, token):
    return await _inv_get_route("/inventory/serial_search", {"q": id or data.get("q") or ""}, token)


async def _inv_available_serials(res, id, id2, data, token):
    return await _inv_get_route("/inventory/available_serials", {"product_id": "", **data}, token)


async def _inv_invoice_lookup(res, id, id2, data, token):
    return await _inv_get_route("/invoice/lookup", {"invoice_no": id or data.get("invoice_no") or ""}, token)


INVENTORY = Res(
    "inventory", "erp_inventory", _inv_loader, lambda p: [p.get("_id")], _inv_brief,
    lambda p: " ".join(str(p.get(f) or "") for f in _INV_FIELDS),
    """Inventory page. Stock = product_name + model_no + product_type same hone par ek hi product.
    - filter: search (ranked, inventory page jaisa) | filters = {category|product_type: product|spare_parts|service_parts|damaged|accessories,
      stock: ok|low|out, low_stock: true, supplier, name, product_id, date, hologram: Active|Partial|Pending,
      warranty: "under warranty"|"over warranty", part_category, min_qty, max_qty}
      sort = qty_high|qty_low|name|value_high|newest.  group_by e.g. type | supplier | stock (sum_field=qty)
    - view: id = lot id (`id` column) ya product_id -> serial numbers wagairah ke saath
    - add: data = {product_name, product_id (blank ok), quantity, purchase_date, lot_no, supplier, price, tax_rate, model_no,
      product_type, serial_numbers, invoice_no, payment_mode, ...}
    - update: id + data = {updated_values:{...}, new_serial_numbers, remove_serial_numbers, faulty_serial_numbers,
      new_hologram_numbers, remove_hologram_numbers, source_product_type, model_no}
    - delete: id (admin) | repair: id of damaged lot
    - serial_history: id = serial | serial_search: id = 3+ chars | available_serials: data = {product_id, model_no, product_name,
      product_type} | invoice_lookup: id = invoice no""",
    filters={
        "category": lambda p, b, v: b["type"] == _low(v), "product_type": lambda p, b, v: b["type"] == _low(v),
        "stock": lambda p, b, v: b["stock"] == _low(v),
        "low_stock": lambda p, b, v: (_qty(p) <= 10) == _truthy(v),
        "supplier": lambda p, b, v: _low(v) in _low(p.get("supplier")),
        "name": lambda p, b, v: _low(v) in _low(p.get("product_name")),
        "product_id": lambda p, b, v: _low(v) in _low(p.get("product_id")),
        "date": lambda p, b, v: p.get("purchase_date") == str(v)[:10],
        "hologram": lambda p, b, v: _low(_holo_status(p)) == _low(v),
        "warranty": lambda p, b, v: _low(p.get("warranty_status")) == _low(v),
        "part_category": lambda p, b, v: _low(p.get("part_category")) == _low(v),
        "min_qty": lambda p, b, v: _qty(p) >= _num(v), "max_qty": lambda p, b, v: _qty(p) <= _num(v),
    },
    sorts={"qty_high": (lambda p, b: _qty(p), True), "qty_low": (lambda p, b: _qty(p), False),
           "name": (lambda p, b: _low(p.get("product_name")), False),
           "value_high": (lambda p, b: _num(p.get("price")) * _qty(p), True),
           "newest": (lambda p, b: p.get("purchase_date") or "", True)},
    cards=_inv_cards, date_of=lambda p: p.get("purchase_date"), search_fn=_inv_rank, detail_fn=_inv_detail,
    alt_ids=lambda p: [p.get("product_id"), p.get("lot_no")],
    routes={"add": R("POST", "/inventory/create"), "update": R("POST", "/inventory/update/{id}", "updated_values"),
            "delete": R("POST", "/inventory/delete/{id}"), "repair": R("POST", "/inventory/repair/{id}")},
    extra={"serial_history": _inv_serial_history, "serial_search": _inv_serial_search,
           "available_serials": _inv_available_serials, "invoice_lookup": _inv_invoice_lookup},
)


# ------------------------------ SHIPMENTS (shipment.js) ------------------------------
async def _ship_loader(token):
    return list(await _dataset("/shipment/", token))


def _s_status(s):
    return "received" if s.get("received_date") else "pending"


def _s_total(s):
    if s.get("total_inr") not in (None, ""):
        return _num(s.get("total_inr"))
    base = sum(_num(p.get("price")) * _num(p.get("quantity")) for p in s.get("products") or [])
    return round(base * (_num(s.get("exchange_rate")) or 1), 2)


def _s_parts(s):
    return [{"product": p.get("product_name"), "name": pt.get("part_name"), "qty": pt.get("quantity"),
             "status": pt.get("status") or "assembly"} for p in s.get("products") or [] for pt in p.get("parts") or []]


def _s_brief(s):
    return {"shipment_id": s.get("shipment_id") or s.get("id"), "date": _d10(s.get("dispatch_date")),
            "received_date": _d10(s.get("received_date")) or None, "status": _s_status(s),
            "company": s.get("company_name"), "invoice_no": s.get("invoice_no"),
            "products": "; ".join(f"{p.get('product_name')} x{p.get('quantity')}" for p in s.get("products") or [])[:240],
            "parts": len(_s_parts(s)), "payment_mode": s.get("payment_mode") or "paid",
            "currency": s.get("currency") or "INR", "total_inr": _s_total(s),
            "paid_amount": s.get("paid_amount"), "credit_days": s.get("credit_days"), "due_date": s.get("due_date")}


def _fy_range():
    n = date.today()
    y = n.year if n.month >= 4 else n.year - 1
    return date(y, 4, 1), date(y + 1, 3, 31)


def _s_cards(raws, token):
    t, (fs, fe) = TODAY(), _fy_range()
    made = lambda s: _day(s.get("created_at") or s.get("dispatch_date"))
    return {"this_month": sum(1 for s in raws if made(s) and (made(s).year, made(s).month) == (t.year, t.month)),
            "today": sum(1 for s in raws if _d10(s.get("created_at")) == t.isoformat()),
            "received": sum(1 for s in raws if _s_status(s) == "received"),
            "pending": sum(1 for s in raws if _s_status(s) == "pending"),
            "financial_year": sum(1 for s in raws if made(s) and fs <= made(s) <= fe),
            "total_value_inr": round(sum(_s_total(s) for s in raws), 2),
            "pending_value_inr": round(sum(_s_total(s) for s in raws if _s_status(s) == "pending"), 2)}


SHIPMENTS = Res(
    "shipments", "erp_shipments", _ship_loader, lambda s: [s.get("shipment_id") or s.get("id")], _s_brief,
    lambda s: " ".join([str(s.get("shipment_id") or ""), str(s.get("company_name") or ""), str(s.get("company_address") or ""),
                        str(s.get("invoice_no") or ""), *[str(p.get("product_name") or "") for p in s.get("products") or []],
                        *[str(x["name"] or "") for x in _s_parts(s)]]),
    """Shipment page (supplier se aane wala stock).
    - filter: filters = {status: received|pending, payment: paid|partial|credit, currency, company, part_status:
      assembly|purchase|warranty, invoice_no, from/to (dispatch_date), min_amount, max_amount}
      sort = newest|oldest|amount_high|amount_low.  group_by e.g. status | company | payment_mode | month (sum_field=total_inr)
    - view: id = shipment_id | summary: cards (this month / today / received / pending / FY / values)
    - add: data = {company_name, dispatch_date, products:[{product_name, quantity, price, warranty,
      parts:[{part_name, quantity, status: assembly|purchase|warranty}]}], company_address, invoice_no, received_date,
      customer_id, gst_number, payment_mode (paid|partial|credit), paid_amount, credit_days, currency, exchange_rate}
    - update: id + data | mark_received: id, data = {received_date} | delete: id (admin) | fx_rate: data = {currency}""",
    filters={
        "status": lambda s, b, v: _s_status(s) == _low(v),
        "payment": lambda s, b, v: _low(s.get("payment_mode") or "paid") == _low(v),
        "currency": lambda s, b, v: _low(s.get("currency") or "INR") == _low(v),
        "company": lambda s, b, v: _low(v) in _low(s.get("company_name")),
        "part_status": lambda s, b, v: any(_low(x["status"]) == _low(v) for x in _s_parts(s)),
        "invoice_no": lambda s, b, v: _low(v) in _low(s.get("invoice_no")),
        "min_amount": lambda s, b, v: _s_total(s) >= _num(v), "max_amount": lambda s, b, v: _s_total(s) <= _num(v),
    },
    sorts={"newest": (lambda s, b: s.get("created_at") or s.get("dispatch_date") or "", True),
           "oldest": (lambda s, b: s.get("created_at") or s.get("dispatch_date") or "", False),
           "amount_high": (lambda s, b: _s_total(s), True), "amount_low": (lambda s, b: _s_total(s), False)},
    default_sort="newest", cards=_s_cards, date_of=lambda s: s.get("dispatch_date"),
    routes={"add": R("POST", "/shipment/create"), "update": R("POST", "/shipment/update/{id}", "updated_values"),
            "delete": R("POST", "/shipment/delete/{id}"), "mark_received": R("POST", "/shipment/mark_received/{id}"),
            "fx_rate": R("GET", "/shipment/fx_rate")},
)


# ------------------------------ ASSEMBLY (assembly.js) -------------------------------
async def _asm_loader(token):
    return list(await _dataset("/assembly/", token))


def _a_source(a):
    inv = any(p.get("source") != "local" for p in a.get("parts_used") or [])
    loc = any(p.get("source") == "local" for p in a.get("parts_used") or [])
    return "both" if inv and loc else "inventory" if inv else "local" if loc else "none"


_A_LABEL = {"pending": "Pending", "pending_approval": "Awaiting Approval", "partially_approved": "Partially Approved",
            "completed": "Fully Approved"}


def _a_brief(a):
    return {"assembly_id": a.get("assembly_id"), "date": _d10(a.get("created_at")), "product_name": a.get("product_name"),
            "product_id": a.get("product_id"), "model_number": a.get("model_number"), "qty": a.get("quantity"),
            "status": a.get("status"), "status_label": _A_LABEL.get(a.get("status"), a.get("status")),
            "serials": len(a.get("serials") or []), "approved": len(a.get("approved_serials") or []),
            "source": _a_source(a),
            "parts": "; ".join(f"{p.get('part_name')} x{p.get('quantity')} ({p.get('source')})" for p in a.get("parts_used") or [])[:240]}


async def _asm_cards(raws, token):
    t = TODAY()
    parts = await _dataset_safe("/assembly/available_parts", token)
    return {"this_month": sum(1 for a in raws if _day(a.get("created_at")) and (_day(a["created_at"]).year, _day(a["created_at"]).month) == (t.year, t.month)),
            "today": sum(1 for a in raws if _d10(a.get("created_at")) == t.isoformat()),
            "pending": sum(1 for a in raws if a.get("status") == "pending"),
            "awaiting_approval": sum(1 for a in raws if a.get("status") in ("pending_approval", "partially_approved")),
            "fully_approved": sum(1 for a in raws if a.get("status") == "completed"),
            "available_parts_in_inventory": len(parts)}


async def _asm_available_parts(res, id, id2, data, token):
    rows = await _dataset("/assembly/available_parts", token)
    q = _low(id or data.get("search"))
    if q:
        rows = [r for r in rows if q in _low(json.dumps(r, default=str))]
    return {"ok": True, "total": len(rows), "dataset": _shrink(rows[:int(data.get("limit") or 50)], 100, 300)}


ASSEMBLY = Res(
    "assembly", "erp_assembly", _asm_loader, lambda a: [a.get("assembly_id")], _a_brief,
    lambda a: " ".join([str(a.get("assembly_id") or ""), str(a.get("product_name") or ""), str(a.get("product_id") or ""),
                        str(a.get("model_number") or ""),
                        *[f"{p.get('part_name')} {p.get('belongs_to') or ''}" for p in a.get("parts_used") or []],
                        *[f"{s.get('serial_number')} {s.get('hologram_number') or ''}" for s in a.get("serials") or []]]),
    """Assembly page. Flow: add -> mark_completed -> approve.
    - filter: filters = {status: pending|pending_approval|partially_approved|completed, source: inventory|local|both,
      product, serial, from/to (created date)}; sort = newest|oldest; group_by e.g. status | product_name | month (sum_field=qty)
    - view: id = assembly_id (serial/hologram list ke saath) | summary: cards
    - add: data = {product_name, product_id, model_number, quantity, parts_used:[{part_name, quantity, source:
      inventory|local, belongs_to}], serials:[{serial_number, lot_no}]} (har unit ka ek serial, lot_no zaroori)
    - update: id + data = {product_name, product_id, model_number, parts_used} (sirf pending me)
    - mark_completed: id | approve: id, data = {serial_numbers:[...]} | delete: id (admin)
    - available_parts: inventory ke spare parts jo assembly me lag sakte hain (id = optional text filter)""",
    filters={
        "status": lambda a, b, v: a.get("status") == _low(v),
        "source": lambda a, b, v: _a_source(a) == _low(v),
        "product": lambda a, b, v: _low(v) in _low(a.get("product_name")),
        "serial": lambda a, b, v: any(_low(v) in _low(s.get("serial_number")) or _low(v) in _low(s.get("hologram_number"))
                                      for s in a.get("serials") or []),
    },
    sorts={"newest": (lambda a, b: a.get("created_at") or "", True), "oldest": (lambda a, b: a.get("created_at") or "", False)},
    default_sort="newest", cards=_asm_cards, date_of=lambda a: a.get("created_at"),
    routes={"add": R("POST", "/assembly/create"), "update": R("POST", "/assembly/edit/{id}"),
            "mark_completed": R("POST", "/assembly/mark_completed/{id}"), "approve": R("POST", "/assembly/approve/{id}"),
            "delete": R("POST", "/assembly/delete/{id}")},
    extra={"available_parts": _asm_available_parts},
)


# ----------------------------- ALLOCATIONS (allocated.js) ----------------------------
async def _alloc_loader(token):
    out = []
    now = datetime.now()
    for a in await _dataset("/allocation/", token):
        a = dict(a)
        kind = "spare_part" if a.get("allocation_type") == "spare_part" else "demo_unit" if a.get("allocation_type") == "demo_unit" else "product"
        if a.get("return_status") == "returned":
            status, left = "returned", None
        else:
            try:
                due = datetime.fromisoformat(str(a.get("return_due_date")).replace("Z", "+00:00"))
                due = due.replace(tzinfo=None) if due.tzinfo is None else due.astimezone().replace(tzinfo=None)
                secs = (due - now).total_seconds()
                status, left = ("overdue", None) if secs <= 0 else ("pending", int(-(-secs // 86400)))
            except ValueError:
                status, left = "pending", None
        who = f"Service #{str((a.get('spare_part') or {}).get('service_id') or '')[:8]}" if kind == "spare_part" \
            else (a.get("sales_person") or {}).get("name") or ""
        a["__x"] = {"kind": kind, "status": status, "days_left": left, "who": who}
        out.append(a)
    return out


def _al_product(a):
    if a["__x"]["kind"] == "spare_part":
        sp = a.get("spare_part") or {}
        return f"{sp.get('part_name') or ''} x{sp.get('quantity') or 1}".strip()
    return ", ".join(f"{i.get('product_name') or ''} x{i.get('quantity') or 1}" for i in a.get("items") or [])


def _al_serials(a):
    return [] if a["__x"]["kind"] == "spare_part" else [s for i in a.get("items") or [] for s in i.get("serial_numbers") or []]


def _al_brief(a):
    x = a["__x"]
    return {"allocation_id": a.get("allocation_id"), "kind": x["kind"], "product": _al_product(a)[:200],
            "serials": ", ".join(map(str, _al_serials(a)))[:200], "who": x["who"], "company": a.get("company_name"),
            "date": _d10(a.get("allotment_date")), "return_due": _d10(a.get("return_due_date")),
            "status": x["status"], "days_left": x["days_left"]}


def _al_cards(raws, token):
    c = lambda s: sum(1 for a in raws if a["__x"]["status"] == s)
    return {"total": len(raws), "pending": c("pending"), "overdue": c("overdue"), "returned": c("returned"),
            "demo_units": sum(1 for a in raws if a["__x"]["kind"] == "demo_unit"),
            "spare_parts": sum(1 for a in raws if a["__x"]["kind"] == "spare_part"),
            "products": sum(1 for a in raws if a["__x"]["kind"] == "product")}


ALLOCATIONS = Res(
    "allocations", "erp_allocations", _alloc_loader, lambda a: [a.get("allocation_id")], _al_brief,
    lambda a: " ".join([str(a.get("allocation_id") or ""), str(a.get("company_name") or ""), str(a.get("address") or ""),
                        a["__x"]["who"], _al_product(a), *map(str, _al_serials(a))]),
    """Allocated page (users / distributors / services ko diya stock). Status server side nikalta hai: pending / overdue / returned.
    - filter: filters = {kind: product|demo_unit|spare_part, status: pending|overdue|returned, who (sales person / service),
      company, product, serial, from/to (allotment date)}; sort = newest|oldest|due_soon;
      group_by e.g. status | kind | who | company (sum_field optional)
    - view: id = allocation_id | summary: cards
    - add: data = {allocated_to (username), items:[{product_id, product_name, quantity, model_no, product_type, serial_numbers}],
      company_name, address, gst_number, phone_number} YA spare part: {spare_part:{service_id, part_name, quantity, ...}}
    - update: id + data = {serial_numbers, company_name, address, gst_number, phone_number, remarks}
    - return: id | send_to_dispatch: id | report_damage: id, data = {issue, image(base64)} | damage_image: id
    - convert_to_order: id, data = {customer_id | company_name/company_address/gst_number, price, tax_rate, invoice_no,
      invoice_date, payment_mode, payment_details}
    - create_demo_unit / assign_demo / reassign_demo (id): demo unit flows | delete: nahi hota (return use karo)""",
    filters={
        "kind": lambda a, b, v: a["__x"]["kind"] == _low(v), "status": lambda a, b, v: a["__x"]["status"] == _low(v),
        "who": lambda a, b, v: _low(v) in _low(a["__x"]["who"]),
        "company": lambda a, b, v: _low(v) in _low(a.get("company_name")),
        "product": lambda a, b, v: _low(v) in _low(_al_product(a)),
        "serial": lambda a, b, v: any(_low(v) in _low(s) for s in _al_serials(a)),
    },
    sorts={"newest": (lambda a, b: a.get("allotment_date") or "", True), "oldest": (lambda a, b: a.get("allotment_date") or "", False),
           "due_soon": (lambda a, b: (a.get("return_due_date") or "") if a["__x"]["status"] != "returned" else None, False)},
    default_sort="newest", cards=_al_cards, date_of=lambda a: a.get("allotment_date"),
    routes={"add": R("POST", "/allocation/create"), "update": R("POST", "/allocation/edit/{id}"),
            "return": R("POST", "/allocation/return/{id}"), "send_to_dispatch": R("POST", "/allocation/send_to_dispatch/{id}"),
            "report_damage": R("POST", "/allocation/report_damage/{id}"), "damage_image": R("GET", "/allocation/{id}/damage_image"),
            "convert_to_order": R("POST", "/allocation/convert_to_order_direct/{id}"),
            "create_demo_unit": R("POST", "/allocation/create_demo_unit"), "assign_demo": R("POST", "/allocation/assign_demo"),
            "reassign_demo": R("POST", "/allocation/reassign_demo/{id}")},
)


# ------------------------------- DISPATCH (dispatch.js) ------------------------------
_DISPATCH_KEYS = [("pending_orders", "order", False), ("dispatched_orders", "order", True),
                  ("pending_spare_parts", "spare_part", False), ("dispatched_spare_parts", "spare_part", True),
                  ("pending_product_allocations", "product_allocation", False),
                  ("dispatched_product_allocations", "product_allocation", True),
                  ("pending_services", "service", False), ("dispatched_services", "service", True)]


async def _disp_loader(token):
    status, payload = await _get_json("/dispatch/", token)
    if status >= 400:
        raise ErpError(status, _err_text(payload.get("detail") if isinstance(payload, dict) else payload))
    out = []
    for key, kind, _ in _DISPATCH_KEYS:
        for d in payload.get(key) or []:
            d = dict(d)
            disp = d.get("dispatch")
            d["__x"] = {"kind": kind, "state": "pending" if not disp else "dispatched" if disp.get("docket_no") else "in_progress"}
            out.append(d)
    return out


def _d_id(d):
    return d.get("order_id") if d["__x"]["kind"] == "order" else d.get("service_id") if d["__x"]["kind"] == "service" else d.get("allocation_id")


def _d_product(d):
    k = d["__x"]["kind"]
    if k == "spare_part":
        sp = d.get("spare_part") or {}
        return f"{sp.get('part_name') or ''} x{sp.get('quantity') or 1}"
    if k == "service":
        return str(d.get("product_name") or d.get("product_id") or "")
    return ", ".join(f"{i.get('product_name') or ''} x{i.get('quantity') or 1}" for i in d.get("items") or [])


def _d_serials(d):
    return [] if d["__x"]["kind"] == "spare_part" else [s for i in d.get("items") or [] for s in i.get("serial_numbers") or []]


def _d_brief(d):
    disp, x = d.get("dispatch") or {}, d["__x"]
    return {"id": _d_id(d), "kind": x["kind"], "state": x["state"], "product": _d_product(d)[:200],
            "company": (d.get("customer") or {}).get("company_name") or d.get("company_name") or "",
            "invoice_no": disp.get("invoice_no"), "docket_no": disp.get("docket_no"),
            "courier": disp.get("dispatched_through"), "mode": disp.get("mode_of_delivery"),
            "dispatched_at": _d10(disp.get("dispatched_at")) or None, "date": _d10(disp.get("dispatched_at") or d.get("order_date") or d.get("allotment_date") or d.get("created_at"))}


def _d_cards(raws, token):
    t = TODAY()
    npend = lambda k=None: sum(1 for d in raws if d["__x"]["state"] != "dispatched" and (k is None or d["__x"]["kind"] == k))
    return {"total_pending": npend(), "orders_pending": npend("order"), "spare_pending": npend("spare_part"),
            "product_pending": npend("product_allocation"), "service_pending": npend("service"),
            "dispatched_today": sum(1 for d in raws if (d.get("dispatch") or {}).get("dispatched_at")
                                    and _local_day(d["dispatch"]["dispatched_at"]) == t),
            "total_dispatched": sum(1 for d in raws if d.get("dispatch"))}


async def _disp_confirm(res, id, id2, data, token):
    if not id:
        raise ErpError(400, "confirm needs `id` (order_id / service_id / allocation_id)")
    row = next((d for d in await _disp_loader(token) if _d_id(d) == id), None)
    if row is None:
        raise ErpError(404, f"'{id}' dispatch queue me nahi mila")
    for need in ("invoice_no", "invoice_date"):
        if not str(data.get(need) or "").strip():
            raise ErpError(400, f"data.{need} is required")
    body = dict(data)
    body["invoice_no"] = str(body["invoice_no"]).strip().lower()
    if body.get("docket_no"):
        body["docket_no"] = str(body["docket_no"]).strip().lower()
    kind = row["__x"]["kind"]
    path = {"order": "/dispatch/confirm/order/", "service": "/dispatch/confirm/service/"}.get(kind, "/dispatch/confirm/spare_part/")
    status, payload = await _http("POST", path + quote(id, safe=""), None, body, token or None)
    if status >= 400:
        raise ErpError(status, _err_text(payload.get("detail") if isinstance(payload, dict) else payload))
    _invalidate()
    return {"ok": True, "kind": kind, **(_shrink(payload) if isinstance(payload, dict) else {"result": payload})}


DISPATCH = Res(
    "dispatch", "erp_dispatch", _disp_loader, lambda d: [_d_id(d)], _d_brief,
    lambda d: " ".join([str(_d_id(d) or ""), str((d.get("dispatch") or {}).get("docket_no") or ""),
                        str((d.get("dispatch") or {}).get("invoice_no") or ""),
                        str((d.get("dispatch") or {}).get("dispatched_through") or ""), _d_product(d), *map(str, _d_serials(d))]),
    """Dispatch page - orders + spare parts + product allocations + services ek hi list me.
    - filter: filters = {kind|type: order|spare_part|product_allocation|service, state|status: pending|in_progress|dispatched,
      invoice_no, docket_no, courier, product, serial, company, dispatched_today: true, from/to (dispatch date)}
      state: pending = abhi dispatch nahi, in_progress = invoice hai docket nahi, dispatched = docket hai.
      group_by e.g. kind | state | courier
    - view: id | summary: cards (pending per type, dispatched today, total dispatched)
    - confirm: id = order_id / service_id / allocation_id (kind khud pata chalta hai), data = {invoice_no (required),
      invoice_date (required), docket_no, mode_of_delivery, dispatched_through, ship_to_different: bool,
      ship_to_address:{company_name, address}, image (optional base64)}   (add/update/delete is page par nahi hote)""",
    filters={
        "kind": lambda d, b, v: d["__x"]["kind"] == _low(v), "type": lambda d, b, v: d["__x"]["kind"] == _low(v),
        "state": lambda d, b, v: d["__x"]["state"] == _low(v), "status": lambda d, b, v: d["__x"]["state"] == _low(v),
        "invoice_no": lambda d, b, v: _low(v) in _low((d.get("dispatch") or {}).get("invoice_no")),
        "docket_no": lambda d, b, v: _low(v) in _low((d.get("dispatch") or {}).get("docket_no")),
        "courier": lambda d, b, v: _low(v) in _low((d.get("dispatch") or {}).get("dispatched_through")),
        "product": lambda d, b, v: _low(v) in _low(_d_product(d)),
        "serial": lambda d, b, v: any(_low(v) in _low(s) for s in _d_serials(d)),
        "company": lambda d, b, v: _low(v) in _low(b.get("company")),
        "dispatched_today": lambda d, b, v: (_local_day((d.get("dispatch") or {}).get("dispatched_at")) == TODAY()) == _truthy(v),
    },
    sorts={"newest": (lambda d, b: b.get("date") or "", True), "oldest": (lambda d, b: b.get("date") or "", False)},
    default_sort="newest", cards=_d_cards, date_of=lambda d: (d.get("dispatch") or {}).get("dispatched_at") or None,
    extra={"confirm": _disp_confirm},
)


# ------------------------------ SERVICES (script.js) ---------------------------------
async def _svc_loader(token):
    svcs, orders = await asyncio.gather(_dataset("/service/", token), _dataset_safe("/order/", token))
    by_serial, by_pid = {}, {}
    for o in orders:
        for it in o.get("items") or []:
            for sn in it.get("serial_numbers") or []:
                by_serial.setdefault(sn, (o, it))
            pid = str(it.get("product_id") or "").strip()
            if pid:
                by_pid.setdefault(pid, (o, it))
    t, out = TODAY(), []
    for s in reversed(svcs):
        s = dict(s)
        pid = str(s.get("product_id") or "").strip()
        m = (by_serial.get(s.get("serial_no")) if s.get("serial_no") else None) or (by_pid.get(pid) if pid else None)
        order, item = m if m else (None, {})
        until, manual = None, False
        if s.get("warranty_until"):
            until, manual = _day(s["warranty_until"]), True
        elif order and order.get("order_date"):
            od = _day(order["order_date"])
            until = od + timedelta(days=365) if od else None
        left = (until - t).days if until else None
        s["__x"] = {"product": s.get("product_name") or item.get("product_name") or pid or "-",
                    "model": s.get("model_no") or item.get("model_no") or "-",
                    "order_id": order.get("order_id") if order else None,
                    "warranty": "none" if left is None else "under" if left >= 0 else "expired",
                    "warranty_until": until.isoformat() if until else None, "warranty_days_left": left, "warranty_extended": manual,
                    "dispatch_state": None if s.get("status") != "completed" else
                    "pending" if not s.get("dispatch") else "dispatched" if (s.get("dispatch") or {}).get("docket_no") else "in_progress"}
        out.append(s)
    return out


def _sv_brief(s):
    x = s["__x"]
    return {"service_id": s.get("service_id"), "status": s.get("status"), "product": x["product"], "model": x["model"],
            "serial_no": s.get("serial_no"), "company": s.get("company_name"), "technician": s.get("technician_alloted"),
            "date": s.get("purchase_date"), "issue": str(s.get("issue") or "")[:140], "location": s.get("location"),
            "warranty": x["warranty"], "warranty_until": x["warranty_until"], "warranty_days_left": x["warranty_days_left"],
            "dispatch_state": x["dispatch_state"], "order_id": x["order_id"]}


def _sv_cards(raws, token):
    by_status, by_tech = {}, {}
    for s in raws:
        by_status[s.get("status")] = by_status.get(s.get("status"), 0) + 1
        by_tech[s.get("technician_alloted") or "(unassigned)"] = by_tech.get(s.get("technician_alloted") or "(unassigned)", 0) + 1
    return {"total": len(raws), "by_status": by_status,
            "under_warranty": sum(1 for s in raws if s["__x"]["warranty"] == "under"),
            "warranty_expired": sum(1 for s in raws if s["__x"]["warranty"] == "expired"),
            "completed_awaiting_dispatch": sum(1 for s in raws if s["__x"]["dispatch_state"] in ("pending", "in_progress")),
            "by_technician": dict(sorted(by_tech.items(), key=lambda kv: -kv[1])[:15])}


SERVICES = Res(
    "services", "erp_services", _svc_loader, lambda s: [s.get("service_id")], _sv_brief,
    lambda s: " ".join([str(s.get("serial_no") or ""), str(s.get("product_id") or ""), s["__x"]["product"], s["__x"]["model"],
                        str(s.get("service_id") or ""), str(s.get("company_name") or "")]),
    """Service page. Warranty (order date + 365 din ya warranty_until) server side nikalti hai - order match karke.
    - filter: filters = {status: active|in_progress|pending_approval|completed|rejected, technician, date (purchase_date),
      warranty: under|expired|none, company, serial_no, location: indoor|outdoor, dispatch_state: pending|in_progress|dispatched,
      from/to (purchase_date)}; sort = newest|oldest; group_by e.g. status | technician | warranty | company
    - view: id = service_id | summary: cards
    - add: data = {product_id, serial_no, technician_id, purchase_date, issue, image, video, company_name, product_name,
      model_no, location (indoor|outdoor), spare_parts, address}
    - update: id + data = sirf badle hue fields {company_name, product_name, model_no, serial_no, technician_alloted,
      purchase_date, issue, location, address, spare_parts, service_charges}
    - delete: id | status: id, data = {service_status, reason, spare_parts_used, spare_parts, service_charges, parts_used:[...]}
    - start / spare_decision / submit_report / request_status_update / request_spare_part / add_faulty / update_charges: id + data
    - send_back: id, {reason} | reschedule: id, {scheduled_date, reason} | approve_completion: id, {service_charges}
    - manager_confirm: id | extend_warranty: id, {warranty_until} | media: id | available_hologram_parts""",
    filters={
        "status": lambda s, b, v: _low(s.get("status")) == _low(v),
        "technician": lambda s, b, v: _low(v) in _low(s.get("technician_alloted")),
        "date": lambda s, b, v: s.get("purchase_date") == str(v)[:10],
        "warranty": lambda s, b, v: s["__x"]["warranty"] == _low(v),
        "company": lambda s, b, v: _low(v) in _low(s.get("company_name")),
        "serial_no": lambda s, b, v: _low(v) in _low(s.get("serial_no")),
        "location": lambda s, b, v: _low(s.get("location")) == _low(v),
        "dispatch_state": lambda s, b, v: _low(s["__x"]["dispatch_state"]) == _low(v),
    },
    sorts={"newest": (lambda s, b: s.get("created_at") or s.get("purchase_date") or "", True),
           "oldest": (lambda s, b: s.get("created_at") or s.get("purchase_date") or "", False)},
    cards=_sv_cards, date_of=lambda s: s.get("purchase_date"),
    routes={"add": R("POST", "/services/create"), "update": R("POST", "/service/edit/{id}"),
            "delete": R("POST", "/service/delete/{id}"), "status": R("POST", "/service/update/{id}"),
            "start": R("POST", "/service/start/{id}"), "spare_decision": R("POST", "/service/spare_decision/{id}"),
            "submit_report": R("POST", "/service/submit_report/{id}"),
            "request_status_update": R("POST", "/service/request_status_update/{id}"),
            "request_spare_part": R("POST", "/service/request_spare_part/{id}"),
            "add_faulty": R("POST", "/service/add_faulty/{id}"), "update_charges": R("POST", "/service/update_charges/{id}"),
            "send_back": R("POST", "/service/send_back/{id}"), "reschedule": R("POST", "/service/reschedule/{id}"),
            "approve_completion": R("POST", "/service/approve_completion/{id}"),
            "manager_confirm": R("POST", "/service/manager_confirm/{id}"),
            "extend_warranty": R("POST", "/service/extend_warranty/{id}"), "media": R("GET", "/service/media/{id}"),
            "available_hologram_parts": R("GET", "/service/available_hologram_parts")},
)


# ---------------------------- INSTALLATIONS (script.js) ------------------------------
async def _inst_loader(token):
    return list(reversed(await _dataset("/installation/", token)))


def _auto_brief(raw: dict, id_key: str):
    out = {}
    for k, v in raw.items():
        if str(k).startswith("__") or k in _HIDDEN_KEYS:
            continue
        if isinstance(v, (int, float, bool)) or v is None or (isinstance(v, str) and not v.startswith("data:") and len(v) <= 160):
            out[k] = v
        if len(out) >= 16:
            break
    out.setdefault(id_key, raw.get(id_key))
    return out


INSTALLATIONS = Res(
    "installations", "erp_installations", _inst_loader, lambda i: [i.get("installation_id")],
    lambda i: _auto_brief(i, "installation_id"), lambda i: json.dumps({k: v for k, v in i.items() if not isinstance(v, str) or len(v) < 300},
                                                                      default=str),
    """Installations (Service page ka installation section). Fields flexible hain - rows me jo scalar fields hain wahi dikhte hain.
    - filter: filters = {status, assigned_to, ...koi bhi field}; search free text | view: id = installation_id
    - add: data = installation request fields | assign: id, data = {technician} | schedule: id, data = {scheduled_date, ...}
    - status: id, data = {status, ...} | media: id (images/videos omit hote hain)
    - update/delete is page par nahi hote""",
    routes={"add": R("POST", "/installation/create"), "assign": R("POST", "/installation/assign/{id}"),
            "schedule": R("POST", "/installation/schedule/{id}"), "status": R("POST", "/installation/status/{id}"),
            "media": R("GET", "/installation/media/{id}")},
    date_of=lambda i: i.get("created_at") or i.get("scheduled_date"),
)


# -------------------------- CUSTOMERS / COMPANIES (users.js) -------------------------
def _entry_key(x):
    return str(x.get("return_date") or x.get("debit_date") or x.get("added_at") or "")


def _purchase_info(p, today):
    amount, mode = _num(p.get("amount")), p.get("payment_mode") or "paid"
    settled = sum(_num(x.get("amount")) for x in p.get("payments") or [])
    upfront = amount if mode == "paid" else (_num(p.get("paid_amount")) if mode == "partial" else 0.0)
    paid = min(upfront + settled, amount)
    balance = 0.0 if mode == "paid" else max(round(amount - paid, 2), 0.0)
    days = int(_num(p.get("credit_days")))
    due = None
    if mode != "paid":
        due = _day(p.get("due_date")) or ((_day(p.get("date")) or today) + timedelta(days=days) if days else None)
    overdue = balance > 0.005 and due is not None and due < today
    return {"purchase_id": p.get("purchase_id"), "date": p.get("date"), "amount": amount, "payment_mode": mode,
            "paid": round(paid, 2), "balance": balance, "credit_days": days, "due_date": due.isoformat() if due else "",
            "overdue": overdue, "late_days": (today - due).days if overdue else 0}


def _company_stats(c, orders, today):
    mine = sorted([o for o in orders if (o.get("customer") or {}).get("customer_id") == c.get("customer_id")],
                  key=lambda o: str(o.get("order_date") or ""), reverse=True)
    order_credit = sum(_num(o.get("total_mrp")) for o in mine if o.get("payment_mode") == "Credit")
    returns = sorted(c.get("returns") or [], key=_entry_key, reverse=True)
    debits = sorted(c.get("debits") or [], key=_entry_key, reverse=True)
    infos = [_purchase_info(p, today) for p in sorted(c.get("purchases") or [], key=lambda p: str(p.get("date") or ""), reverse=True)]
    manual_credit, manual_debit = sum(_num(r.get("amount")) for r in returns), sum(_num(r.get("amount")) for r in debits)
    purchase_credit = sum(i["balance"] for i in infos)
    credit, debit, limit = manual_credit + purchase_credit, order_credit + manual_debit, _num(c.get("credit_limit"))
    outstanding, diff = max(debit - credit, 0.0), credit - debit
    return {"orders": mine, "returns": returns, "debits": debits, "purchases": infos, "credit_limit": limit,
            "debit": round(debit, 2), "credit": round(credit, 2), "outstanding": round(outstanding, 2),
            "available_limit": round(limit - outstanding, 2),
            "overall": f"{abs(diff):.2f}" + (" Dr" if diff < 0 else " Cr" if diff > 0 else ""), "over_limit": debit > limit,
            "total_orders": round(sum(_num(o.get("total_mrp")) for o in mine), 2), "order_credit": round(order_credit, 2),
            "manual_debit": round(manual_debit, 2), "manual_credit": round(manual_credit, 2),
            "purchase_credit": round(purchase_credit, 2),
            "overdue_amount": round(sum(i["balance"] for i in infos if i["overdue"]), 2),
            "overdue_count": sum(1 for i in infos if i["overdue"])}


async def _cust_loader(token):
    customers, orders = await asyncio.gather(_dataset("/customer/", token), _dataset("/order/", token))
    today, out = TODAY(), []
    for c in reversed(customers):
        c = dict(c)
        c["__x"] = _company_stats(c, orders, today)
        out.append(c)
    return out


_STAT_ALIAS = {"credit_limit": "credit_limit", "limit": "credit_limit", "credit_given": "debit", "debit": "debit",
               "debit_amount": "debit", "dr": "debit", "returned": "credit", "credit": "credit", "credit_amount": "credit",
               "cr": "credit", "outstanding": "outstanding", "overall": "outstanding", "balance": "outstanding",
               "total_orders": "total_orders", "orders": "total_orders", "overdue": "overdue_amount",
               "overdue_amount": "overdue_amount"}


def _cust_norm_filters(f: dict) -> dict:
    """{field:'debit', min:200000}  ->  {min_debit:200000}  (+ aliases)."""
    field = _STAT_ALIAS.get(_low(f.pop("field", "") or f.pop("sort_by", "")).replace(" ", "_"), "")
    for k in ("min", "max"):
        if k in f and field:
            f[f"{k}_{field}"] = f.pop(k)
    for k in list(f):
        for pre in ("min_", "max_"):
            if k.startswith(pre) and _STAT_ALIAS.get(k[4:]) and _STAT_ALIAS[k[4:]] != k[4:]:
                f[pre + _STAT_ALIAS[k[4:]]] = f.pop(k)
    return f


def _c_brief(c):
    s = c["__x"]
    return {"customer_id": c.get("customer_id"), "company_name": c.get("company_name"), "gst_number": c.get("gst_number"),
            "contact": f"{c.get('contractor_person') or ''} {c.get('contractor_number') or ''}".strip(),
            "credit_limit": s["credit_limit"], "debit": s["debit"], "credit": s["credit"], "overall": s["overall"],
            "outstanding": s["outstanding"], "total_orders": s["total_orders"], "overdue_amount": s["overdue_amount"],
            "over_limit": s["over_limit"]}


def _c_cards(raws, token):
    return {"companies": len(raws), "over_limit": sum(1 for c in raws if c["__x"]["over_limit"]),
            "with_overdue": sum(1 for c in raws if c["__x"]["overdue_amount"] > 0),
            **{(k if k.startswith("total_") else f"total_{k}"): round(sum(c["__x"][k] for c in raws), 2)
               for k in ("credit_limit", "debit", "credit", "outstanding", "total_orders", "overdue_amount")}}


async def _c_ledger(c, token):
    s = c["__x"]
    orders = [{"order_id": o.get("order_id"), "order_date": o.get("order_date"),
               "items": ", ".join(f"{i.get('product_name')} x{i.get('quantity')}" for i in o.get("items") or []),
               "payment_mode": o.get("payment_mode"), "total": _num(o.get("total_mrp")),
               "on_credit": o.get("payment_mode") == "Credit", "status": o.get("status")} for o in s["orders"]]
    return _shrink({"summary": {k: v for k, v in s.items() if k not in ("orders", "returns", "debits", "purchases")},
                    "orders": orders, "credit_entries": s["returns"], "debit_entries": s["debits"],
                    "purchases": s["purchases"]}, 50, 400)


_cust_filters = {
    "status": lambda c, b, v: (c["__x"]["overdue_count"] > 0 if _low(v) == "overdue" else c["__x"]["over_limit"] if _low(v) == "over_limit"
                               else c["__x"]["outstanding"] > 0 if _low(v) == "outstanding" else c["__x"]["outstanding"] <= 0 if _low(v) == "clear" else True),
    "company": lambda c, b, v: _low(v) in _low(c.get("company_name")),
    "gst": lambda c, b, v: _low(v) in _low(c.get("gst_number")),
}
for _f in ("credit_limit", "debit", "credit", "outstanding", "total_orders", "overdue_amount"):
    _cust_filters[f"min_{_f}"] = (lambda f: lambda c, b, v: c["__x"][f] >= _num(v))(_f)
    _cust_filters[f"max_{_f}"] = (lambda f: lambda c, b, v: c["__x"][f] <= _num(v))(_f)

CUSTOMERS = Res(
    "customers", "erp_customers", _cust_loader, lambda c: [c.get("customer_id")], _c_brief,
    lambda c: " ".join(str(c.get(k) or "") for k in ("company_name", "gst_number", "company_address", "contractor_person",
                                                     "contractor_number", "contractor_email")),
    """Companies / customers (Users page ka Companies tab) - credit, debit, outstanding, overdue sab server side.
    - filter: search = company / GST / contact / address | filters = {status: overdue|over_limit|outstanding|clear,
      min_/max_ + credit_limit|debit|credit|outstanding|total_orders|overdue_amount (ya {field:'debit', min: 200000})}
      sort = outstanding|debit|credit|credit_limit|total_orders|overdue_amount|company_name
      Amount sawal ke liye filters use karo, search nahi. Example: debit > 2 lakh -> filters={"min_debit":200000}
    - view: id = customer_id -> poora ledger (orders, credit/debit entries, purchases with due/overdue)
    - add: data = {company_name (required), company_address, gst_number, contractor_person, contractor_number,
      contractor_email, credit_limit} | update: id + data | delete: id (admin)
    - add_return / add_debit: id, data = {amount, note, return_date} | purchase_payment: id = customer_id, id2 = purchase_id,
      data = {amount, note, payment_date}""",
    filters=_cust_filters,
    sorts={k: ((lambda k: lambda c, b: c["__x"][k])(k), True) for k in
           ("outstanding", "debit", "credit", "credit_limit", "total_orders", "overdue_amount")} |
          {"company_name": (lambda c, b: _low(c.get("company_name")), False)},
    cards=_c_cards, norm_filters=_cust_norm_filters, detail_fn=_c_ledger,
    routes={"add": R("POST", "/customer/create"), "update": R("POST", "/customer/update/{id}", "updated_values"),
            "delete": R("POST", "/customer/delete/{id}"), "add_return": R("POST", "/customer/add_return/{id}"),
            "add_debit": R("POST", "/customer/add_debit/{id}"),
            "purchase_payment": R("POST", "/customer/purchase_payment/{id}/{id2}")},
)


# ------------------------------- USERS (users.js) ------------------------------------
async def _users_loader(token):
    return list(await _dataset_any(["/account/", "/account/users"], token))


def _u_brief(u):
    return {"username": u.get("username"), "name": u.get("full_name") or u.get("name"), "role": u.get("role"),
            "company": u.get("company_name"), "mobile": u.get("mobile_no") or u.get("phone"),
            "email": u.get("email_id"), "manager": u.get("manager") if u.get("role") == "distributor" else None}


USERS = Res(
    "users", "erp_users", _users_loader, lambda u: [u.get("username")], _u_brief,
    lambda u: f"{u.get('username') or ''} {u.get('name') or ''} {u.get('full_name') or ''}",
    """Users / accounts (passwords kabhi return nahi hote).
    - filter: search = username / name | filters = {role: admin|accounts|service_manager|technician|distributor|assembly|...,
      manager, company}; group_by role | manager | company
    - view: id = username | summary: users per role
    - add (admin): data = {username, password, confirm_password, name, email_id, gst_number, company_name, mobile_no, role,
      manager?, credit_limit?} | update (admin): id = username, data = fields to change | delete (admin): id = username""",
    filters={"role": lambda u, b, v: _low(u.get("role")) == _low(v),
             "manager": lambda u, b, v: _low(u.get("manager")) == _low(v),
             "company": lambda u, b, v: _low(v) in _low(u.get("company_name"))},
    sorts={"name": (lambda u, b: _low(u.get("name") or u.get("full_name")), False),
           "username": (lambda u, b: _low(u.get("username")), False)},
    cards=lambda raws, t: {"total": len(raws), "by_role": {r: sum(1 for u in raws if u.get("role") == r)
                                                          for r in sorted({u.get("role") for u in raws if u.get("role")})}},
    routes={"add": R("POST", "/account/create_account/"), "update": R("POST", "/login/update_account/{id}", "updated_values"),
            "delete": R("POST", "/login/delete_account/{id}")},
)


# ------------------------- REQUESTS (allocated.js / orders.js) -----------------------
async def _req_loader(token):
    return list(reversed(await _dataset_any(["/request/", "/request/mine"], token)))


def _r_summary(r):
    d = r.get("details") or {}
    items = d.get("items") or d.get("items_list") or []
    if items:
        return ", ".join(f"{i.get('product_name') or ''} x{i.get('quantity') or 1}" for i in items)
    if (d.get("customer") or {}).get("company_name"):
        return d["customer"]["company_name"]
    if d.get("service_id"):
        return f"Service #{str(d['service_id'])[:8]}"
    return ""


def _r_brief(r):
    return {"request_id": r.get("request_id"), "type": r.get("request_type"), "status": r.get("status"),
            "raised_by": r.get("raised_by"), "date": _d10(r.get("created_at")), "summary": _r_summary(r)[:240]}


REQUESTS = Res(
    "requests", "erp_requests", _req_loader, lambda r: [r.get("request_id")], _r_brief,
    lambda r: f"{r.get('request_id') or ''} {r.get('request_type') or ''} {r.get('raised_by') or ''} {_r_summary(r)}",
    """Approval requests (order / demo_unit / service / spare_part / convert_to_order / reschedule / return ...).
    - filter: filters = {type|request_type, status: pending|approved|rejected, raised_by, from/to}; group_by type | status | raised_by
      (admin/accounts/service_manager poori list dekhte hain, baaki ko apni requests)
    - view: id = request_id | summary: cards (pending per type)
    - approve: id, data = (convert_to_order ke liye) {invoice_no, invoice_date, payment_mode, payment_details}
    - reject: id, data = {reason} | partial_approve: id, data = {approved_indexes:[0,2], remarks}
    - partial_respond: id, data = {accept: true|false, reason} (distributor) | proof: id""",
    filters={"type": lambda r, b, v: _low(r.get("request_type")) == _low(v),
             "request_type": lambda r, b, v: _low(r.get("request_type")) == _low(v),
             "status": lambda r, b, v: _low(r.get("status")) == _low(v),
             "raised_by": lambda r, b, v: _low(v) in _low(r.get("raised_by"))},
    sorts={"newest": (lambda r, b: r.get("created_at") or "", True), "oldest": (lambda r, b: r.get("created_at") or "", False)},
    default_sort="newest",
    cards=lambda raws, t: {"total": len(raws), "pending": sum(1 for r in raws if r.get("status") == "pending"),
                           "pending_by_type": {k: sum(1 for r in raws if r.get("status") == "pending" and r.get("request_type") == k)
                                               for k in sorted({r.get("request_type") for r in raws if r.get("request_type")})}},
    date_of=lambda r: r.get("created_at"),
    routes={"approve": R("POST", "/request/approve/{id}"), "reject": R("POST", "/request/reject/{id}"),
            "partial_approve": R("POST", "/request/partial_approve/{id}"),
            "partial_respond": R("POST", "/request/partial_respond/{id}"), "proof": R("GET", "/request/{id}/proof")},
)


# ------------------------- WORK (work.html / work_report.html) -----------------------
_PRIO = {"urgent": 4, "high": 3, "normal": 2, "low": 1}


async def _work_loader(token):
    return list(await _dataset_any(["/work/all", "/work/mine"], token))


def _w_overdue(w):
    d = _day(w.get("due_date"))
    return bool(d and d < TODAY() and w.get("status") != "done")


def _w_brief(w):
    return {"work_id": w.get("work_id"), "title": w.get("title"), "priority": w.get("priority"), "status": w.get("status"),
            "assigned_to": w.get("assigned_to_name") or w.get("assigned_to"), "assigned_by": w.get("assigned_by_name") or w.get("assigned_by"),
            "due": _d10(w.get("due_date")) or None, "overdue": _w_overdue(w),
            "replies": sum(1 for r in w.get("replies") or [] if r.get("type") != "system"),
            "unread_admin": bool(w.get("admin_unread")), "unread_assignee": bool(w.get("assignee_unread")),
            "date": _d10(w.get("created_at")), "updated": _d10(w.get("updated_at"))}


def _w_cards(raws, token):
    c = lambda s: sum(1 for w in raws if w.get("status") == s)
    return {"total": len(raws), "open": c("open"), "in_progress": c("in_progress"), "done": c("done"),
            "overdue": sum(1 for w in raws if _w_overdue(w)),
            "urgent_not_done": sum(1 for w in raws if w.get("priority") == "urgent" and w.get("status") != "done"),
            "unread_for_manager": sum(1 for w in raws if w.get("admin_unread"))}


WORK = Res(
    "work", "erp_work", _work_loader, lambda w: [w.get("work_id")], _w_brief,
    lambda w: f"{w.get('title') or ''} {w.get('description') or ''} {w.get('assigned_by_name') or ''} {w.get('assigned_to_name') or ''} {w.get('assigned_to') or ''}",
    """Work page - kaam assign karo, reply do, status badlo. (Work managers poori list dekhte hain, baaki ko sirf apna kaam.)
    - filter: filters = {status: open|in_progress|done, priority: urgent|high|normal|low, assigned_to, assigned_by, overdue: true,
      due__gte / due__lte (YYYY-MM-DD), from/to (created)}; sort = newest|due_soon|priority; group_by status | assigned_to | priority
    - view: id = work_id (replies ke saath) | summary: cards (open / in progress / done / overdue ...)
    - add: data = {assigned_to (username), title (max 120), description (max 2000), priority: low|normal|high|urgent, due_date?}
      (sirf work managers)
    - reply: id, data = {body} | status (ya update): id, data = {status: open|in_progress|done} | seen: id
    - delete: is page par nahi hota""",
    filters={"status": lambda w, b, v: _low(w.get("status")) == _low(v), "priority": lambda w, b, v: _low(w.get("priority")) == _low(v),
             "assigned_to": lambda w, b, v: _low(v) in _low(w.get("assigned_to")) or _low(v) in _low(w.get("assigned_to_name")),
             "assigned_by": lambda w, b, v: _low(v) in _low(w.get("assigned_by")) or _low(v) in _low(w.get("assigned_by_name")),
             "overdue": lambda w, b, v: _w_overdue(w) == _truthy(v)},
    sorts={"newest": (lambda w, b: w.get("updated_at") or w.get("created_at") or "", True),
           "due_soon": (lambda w, b: (w.get("due_date") or None) if w.get("status") != "done" else None, False),
           "priority": (lambda w, b: _PRIO.get(w.get("priority"), 0), True)},
    default_sort="newest", cards=_w_cards, date_of=lambda w: w.get("created_at"),
    routes={"add": R("POST", "/work/assign"), "reply": R("POST", "/work/{id}/reply"),
            "status": R("POST", "/work/{id}/status"), "update": R("POST", "/work/{id}/status"),
            "seen": R("POST", "/work/{id}/seen")},
)


# ======================================================================================
# 5. REGISTER TOOLS  +  one-shot overview
# ======================================================================================
ALL_RESOURCES = [ORDERS, INVENTORY, SHIPMENTS, ASSEMBLY, ALLOCATIONS, DISPATCH, SERVICES, INSTALLATIONS,
                 CUSTOMERS, USERS, REQUESTS, WORK]
ALL_RESOURCES = [r for r in ALL_RESOURCES if not ONLY_TOOLS or r.name in ONLY_TOOLS]
for _res in ALL_RESOURCES:
    _make_tool(_res)
    

async def _overview(pages: str = "", auth_token: str = "") -> dict:
    want = {p.strip().lower() for p in pages.split(",") if p.strip()}
    targets = [r for r in ALL_RESOURCES if r.cards and (not want or r.name in want)]

    async def one(r: Res):
        try:
            raws = await r.loader(auth_token)
            return r.name, {"total": len(raws), "cards": await _maybe(r.cards(raws, auth_token))}
        except ErpError as e:
            return r.name, {"error": e.detail, "status": e.status}

    results = await asyncio.gather(*[one(r) for r in targets])
    return {"ok": True, "date": TODAY().isoformat(), "pages": dict(results)}


_OV_DOC = ("Saare ERP pages ke top cards ek call me. pages = comma separated subset (e.g. 'orders,dispatch'), khali = sab. "
           "Jis page ki permission nahi uska error aata hai, baaki chalte rehte hain.")
if EXPOSE_TOKEN:
    async def _ov_tool(pages: str = "", auth_token: str = "") -> dict:
        return await _overview(pages, auth_token)
else:
    async def _ov_tool(pages: str = "") -> dict:
        return await _overview(pages, "")
_ov_tool.__name__, _ov_tool.__doc__ = f"{TOOL_PREFIX}overview", _OV_DOC
mcp.tool(name=f"{TOOL_PREFIX}overview")(_ov_tool)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8766")))