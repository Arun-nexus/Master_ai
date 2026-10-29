import asyncio
import json
import os
from datetime import date, timedelta
from typing import Any, Literal
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

mcp = FastMCP("Acer Biomedical ERP")

_auth = {"token": os.getenv("ERP_TOKEN", "").strip() or None}
_auth_lock = asyncio.Lock()

_HIDDEN_KEYS = {"password", "confirm_password"}

def R(method, path, wrap=None, search_key=None, defaults=None):
    return (method, path, wrap, search_key, defaults or {})


ORDERS = {
    "list": R("GET", "/order/"),
    "track": R("GET", "/track_order/{id}"),
    "create": R("POST", "/order/create_order/"),
    "update": R("POST", "/order/update/{id}", wrap="updated_order_value"),
    "delete": R("POST", "/order/delete/{id}"),
    "confirm_delivery": R("POST", "/order/confirm_delivery/{id}"),
}

CUSTOMERS = {
    "list": R("GET", "/customer/"),
    "search": R("GET", "/customer/search", search_key="term"),
    "company_list": R("GET", "/customer/company_list"),
    "create": R("POST", "/customer/create"),
    "update": R("POST", "/customer/update/{id}", wrap="updated_values"),
    "delete": R("POST", "/customer/delete/{id}"),
    "add_return": R("POST", "/customer/add_return/{id}"),
    "add_debit": R("POST", "/customer/add_debit/{id}"),
    "purchase_payment": R("POST", "/customer/purchase_payment/{id}/{id2}"),
}

REQUESTS = {
    "list": R("GET", "/request/"),
    "mine": R("GET", "/request/mine"),
    "proof": R("GET", "/request/{id}/proof"),
    "approve": R("POST", "/request/approve/{id}"),
    "reject": R("POST", "/request/reject/{id}"),
    "partial_approve": R("POST", "/request/partial_approve/{id}"),
    "partial_respond": R("POST", "/request/partial_respond/{id}"),
}

ALLOCATIONS = {
    "list": R("GET", "/allocation/"),
    "create": R("POST", "/allocation/create"),
    "edit": R("POST", "/allocation/edit/{id}"),
    "return": R("POST", "/allocation/return/{id}"),
    "send_to_dispatch": R("POST", "/allocation/send_to_dispatch/{id}"),
    "report_damage": R("POST", "/allocation/report_damage/{id}"),
    "damage_image": R("GET", "/allocation/{id}/damage_image"),
    "convert_to_order": R("POST", "/allocation/convert_to_order_direct/{id}"),
}

INVENTORY = {
    "list": R("GET", "/inventory/"),
    "detail": R("GET", "/inventory/detail"),
    "create": R("POST", "/inventory/create"),
    "update": R("POST", "/inventory/update/{id}", wrap="updated_values"),
    "delete": R("POST", "/inventory/delete/{id}"),
    "repair": R("POST", "/inventory/repair/{id}"),
    "serial_history": R("GET", "/inventory/serial_history/{id}"),
    "serial_search": R("GET", "/inventory/serial_search", search_key="q"),
    "available_serials": R("GET", "/inventory/available_serials", defaults={"product_id": ""}),
    "invoice_lookup": R("GET", "/invoice/lookup", search_key="invoice_no"),
}

SHIPMENTS = {
    "list": R("GET", "/shipment/"),
    "get": R("GET", "/shipment/{id}"),
    "create": R("POST", "/shipment/create"),
    "update": R("POST", "/shipment/update/{id}", wrap="updated_values"),
    "delete": R("POST", "/shipment/delete/{id}"),
    "mark_received": R("POST", "/shipment/mark_received/{id}"),
    "fx_rate": R("GET", "/shipment/fx_rate"),
}

ASSEMBLY = {
    "list": R("GET", "/assembly/"),
    "get": R("GET", "/assembly/{id}"),
    "available_parts": R("GET", "/assembly/available_parts"),
    "create": R("POST", "/assembly/create"),
    "edit": R("POST", "/assembly/edit/{id}"),
    "mark_completed": R("POST", "/assembly/mark_completed/{id}"),
    "approve": R("POST", "/assembly/approve/{id}"),
    "delete": R("POST", "/assembly/delete/{id}"),
}

DISPATCH = {
    "queue": R("GET", "/dispatch/"),
    "confirm_order": R("POST", "/dispatch/confirm/order/{id}"),
    "confirm_service": R("POST", "/dispatch/confirm/service/{id}"),
    "confirm_spare_part": R("POST", "/dispatch/confirm/spare_part/{id}"),
}

USERS = {
    "list": R("GET", "/account/users"),
    "list_all": R("GET", "/account/"),
    "technicians": R("GET", "/account/technicians"),
    "distributors": R("GET", "/account/distributors"),
    "create": R("POST", "/account/create_account/"),
    "update": R("POST", "/login/update_account/{id}", wrap="updated_values"),
    "delete": R("POST", "/login/delete_account/{id}"),
}

SERVICES = {
    "list": R("GET", "/service/"),
    "my": R("GET", "/service/my"),
    "active": R("GET", "/service/active"),
    "create": R("POST", "/services/create"),
    "edit": R("POST", "/service/edit/{id}"),
    "update_status": R("POST", "/service/update/{id}"),
    "send_back": R("POST", "/service/send_back/{id}"),
    "reschedule": R("POST", "/service/reschedule/{id}"),
    "approve_completion": R("POST", "/service/approve_completion/{id}"),
    "manager_confirm": R("POST", "/service/manager_confirm/{id}"),
    "extend_warranty": R("POST", "/service/extend_warranty/{id}"),
    "media": R("GET", "/service/media/{id}"),
    "available_hologram_parts": R("GET", "/service/available_hologram_parts"),
    "installations": R("GET", "/installation/"),
    "installation_media": R("GET", "/installation/media/{id}"),
}

WORK = {
    "assign": R("POST", "/work/assign"),
    "list_all": R("GET", "/work/all"),
    "list_mine": R("GET", "/work/mine"),
    "view": R("GET", "/work/{id}"),
}


# --------------------------------------------------------------------------------------
# HTTP + auth
# --------------------------------------------------------------------------------------
async def _login(client: httpx.AsyncClient) -> str | None:
    if not ERP_USERNAME or not ERP_PASSWORD:
        return None
    headers = {"User-Agent": f"erp-mcp key={APP_CLIENT_KEY}"} if APP_CLIENT_KEY else {}
    r = await client.post("/login/", json={"username": ERP_USERNAME, "password": ERP_PASSWORD}, headers=headers)
    if r.status_code != 200:
        return None
    return r.json().get("access_token")


async def _http(method: str, path: str, params: dict | None, body: Any, token: str | None = None):
    """Returns (status_code, payload).
    `token` = the logged-in user's own ERP token (web UI). Without it the shared service account
    (ERP_USERNAME / ERP_PASSWORD) is used, logging in lazily and retrying once if the token expired."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        for attempt in range(2):
            if token:
                bearer = token
            else:
                async with _auth_lock:
                    if not _auth["token"]:
                        _auth["token"] = await _login(client)
                bearer = _auth["token"]
            if not bearer:
                return 401, {"detail": "MCP server could not log in to the ERP - check ERP_USERNAME / ERP_PASSWORD"}
            try:
                r = await client.request(
                    method, path, params=params or None, json=body,
                    headers={"Authorization": f"Bearer {bearer}"},
                )
            except httpx.HTTPError as e:
                return 503, {"detail": f"ERP server not reachable: {e.__class__.__name__}"}
            if r.status_code == 401 and attempt == 0 and not token:
                _auth["token"] = None          # service-account token expired -> login again once
                continue
            break
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"detail": r.text[:500]}


# --------------------------------------------------------------------------------------
# Output cleanup (small models get confused / overflow on huge payloads)
# --------------------------------------------------------------------------------------
def _shrink(o: Any, max_list: int, max_str: int):
    if isinstance(o, str):
        if o.startswith("data:") or len(o) > max_str:
            return f"<{len(o)} chars omitted>"          # base64 images / videos etc.
        return o
    if isinstance(o, list):
        out = [_shrink(i, max_list, max_str) for i in o[:max_list]]
        if len(o) > max_list:
            out.append(f"... +{len(o) - max_list} more")
        return out
    if isinstance(o, dict):
        return {k: _shrink(v, max_list, max_str) for k, v in o.items() if k not in _HIDDEN_KEYS}
    return o


def _as_dict(data: Any):
    if data is None or data == "":
        return {}, None
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return None, "`data` must be a JSON object"
    if not isinstance(data, dict):
        return None, "`data` must be a JSON object"
    return data, None


# --------------------------------------------------------------------------------------
# Newest first: the ERP returns lists oldest first, and paging would then show the oldest 20 records
# --------------------------------------------------------------------------------------
_DATE_KEYS = ("order_date", "created_at", "created_on", "date", "purchase_date", "shipment_date", "request_date",
              "service_date", "installation_date", "added_at", "timestamp")


def _newest_first(rows: list) -> list:
    if not rows or not isinstance(rows[0], dict):
        return rows
    rows = list(reversed(rows))                      # the ERP appends new records at the end
    key = next((k for k in _DATE_KEYS if any(_day(r.get(k)) for r in rows[:50] if isinstance(r, dict))), None)
    if key:                                          # a backdated record must not jump to the top
        rows.sort(key=lambda r: (_day(r.get(key)) if isinstance(r, dict) else None) or date.min, reverse=True)
    return rows


# --------------------------------------------------------------------------------------
# The one dispatcher every tool uses
# --------------------------------------------------------------------------------------
async def _run(table: dict, action: str, id: str, id2: str, search: str, limit: int, offset: int, data: Any,
               token: str = ""):
    spec = table.get(action)
    if not spec:
        return {"ok": False, "error": f"unknown action '{action}'", "valid_actions": list(table)}
    method, path, wrap, search_key, defaults = spec

    body_dict, err = _as_dict(data)
    if err:
        return {"ok": False, "error": err}

    for name, value in (("id", id), ("id2", id2)):
        if "{" + name + "}" in path:
            if not str(value or "").strip():
                return {"ok": False, "error": f"action '{action}' needs `{name}`"}
            path = path.replace("{" + name + "}", quote(str(value).strip(), safe=""))

    params, body = None, None
    if method == "GET":
        params = {**defaults, **body_dict}
        if search_key and search:
            params[search_key] = search
    elif body_dict:
        body = body_dict
        if wrap and wrap not in body_dict:
            body = {wrap: body_dict}
    elif wrap:
        return {"ok": False, "error": f"action '{action}' needs `data`"}

    status, payload = await _http(method, path, params, body, token or None)
    if status >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        return {"ok": False, "status": status, "error": detail}

    # list-type responses: {"message": ..., "dataset": [...]}
    if isinstance(payload, dict) and isinstance(payload.get("dataset"), list):
        rows = payload["dataset"]
        if search and not search_key:
            term = search.lower()
            rows = [r for r in rows if term in json.dumps(r, default=str).lower()]
        rows = _newest_first(rows)
        total = len(rows)
        limit = max(1, min(int(limit or 20), 200))
        offset = max(0, int(offset or 0))
        page = rows[offset:offset + limit]
        result = {k: v for k, v in payload.items() if k != "dataset"}
        result.update({
            "ok": True,
            "total": total,
            "returned": len(page),
            "offset": offset,
            "has_more": offset + len(page) < total,
            "dataset": _shrink(page, max_list=20, max_str=400),
        })
        return result

    result = _shrink(payload, max_list=500, max_str=2500)
    return {"ok": True, **result} if isinstance(result, dict) else {"ok": True, "result": result}


COMPANY_VIEWS = ("summary", "ledger")
SORT_FIELDS = {"credit_limit": "credit_limit", "limit": "credit_limit", "credit_given": "debit", "debit": "debit",
               "debit_amount": "debit", "dr": "debit", "returned": "credit", "credit": "credit",
               "credit_amount": "credit", "cr": "credit", "outstanding": "outstanding", "overall": "outstanding",
               "balance": "outstanding", "total_orders": "total_orders", "orders": "total_orders"}


def _amt(v):
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace(",", "").replace("\u20b9", "").strip())
    except ValueError:
        return None


def _amount_filter(body):
    field = SORT_FIELDS.get(str(body.get("field") or body.get("sort_by") or "").strip().lower().replace(" ", "_"), "")
    lo, hi = _amt(body.get("min")), _amt(body.get("max"))
    for k, v in body.items():
        key = str(k).lower().replace(" ", "_")
        for prefix in ("min_", "max_"):
            if key.startswith(prefix) and key[4:] in SORT_FIELDS:
                field = field or SORT_FIELDS[key[4:]]
                if prefix == "min_":
                    lo = _amt(v) if lo is None else lo
                else:
                    hi = _amt(v) if hi is None else hi
        for suffix in ("_min", "_max"):
            if key.endswith(suffix) and key[:-4] in SORT_FIELDS:
                field = field or SORT_FIELDS[key[:-4]]
                if suffix == "_min":
                    lo = _amt(v) if lo is None else lo
                else:
                    hi = _amt(v) if hi is None else hi
    return field, lo, hi


def _num(x):
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def _day(v):
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _entry_key(x):
    return str(x.get("return_date") or x.get("debit_date") or x.get("added_at") or "")


def _purchase_info(p, today):
    amount = _num(p.get("amount"))
    mode = p.get("payment_mode") or "paid"
    settled = sum(_num(x.get("amount")) for x in p.get("payments") or [])
    upfront = amount if mode == "paid" else (_num(p.get("paid_amount")) if mode == "partial" else 0.0)
    paid = min(upfront + settled, amount)
    balance = 0.0 if mode == "paid" else max(round(amount - paid, 2), 0.0)
    days = int(_num(p.get("credit_days")))
    due = None
    if mode != "paid":
        due = _day(p.get("due_date")) or ((_day(p.get("date")) or today) + timedelta(days=days) if days else None)
    overdue = balance > 0.005 and due is not None and due < today
    return {
        "purchase_id": p.get("purchase_id"), "date": p.get("date"), "amount": amount, "payment_mode": mode,
        "paid": round(paid, 2), "balance": balance, "credit_days": days,
        "due_date": due.isoformat() if due else "", "overdue": overdue,
        "late_days": (today - due).days if overdue else 0,
    }


def _company_stats(c, orders, today):
    mine = [o for o in orders if (o.get("customer") or {}).get("customer_id") == c.get("customer_id")]
    mine.sort(key=lambda o: str(o.get("order_date") or ""), reverse=True)
    order_credit = sum(_num(o.get("total_mrp")) for o in mine if o.get("payment_mode") == "Credit")
    returns = sorted(c.get("returns") or [], key=_entry_key, reverse=True)
    debits = sorted(c.get("debits") or [], key=_entry_key, reverse=True)
    purchases = sorted(c.get("purchases") or [], key=lambda p: str(p.get("date") or ""), reverse=True)
    infos = [_purchase_info(p, today) for p in purchases]
    manual_credit = sum(_num(r.get("amount")) for r in returns)
    manual_debit = sum(_num(r.get("amount")) for r in debits)
    purchase_credit = sum(i["balance"] for i in infos)
    credit = manual_credit + purchase_credit
    debit = order_credit + manual_debit
    limit = _num(c.get("credit_limit"))
    outstanding = max(debit - credit, 0.0)
    diff = credit - debit
    return {
        "orders": mine, "returns": returns, "debits": debits, "purchases": infos,
        "credit_limit": limit, "debit": round(debit, 2), "credit": round(credit, 2),
        "outstanding": round(outstanding, 2), "available_limit": round(limit - outstanding, 2),
        "overall": f"{abs(diff):.2f}" + (" Dr" if diff < 0 else " Cr" if diff > 0 else ""),
        "over_limit": debit > limit,
        "total_orders": round(sum(_num(o.get("total_mrp")) for o in mine), 2),
        "order_credit": round(order_credit, 2), "manual_debit": round(manual_debit, 2),
        "manual_credit": round(manual_credit, 2), "purchase_credit": round(purchase_credit, 2),
        "overdue_amount": round(sum(i["balance"] for i in infos if i["overdue"]), 2),
        "overdue_count": sum(1 for i in infos if i["overdue"]),
    }


async def _fetch_dataset(path, token):
    status, payload = await _http("GET", path, None, None, token or None)
    if status >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        return None, {"ok": False, "status": status, "error": detail}
    return (payload.get("dataset") or []) if isinstance(payload, dict) else [], None


async def _company_view(action, id, search, limit, offset, data, token):
    body, err = _as_dict(data)
    if err:
        return {"ok": False, "error": err}
    customers, fail = await _fetch_dataset("/customer/", token)
    if fail:
        return fail
    orders, fail = await _fetch_dataset("/order/", token)
    if fail:
        return fail
    today = date.today()

    if action == "ledger":
        if not str(id or "").strip():
            return {"ok": False, "error": "action 'ledger' needs `id` (customer_id)"}
        c = next((x for x in customers if x.get("customer_id") == str(id).strip()), None)
        if not c:
            return {"ok": False, "error": "company not found"}
        s = _company_stats(c, orders, today)
        company = {k: v for k, v in c.items() if k not in ("_id", "returns", "debits", "purchases")}
        order_rows = [{
            "order_id": o.get("order_id"), "order_date": o.get("order_date"),
            "items": ", ".join(f"{i.get('product_name')} x{i.get('quantity')}" for i in o.get("items") or []),
            "payment_mode": o.get("payment_mode"), "total": _num(o.get("total_mrp")),
            "on_credit": o.get("payment_mode") == "Credit", "status": o.get("status"),
        } for o in s["orders"]]
        summary = {k: v for k, v in s.items() if k not in ("orders", "returns", "debits", "purchases")}
        return _shrink(
            {"ok": True, "company": company, "summary": summary, "orders": order_rows,
             "credit_entries": s["returns"], "debit_entries": s["debits"], "purchases": s["purchases"]},
            max_list=50, max_str=400)

    term = (search or "").strip().lower()
    field, lo, hi = _amount_filter(body)
    want = str(body.get("status") or "")
    rows = []
    for c in reversed(customers):
        if term:
            hay = " ".join(str(c.get(k) or "") for k in (
                "company_name", "gst_number", "company_address", "contractor_person", "contractor_number")).lower()
            if term not in hay:
                continue
        s = _company_stats(c, orders, today)
        if want == "overdue" and not s["overdue_count"]:
            continue
        if want == "over_limit" and not s["over_limit"]:
            continue
        if want == "outstanding" and not s["outstanding"] > 0:
            continue
        if want == "clear" and s["outstanding"] > 0:
            continue
        if field:
            if lo is not None and s[field] < lo:
                continue
            if hi is not None and s[field] > hi:
                continue
        rows.append({
            "customer_id": c.get("customer_id"), "company_name": c.get("company_name"),
            "gst_number": c.get("gst_number"),
            "contact": f"{c.get('contractor_person') or ''} {c.get('contractor_number') or ''}".strip(),
            "credit_limit": s["credit_limit"], "debit": s["debit"], "credit": s["credit"],
            "overall": s["overall"], "outstanding": s["outstanding"], "total_orders": s["total_orders"],
            "overdue_amount": s["overdue_amount"], "over_limit": s["over_limit"],
        })
    if field:
        rows.sort(key=lambda r: r[field], reverse=True)
    totals = {k: round(sum(r[k] for r in rows), 2) for k in ("credit_limit", "debit", "credit", "outstanding", "total_orders")}
    total = len(rows)
    columns = [
        {"label": "Company", "type": "text"}, {"label": "GST", "type": "text"}, {"label": "Contact", "type": "text"},
        {"label": "Credit limit", "type": "money"}, {"label": "Debit", "type": "money"},
        {"label": "Credit", "type": "money"}, {"label": "Overall", "type": "text"},
        {"label": "Total orders", "type": "money"}, {"label": "Overdue", "type": "money"},
    ]
    table_rows = [[r["company_name"], r["gst_number"], r["contact"], r["credit_limit"], r["debit"], r["credit"],
                   r["overall"], r["total_orders"], r["overdue_amount"]] for r in rows[:500]]
    preview = [f"{r['company_name']} | debit {r['debit']} | credit {r['credit']} | overall {r['overall']}"
               for r in rows[:5]]
    return {
        "ok": True, "total": total, "totals": totals, "over_limit_count": sum(1 for r in rows if r["over_limit"]),
        "overdue_count": sum(1 for r in rows if r["overdue_amount"] > 0), "preview": preview,
        "table": {"columns": columns, "rows": table_rows},
        "note": f"The full table of {total} companies is shown to the user automatically. Do not list the rows; "
                "reply with the count, the totals and at most the top few.",
    }


# --------------------------------------------------------------------------------------
# TOOLS  (har tool ka docstring hi AI model ko dikhta hai - isliye short aur exact rakha hai)
# Common params: action, id, id2, search, limit, offset, data
#   id/id2  : record ka id (order_id, customer_id ...)      search : text filter (list actions)
#   limit/offset : paging (default 20)                      data   : JSON object (body / query)
# --------------------------------------------------------------------------------------
@mcp.tool()
async def erp_orders(
    action: Literal[tuple(ORDERS)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Orders page. Use `action`:
    - list: all orders (use `search` to filter, e.g. company name / invoice no)
    - track: one order's full details (`id` = order_id)
    - create: new order. data = {customer_id OR customer:{company_name, company_address, gst_number,
      contractor_person, contractor_number, contractor_email}, items:[{product_id (can be blank),
      product_name, model_no, product_type, quantity, price, tax_rate, serial_numbers (optional, one per unit)}],
      payment_mode, payment_details (optional), discount (optional), warranty_years (default 1),
      warranty_charge (default 0), reference (username, REQUIRED)}
    - update: `id` + data = fields to change (e.g. {"payment_mode": "UPI"})
    - delete: `id` (admin only)
    - confirm_delivery: `id` (admin/accounts)"""
    return await _run(ORDERS, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_customers(
    action: Literal[tuple(CUSTOMERS) + COMPANY_VIEWS],
    id: str = "",
    id2: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Customers / companies and their ledger. Use `action`:
    - list: all customers (`search` filters locally)
    - search: find customers by text (`search` = company name etc.)
    - company_list: light list of company names only
    - create: data = {company_name (required), company_address, gst_number, contractor_person,
      contractor_number, contractor_email, credit_limit}
    - update: `id` = customer_id, data = fields to change
    - delete: `id` (admin only)
    - add_return: `id`, data = {amount, note, return_date}
    - add_debit: `id`, data = {amount, note, return_date}
    - purchase_payment: `id` = customer_id, `id2` = purchase_id, data = {amount, note, payment_date}
    - summary: Companies table of the Users page: per company credit_limit, debit, credit, overall (Dr/Cr),
      total_orders, overdue_amount + totals. `search` = company / GST / contact / address.
      Optional data = {field: "credit_limit"|"debit"|"credit"|"outstanding"|"total_orders", min, max (numbers),
      status: "overdue"|"over_limit"|"outstanding"|"clear"}. Example: debit more than 2 lakh ->
      action=summary, data={"field": "debit", "min": 200000}. Use data for amount questions, not `search`.
    - ledger: `id` = customer_id -> one company's full history: balances, orders, credit/debit entries,
      purchases with due dates and overdue"""
    if action in COMPANY_VIEWS:
        return await _company_view(action, id, search, limit, offset, data, auth_token)
    return await _run(CUSTOMERS, action, id, id2, search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_requests(
    action: Literal[tuple(REQUESTS)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Approval requests (order / demo unit / service / spare part / convert-to-order). Use `action`:
    - list: all requests (admin/accounts/service_manager); `search` filters
    - mine: only my requests
    - proof: `id` = request_id, proof file info
    - approve: `id`. data (only for convert_to_order requests) = {invoice_no, invoice_date, payment_mode, payment_details}
    - reject: `id`, data = {reason}
    - partial_approve: `id`, data = {approved_indexes: [0, 2], remarks}
    - partial_respond: `id`, data = {accept: true/false, reason} (distributor)"""
    return await _run(REQUESTS, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_allocations(
    action: Literal[tuple(ALLOCATIONS)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Allocated page (stock given to users / distributors / services). Use `action`:
    - list: all allocations (`search` filters)
    - create: data = {allocated_to (username), items:[{product_id, product_name, quantity, model_no,
      product_type, serial_numbers}], company_name, address, gst_number, phone_number}
      OR for a spare part: data = {spare_part:{service_id, part_name, quantity, ...}}
    - edit: `id`, data = {serial_numbers, company_name, address, gst_number, phone_number, remarks} (only changed ones)
    - return: `id` (puts the stock back)
    - send_to_dispatch: `id`
    - report_damage: `id`, data = {issue, image (base64 data URL)}
    - damage_image: `id` (image is not shown, only whether it exists)
    - convert_to_order: demo unit -> order. `id`, data = {customer_id OR company_name/company_address/gst_number,
      price, tax_rate, invoice_no, invoice_date, payment_mode, payment_details}"""
    return await _run(ALLOCATIONS, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_inventory(
    action: Literal[tuple(INVENTORY)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Inventory page. Stock is the same product only if product_name + model_no + product_type all match. Use `action`:
    - list: all stock lots (`search` = product name / model / lot / supplier)
    - detail: data = {product_id, model_no, product_type, lot_id} (any of them) -> full lot with serial numbers
    - create: data = {product_name, product_id (can be blank), quantity, purchase_date, lot_no, supplier, price,
      tax_rate, model_no, product_type ("product"|"spare_parts"|"service_parts"|"damaged"|"accessories"),
      serial_numbers, invoice_no, payment_mode, ...}
    - update: `id` = lot id, data = {updated_values:{...}, new_serial_numbers, remove_serial_numbers,
      faulty_serial_numbers, new_hologram_numbers, remove_hologram_numbers, source_product_type, model_no}
    - delete: `id` (admin only)
    - repair: `id` of a damaged lot
    - serial_history: `id` = serial number -> where it came from / went
    - serial_search: `search` = at least 3 characters of a serial number
    - available_serials: data = {product_id, model_no, product_name, product_type}
    - invoice_lookup: `search` = invoice number"""
    return await _run(INVENTORY, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_shipments(
    action: Literal[tuple(SHIPMENTS)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Shipment page (incoming stock from suppliers). Use `action`:
    - list: all shipments (`search` filters)
    - get: `id` = shipment_id
    - create: data = {company_name, dispatch_date, products:[{product_name, quantity, price, warranty,
      parts:[{part_name, quantity, status: "assembly"|"purchase"|"warranty"}]}], company_address, invoice_no,
      received_date, customer_id, gst_number, payment_mode ("paid"|"partial"|"credit"), paid_amount,
      credit_days, currency, exchange_rate}
    - update: `id`, data = fields to change
    - mark_received: `id`, data = {received_date: "YYYY-MM-DD"}
    - delete: `id` (admin only)
    - fx_rate: data = {currency: "USD"}"""
    return await _run(SHIPMENTS, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_assembly(
    action: Literal[tuple(ASSEMBLY)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Assembly page. Flow: create -> mark_completed -> approve. Use `action`:
    - list: all assemblies (`search` filters)
    - get: `id` = assembly_id
    - available_parts: spare parts currently in inventory that can be used
    - create: data = {product_name, product_id, model_number, quantity, parts_used:[{part_name, quantity,
      source: "inventory"|"local"}], serials:[{serial_number, lot_no}]} (one serial per unit, lot_no required)
    - edit: `id`, data = {product_name, product_id, model_number, parts_used:[...]} (only while pending)
    - mark_completed: `id` (assembly team submits batch)
    - approve: `id`, data = {serial_numbers: [serials that physically arrived]}
    - delete: `id` (admin only)"""
    return await _run(ASSEMBLY, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_dispatch(
    action: Literal[tuple(DISPATCH)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Dispatch page. Use `action`:
    - queue: everything waiting for dispatch (orders, services, spare parts); `search` filters
    - confirm_order: `id` = order_id
    - confirm_service: `id` = service_id
    - confirm_spare_part: `id` = allocation_id
    For all confirm_* actions data = {invoice_no (required), invoice_date (required), docket_no,
    mode_of_delivery, dispatched_through, ship_to_different (true/false),
    ship_to_address:{company_name, address}, image (optional base64)}"""
    return await _run(DISPATCH, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_users(
    action: Literal[tuple(USERS)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Users / accounts page (passwords are never returned). Use `action`:
    - list: system users (`search` filters by name / username / role)
    - list_all: every account record
    - technicians: only technicians
    - distributors: only distributors
    - create: data = {username, password, confirm_password, name, email_id, gst_number, company_name,
      mobile_no, role, manager (optional), credit_limit (optional)} (admin only)
    - update: `id` = username, data = fields to change (admin only)
    - delete: `id` = username (admin only)"""
    return await _run(USERS, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_services(
    action: Literal[tuple(SERVICES)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Service page (services + installations). Use `action`:
    - list: all services (`search` filters) | my: my services | active: services in progress
    - create: data = {product_id, serial_no, technician_id, purchase_date, issue, image, video, company_name,
      product_name, model_no, location ("indoor"|"outdoor"), spare_parts, address}
    - edit: `id` = service_id, data = only changed fields {company_name, product_name, model_no, serial_no,
      technician_alloted, purchase_date, issue, location, address, spare_parts, service_charges}
    - update_status: `id`, data = {service_status, reason, spare_parts_used, spare_parts, service_charges,
      parts_used:[{part_name, old_hologram_number, new_hologram_number}]}
    - send_back: `id`, data = {reason}
    - reschedule: `id`, data = {scheduled_date, reason}
    - approve_completion: `id`, data = {service_charges (optional)}
    - manager_confirm: `id`
    - extend_warranty: `id`, data = {warranty_until: "YYYY-MM-DD"}
    - media: `id` = service_id (images/videos are not shown, only whether they exist)
    - available_hologram_parts: parts that can be swapped on completion
    - installations: all installations (`search` filters) | installation_media: `id` = installation_id"""
    return await _run(SERVICES, action, id, "", search, limit, offset, data, auth_token)


@mcp.tool()
async def erp_work(
    action: Literal[tuple(WORK)],
    id: str = "",
    search: str = "",
    limit: int = 20,
    offset: int = 0,
    data: dict | str | None = None,
    auth_token: str = "",     # set by the web app (user's own ERP token); hidden from the model
) -> dict:
    """Work page - assign work and view it. Use `action`:
    - assign: data = {assigned_to (username), title (max 120 chars), description (max 2000), priority
      ("low"|"normal"|"high"|"urgent"), due_date (optional)} (only work managers can assign)
    - list_all: every assigned work (work managers only; `search` filters by title / person / status)
    - list_mine: work assigned to me
    - view: `id` = work_id -> full work with replies"""
    return await _run(WORK, action, id, "", search, limit, offset, data, auth_token)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8765")))