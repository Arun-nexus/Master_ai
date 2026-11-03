import argparse
import asyncio
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm


def subset(exp, got):
    if isinstance(got, str) and isinstance(exp, dict):
        try:
            got = json.loads(got)
        except ValueError:
            return False
    if isinstance(exp, dict):
        return isinstance(got, dict) and all(k in got and subset(v, got[k]) for k, v in exp.items())
    if isinstance(exp, str) and isinstance(got, str):
        return exp.lower() == got.lower()
    return exp == got


def tool_case(ea, agent, model, case):
    want = case.get("tool") or ea.CARDS[case["card"]][0]
    messages = [{"role": "system", "content": ea.prompt_with_date()}, {"role": "user", "content": case["q"]}]
    use = {"latency": 0, "in": 0, "out": 0, "cost": 0}
    active = []
    for _ in range(4):
        schemas = [ea.OPEN_CARD_SCHEMA] + [agent.schemas[ea.CARDS[c][0]] for c in active if ea.CARDS[c][0] in agent.schemas]
        raw = llm.chat_generate(messages, schemas, 1024, model=model)
        u = llm.last_usage()
        for k in use:
            use[k] += u.get(k, 0)
        calls = [ea.normalize_call(c) for c in ea.parse_tool_calls(raw)]
        if not calls:
            return False, "no tool call: " + raw[:120], use
        real = [c for c in calls if c["name"] != "open_card"]
        if real:
            c = real[0]
            dump = json.dumps(c["arguments"], ensure_ascii=False).lower()
            ok = c["name"] == want and all(re.search(p, dump) for p in case.get("re", []))
            return ok, json.dumps(c, ensure_ascii=False)[:200], use
        card = str(calls[0]["arguments"].get("card", "")).strip().lower()
        valid = card in ea.CARDS
        if valid and card not in active:
            active.append(card)
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": calls[0]}]})
        messages.append({"role": "tool", "name": "open_card", "content": (
            f"Card '{card}' opened. Now call {ea.CARDS[card][0]} with the `action` you need." if valid
            else f"Error: unknown card '{card}'. Valid cards: {', '.join(ea.CARDS)}")})
    return False, "too many rounds", use


def text_case(model, case):
    out = llm.chat_generate([{"role": "user", "content": case["prompt"]}], None, 400, model=model)
    ok = all(s.lower() in out.lower() for s in case.get("contains", []))
    return ok, out[:200]


def evaluate(model, cases, ea, agent, runs, workers):
    def one(case):
        err = False
        try:
            if case["kind"] == "tool":
                ok, detail, u = tool_case(ea, agent, model, case)
            else:
                ok, detail = text_case(model, case)
                u = llm.last_usage()
        except Exception as e:
            ok, detail, u, err = False, f"{type(e).__name__}: {e}"[:200], llm.last_usage(), True
        return {"model": model, "q": case.get("q") or case.get("prompt", "")[:60], "ok": ok, "err": err, "detail": detail,
                "latency": u.get("latency", 0), "in": u.get("in", 0), "out": u.get("out", 0), "cost": u.get("cost", 0)}
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, cases * runs))


def summarize(rows):
    valid = [r for r in rows if not r["err"]] or rows
    n = len(valid)
    lat = sorted(r["latency"] for r in valid)
    cost = sum(r["cost"] for r in valid)
    return {
        "pass_pct": round(100 * sum(r["ok"] for r in valid) / n, 1),
        "errors": sum(r["err"] for r in rows),
        "avg_s": round(statistics.mean(lat), 2),
        "p95_s": round(lat[min(n - 1, int(n * 0.95))], 2),
        "avg_in": sum(r["in"] for r in valid) // n,
        "avg_out": sum(r["out"] for r in valid) // n,
        "usd_per_1k": round(cost / n * 1000, 3),
    }


async def boot():
    import erp_agent as ea
    agent = ea.ERPAgent()
    await agent.start()
    return ea, agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=llm.PARAMS.get("eval", {}).get("models", []))
    ap.add_argument("--cases", default="eval_cases.jsonl")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="eval_results.json")
    a = ap.parse_args()

    cases = [json.loads(l) for l in Path(a.cases).read_text("utf-8").splitlines() if l.strip()]
    ea = agent = None
    if any(c["kind"] == "tool" for c in cases):
        ea, agent = asyncio.run(boot())

    results = {}
    for m in a.models:
        rows = evaluate(m, cases, ea, agent, a.runs, a.workers)
        results[m] = {"summary": summarize(rows), "rows": rows}

    print(f"{'MODEL':22s} {'err':>4s} {'pass%':>6s} {'avg_s':>6s} {'p95_s':>6s} {'in':>6s} {'out':>5s} {'$/1k':>8s}")
    for m, r in results.items():
        s = r["summary"]
        print(f"{m:22s} {s['errors']:4d} {s['pass_pct']:6.1f} {s['avg_s']:6.2f} {s['p95_s']:6.2f} {s['avg_in']:6d} {s['avg_out']:5d} {s['usd_per_1k']:8.3f}")
    for m, r in results.items():
        bad = [x for x in r["rows"] if not x["ok"]][:5]
        if bad:
            print(f"\n{m} failures:")
            for x in bad:
                print(f"  - {x['q']}\n    {x['detail']}")
    Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()