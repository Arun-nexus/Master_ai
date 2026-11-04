
import argparse
import json
import os
import sys

import requests

BASE = os.environ.get("API_URL", "http://localhost:8000")


def add(path: str, name: str) -> None:
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/upload", files={"file": (os.path.basename(path), f)}, data={"name": name})
    r.raise_for_status()
    doc = r.json()
    print(f"\n--- Extracted preview ({doc['chars']} chars) ---\n{doc['preview']}\n")

    choice = input("Approve? [y]es / [n]o / [e]dit : ").strip().lower()
    body: dict = {"approved": choice in ("y", "e"), "edited_text": None}
    if choice == "e":
        fixed = input("Corrected text file ka path: ").strip()
        with open(fixed, encoding="utf-8") as f:
            body["edited_text"] = f.read()

    r = requests.post(f"{BASE}/review/{doc['doc_id']}", json=body)
    r.raise_for_status()
    print(r.json())


def compare(path: str) -> None:
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/search", files={"file": (os.path.basename(path), f)})
    r.raise_for_status()
    res = r.json()
    cands = res["candidates"]
    if not cands:
        sys.exit("Store me koi document nahi mila. Pehle `add` karo.")

    print("\nSimilar candidates:")
    for i, c in enumerate(cands, 1):
        print(f"  {i}. {c['name']}  (score {c['score']})\n     {c['snippet'][:120]!r}")

    raw = input("\nKaun se compare karne hain? (comma separated, e.g. 1,3): ")
    selected = [cands[int(x) - 1]["name"] for x in raw.split(",") if x.strip()]

    with requests.post(
        f"{BASE}/compare",
        json={"query_id": res["query_id"], "selected": selected},
        stream=True,
    ) as resp:
        resp.raise_for_status()
        resp.encoding = "utf-8"
        event = None
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if event == "token":
                    print(data, end="", flush=True)
                elif event == "error":
                    print(f"\n[error] {data}")
                elif event == "done":
                    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("path")
    a.add_argument("--name", required=True)
    c = sub.add_parser("compare")
    c.add_argument("path")
    args = ap.parse_args()
    add(args.path, args.name) if args.cmd == "add" else compare(args.path)


if __name__ == "__main__":
    main()