"""Terminal chat for the ERP agent (same agent the API uses at /erp/chat).
Changes need approval here too (y/n); editing the data is only possible in the web UI."""
import asyncio
import json
import logging

from erp_agent import ERPAgent


async def main():
    logging.basicConfig(level=logging.INFO)
    agent = ERPAgent()
    await agent.start()
    print("Model ready. 'exit' likhke band karo, 'reset' se nayi baat.\n")

    while True:
        user = (await asyncio.to_thread(input, "You: ")).strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit", "q"):
            break
        if user.lower() == "reset":
            agent.reset("cli")
            continue

        out = await agent.chat("cli", user)
        while out["approval"]:
            ap = out["approval"]
            print(f"\n[approval] {ap['card']} -> {ap['action']}")
            print(json.dumps(ap["arguments"], indent=2, ensure_ascii=False))
            ans = (await asyncio.to_thread(input, "Apply karna hai? (y/n): ")).strip().lower()
            out = await agent.approve("cli", ap["id"], ans in ("y", "yes", "haan", "ha"))
        print(f"\nAssistant: {out['reply']}\n")


if __name__ == "__main__":
    asyncio.run(main())