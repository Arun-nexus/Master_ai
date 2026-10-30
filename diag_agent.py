"""Run this next to erp_agent.py:   python diag_agent.py
It starts ONLY the MCP connection (no GPU model, no OCR) and prints the real reason if it fails."""
import asyncio, sys, threading, types, os

# stand-ins for the heavy modules so nothing big is loaded
for name, attrs in {"auth": {"AuthExpired": type("AuthExpired", (Exception,), {})},
                    "extract": {"gpu_lock": threading.Lock()},
                    "rag": {"chat_generate": lambda *a, **k: ""}}.items():
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m

print("python      :", sys.executable)
for pkg in ("fastmcp", "mcp", "langchain_mcp_adapters", "httpx", "python_dotenv"):
    try:
        from importlib.metadata import version
        print(f"{pkg:24s}:", version(pkg.replace('_', '-')))
    except Exception as e:
        print(f"{pkg:24s}: MISSING ({e.__class__.__name__})")

import erp_agent as ea
cfg = ea.SERVER["ABM_ERP"]
print("mode        :", "REMOTE " + cfg["url"] if "url" in cfg else "LOCAL stdio -> " + str(ea.TOOLS_FILE))
if "url" in cfg:
    import httpx
    try:
        r = httpx.get(cfg["url"], timeout=15, follow_redirects=True)
        print("GET url     :", r.status_code, r.headers.get("content-type"), "(406/405/400 are fine: the MCP endpoint is there; 404 = /mcp is NOT deployed)")
    except Exception as e:
        print("GET url     : FAILED", e.__class__.__name__, e)
else:
    import subprocess
    p = subprocess.run([sys.executable, str(ea.TOOLS_FILE)], input=b"", capture_output=True, timeout=60)
    print("server file : exit", p.returncode, "\n  stderr tail:", p.stderr.decode(errors="ignore")[-1200:])

async def main():
    a = ea.ERPAgent()
    try:
        await a.start()
        print("\nOK  tools:", len(a.tools))
    except Exception as e:
        print("\nFAILED:", e)
asyncio.run(main())