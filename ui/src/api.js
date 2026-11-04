const KEY = "erp_auth";

export const auth = {
  get: () => {
    try {
      return JSON.parse(localStorage.getItem(KEY));
    } catch {
      return null;
    }
  },
  set: (v) => localStorage.setItem(KEY, JSON.stringify(v)),
  clear: () => localStorage.removeItem(KEY),
};

export function expired(token) {
  try {
    const p = JSON.parse(atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    return !!p.exp && p.exp * 1000 < Date.now();
  } catch {
    return false;
  }
}

async function req(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const a = auth.get();
  if (a?.token) headers.Authorization = `Bearer ${a.token}`;
  let body = opts.body;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  }
  const r = await fetch("/api" + path, { method: opts.method || "GET", headers, body });
  if (r.status === 401 && a) {
    auth.clear();
    window.dispatchEvent(new Event("logout"));
  }
  if (!r.ok) {
    let d;
    try {
      d = (await r.json()).detail;
    } catch {
      d = r.statusText;
    }
    throw new Error(typeof d === "string" ? d : JSON.stringify(d));
  }
  return r;
}

const j = (p) => p.then((r) => r.json());
const q = (n) => "?name=" + encodeURIComponent(n);

export const api = {
  login: (username, password) => j(req("/login", { method: "POST", json: { username, password } })),
  products: () => j(req("/products")),
  upload: (fd) => j(req("/products/upload", { method: "POST", body: fd })),
  remove: (n) => j(req("/products" + q(n), { method: "DELETE" })),
  text: (n) => j(req("/products/text" + q(n))),
  extract: (fd) => j(req("/erp/extract", { method: "POST", body: fd })),
  tools: () => j(req("/erp/tools")),
  call: (tool, args) => j(req("/erp/call", { method: "POST", json: { tool, args } })),
  match: (fd) => j(req("/compliance/match", { method: "POST", body: fd })),
  sheet: (spec_id, product) => j(req("/compliance/sheet", { method: "POST", json: { spec_id, product } })),
  async streamEvents(path, body, onEvent) {
    const r = await req(path, { method: "POST", json: body });
    const rd = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await rd.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const blk = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const ev = /event: (.*)/.exec(blk)?.[1];
        const data = /data: (.*)/.exec(blk)?.[1];
        if (!ev || data === undefined) continue;
        onEvent(ev, JSON.parse(data));
      }
    }
  },
  async stream(path, body, onToken) {
    await api.streamEvents(path, body, (ev, v) => {
      if (ev === "token") onToken(v);
      else if (ev === "error") throw new Error(v);
    });
  },
};