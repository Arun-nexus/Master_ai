import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";

export default function Chat() {
  const [products, setProducts] = useState([]);
  const [sel, setSel] = useState("");
  const [q, setQ] = useState("");
  const [chats, setChats] = useState({});
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [up, setUp] = useState({ product: "", kind: "catalogue", file: null });
  const [upMsg, setUpMsg] = useState(null);
  const [upBusy, setUpBusy] = useState(false);
  const end = useRef();

  const load = () => api.products().then(setProducts).catch((e) => setUpMsg({ ok: false, text: e.message }));
  useEffect(() => {
    load();
  }, []);

  const msgs = chats[sel] || [];
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  const shown = useMemo(
    () => products.filter((p) => p.name.toLowerCase().includes(q.toLowerCase())),
    [products, q]
  );

  const patch = (k, fn) => setChats((c) => ({ ...c, [k]: fn(c[k] || []) }));

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    const key = sel;
    const hist = [...(chats[key] || []), { role: "user", content: text }];
    setChats((c) => ({ ...c, [key]: [...hist, { role: "assistant", content: "" }] }));
    setInput("");
    setBusy(true);
    try {
      await api.stream("/chat", { product: key || null, messages: hist }, (t) =>
        patch(key, (m) => {
          const n = m.slice();
          n[n.length - 1] = { ...n[n.length - 1], content: n[n.length - 1].content + t };
          return n;
        })
      );
    } catch (e) {
      patch(key, (m) => {
        const n = m.slice();
        n[n.length - 1] = { role: "assistant", content: "⚠ " + e.message };
        return n;
      });
    }
    setBusy(false);
  }

  async function upload() {
    if (!up.product.trim() || !up.file || upBusy) return;
    setUpBusy(true);
    setUpMsg({ ok: true, text: "Uploading... (big PDFs can take a minute)" });
    const fd = new FormData();
    fd.append("product", up.product.trim());
    fd.append("kind", up.kind);
    fd.append("file", up.file);
    try {
      const r = await api.upload(fd);
      setUpMsg({ ok: true, text: `Saved "${r.product}" (${r.chunks} chunks)` });
      setUp({ ...up, file: null });
      load();
    } catch (e) {
      setUpMsg({ ok: false, text: e.message });
    }
    setUpBusy(false);
  }

  return (
    <div className="chat">
      <aside className="side">
        <input className="search" placeholder="Search catalogues..." value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="list">
          <div className={sel === "" ? "item on" : "item"} onClick={() => setSel("")}>
            General chat
          </div>
          {shown.map((p) => (
            <div key={p.name} className={sel === p.name ? "item on" : "item"} onClick={() => setSel(p.name)}>
              <b>{p.name}</b>
              <small>{p.sources.map((s) => s.kind).join(" + ") || "—"}</small>
            </div>
          ))}
        </div>
        <div className="upload">
          <h4>Add catalogue / user manual</h4>
          <input
            list="plist"
            placeholder="Product name"
            value={up.product}
            onChange={(e) => setUp({ ...up, product: e.target.value })}
          />
          <datalist id="plist">
            {products.map((p) => (
              <option key={p.name} value={p.name} />
            ))}
          </datalist>
          <select value={up.kind} onChange={(e) => setUp({ ...up, kind: e.target.value })}>
            <option value="catalogue">Catalogue</option>
            <option value="manual">User manual</option>
          </select>
          <input
            key={up.file ? up.file.name : "empty"}
            type="file"
            accept=".pdf,.docx,.xlsx,.pptx,.png,.jpg,.jpeg,.webp,.gif"
            onChange={(e) => setUp({ ...up, file: e.target.files[0] || null })}
          />
          <button className="primary" onClick={upload} disabled={!up.product.trim() || !up.file || upBusy}>
            {upBusy ? "Uploading..." : "Upload"}
          </button>
          {upMsg && <div className={upMsg.ok ? "note" : "err"}>{upMsg.text}</div>}
        </div>
      </aside>
      <section className="convo">
        <div className="title">{sel || "General chat"}</div>
        <div className="msgs">
          {msgs.length === 0 && (
            <div className="empty">{sel ? `Ask anything about ${sel}` : "Select a product to chat about it, or ask anything."}</div>
          )}
          {msgs.map((m, i) => (
            <div key={i} className={"msg " + m.role}>
              {m.role === "assistant" ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content || "…"}</ReactMarkdown>
              ) : (
                m.content
              )}
            </div>
          ))}
          <div ref={end} />
        </div>
        <div className="composer">
          <textarea
            rows={1}
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button className="primary" onClick={send} disabled={busy || !input.trim()}>
            Send
          </button>
        </div>
      </section>
    </div>
  );
}