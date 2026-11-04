import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";

const DEFAULT_MSG = "Is document ki ERP me entry kar do.";

function Prim({ value, onChange, wide }) {
  if (typeof value === "boolean") return <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />;
  if (typeof value === "number")
    return <input type="number" value={value} onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))} />;
  const s = value == null ? "" : String(value);
  if (wide && (s.length > 80 || s.includes("\n"))) return <textarea rows={3} value={s} onChange={(e) => onChange(e.target.value)} />;
  return <input value={s} onChange={(e) => onChange(e.target.value)} />;
}

function Node({ name, value, onChange }) {
  if (Array.isArray(value)) {
    const rows = value.length > 0 && value.every((r) => r && typeof r === "object" && !Array.isArray(r));
    if (!rows)
      return (
        <label className="fld">
          <span>{name}</span>
          <input
            value={value.map(String).join(", ")}
            onChange={(e) => onChange(e.target.value.split(",").map((x) => x.trim()).filter(Boolean))}
          />
        </label>
      );
    const cols = [...new Set(value.flatMap((r) => Object.keys(r)))];
    const set = (i, k, v) => onChange(value.map((r, j) => (j === i ? { ...r, [k]: v } : r)));
    return (
      <div className="fld full">
        <span>{name}</span>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                {cols.map((c) => (
                  <th key={c}>{c}</th>
                ))}
                <th />
              </tr>
            </thead>
            <tbody>
              {value.map((r, i) => (
                <tr key={i}>
                  {cols.map((c) => (
                    <td key={c}>
                      {r[c] !== null && typeof r[c] === "object" ? (
                        <code>{JSON.stringify(r[c])}</code>
                      ) : (
                        <Prim value={r[c]} onChange={(v) => set(i, c, v)} />
                      )}
                    </td>
                  ))}
                  <td>
                    <button className="ghost sm" onClick={() => onChange(value.filter((_, j) => j !== i))}>
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <button className="ghost sm" onClick={() => onChange([...value, Object.fromEntries(cols.map((c) => [c, ""]))])}>
          + Add row
        </button>
      </div>
    );
  }
  if (value && typeof value === "object")
    return (
      <fieldset className="fld full">
        <legend>{name}</legend>
        <div className="grid">
          {Object.entries(value).map(([k, v]) => (
            <Node key={k} name={k} value={v} onChange={(nv) => onChange({ ...value, [k]: nv })} />
          ))}
        </div>
      </fieldset>
    );
  return (
    <label className="fld">
      <span>{name}</span>
      <Prim value={value} onChange={onChange} wide />
    </label>
  );
}

function Form({ data, onChange }) {
  if (data && typeof data === "object" && !Array.isArray(data))
    return (
      <div className="grid">
        {Object.entries(data).map(([k, v]) => (
          <Node key={k} name={k} value={v} onChange={(nv) => onChange({ ...data, [k]: nv })} />
        ))}
      </div>
    );
  if (Array.isArray(data)) return <Node name="data" value={data} onChange={onChange} />;
  return null;
}

export default function Erp() {
  const [items, setItems] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [att, setAtt] = useState([]);
  const sid = useRef(null);
  const end = useRef();
  const pending = items.some((i) => i.type === "approval" && i.status === "pending");

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [items]);

  function onEvent(ev, data) {
    if (ev === "session") sid.current = data;
    else if (ev === "token")
      setItems((a) => {
        const l = a[a.length - 1];
        if (l?.type === "text" && l.role === "assistant") return [...a.slice(0, -1), { ...l, text: l.text + data }];
        return [...a, { type: "text", role: "assistant", text: data }];
      });
    else if (ev === "tool") setItems((a) => [...a, { type: "tool", name: data.name, action: data.action }]);
    else if (ev === "approval")
      setItems((a) => [...a, { type: "approval", status: "pending", name: data.name, input: data.input, data: data.input.data ?? null }]);
    else if (ev === "error") setItems((a) => [...a, { type: "text", role: "error", text: "⚠ " + data }]);
  }

  async function run(path, body) {
    setBusy(true);
    try {
      await api.streamEvents(path, body, onEvent);
    } catch (e) {
      setItems((a) => [...a, { type: "text", role: "error", text: "⚠ " + e.message }]);
    }
    setBusy(false);
  }

  async function pick(e) {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    setBusy(true);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await api.extract(fd);
      setAtt((a) => [...a, { name: r.name, text: r.text }]);
    } catch (x) {
      setItems((a) => [...a, { type: "text", role: "error", text: "⚠ " + x.message }]);
    }
    setBusy(false);
  }

  function send() {
    const text = input.trim();
    if ((!text && !att.length) || busy || pending) return;
    const files = att.map((f) => `\n\n<attachment name="${f.name}">\n${f.text}\n</attachment>`).join("");
    const shown = (text || (att.length ? DEFAULT_MSG : "")) + (att.length ? `\n📎 ${att.map((f) => f.name).join(", ")}` : "");
    setItems((a) => [...a, { type: "text", role: "user", text: shown }]);
    setInput("");
    setAtt([]);
    run("/erp/chat", { sid: sid.current, message: (text || DEFAULT_MSG) + files });
  }

  function decide(idx, approved) {
    const it = items[idx];
    setItems((a) => a.map((x, i) => (i === idx ? { ...x, status: approved ? "approved" : "rejected" } : x)));
    run("/erp/approve", { sid: sid.current, approved, edited_data: approved ? it.data : null, reason: it.reason || "" });
  }

  const patchItem = (idx, p) => setItems((a) => a.map((x, i) => (i === idx ? { ...x, ...p } : x)));

  function reset() {
    sid.current = null;
    setItems([]);
    setAtt([]);
  }

  return (
    <div className="erpchat">
      <div className="title row between">
        <span>ERP Assistant</span>
        <button className="ghost sm" onClick={reset} disabled={busy}>
          New chat
        </button>
      </div>
      <div className="msgs">
        {items.length === 0 && (
          <div className="empty">HII, I AM YOUR ERP SYSTEM ASSISTANT I CAN DO WHATEVER YOU WANT</div>
        )}
        {items.map((it, i) => {
          if (it.type === "tool")
            return (
              <div key={i} className="toolchip">
                ⚙ {it.name}
                {it.action ? ` · ${it.action}` : ""}
              </div>
            );
          if (it.type === "approval")
            return (
              <div key={i} className={"approve wideform " + it.status}>
                <div className="ah">
                  Please review — <b>{it.name}</b> · <b>{it.input.action}</b>
                  {it.input.id ? ` · id: ${it.input.id}` : ""}
                  {it.input.id2 ? ` · id2: ${it.input.id2}` : ""}
                </div>
                <div className={it.status !== "pending" ? "frozen" : ""}>
                  <Form data={it.data} onChange={(d) => patchItem(i, { data: d })} />
                </div>
                {it.status === "pending" ? (
                  <div className="row">
                    <button className="primary" onClick={() => decide(i, true)} disabled={busy}>
                      Confirm & Save
                    </button>
                    <input
                      placeholder="Reject reason (optional)"
                      value={it.reason || ""}
                      onChange={(e) => patchItem(i, { reason: e.target.value })}
                    />
                    <button className="ghost danger" onClick={() => decide(i, false)} disabled={busy}>
                      Reject
                    </button>
                  </div>
                ) : (
                  <div className="note">{it.status === "approved" ? "Confirmed" : "Rejected"}</div>
                )}
              </div>
            );
          return (
            <div key={i} className={"msg " + (it.role === "error" ? "assistant bad" : it.role)}>
              {it.role === "assistant" ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{it.text}</ReactMarkdown> : it.text}
            </div>
          );
        })}
        {busy && <div className="note">…</div>}
        <div ref={end} />
      </div>
      {att.length > 0 && (
        <div className="chips">
          {att.map((f, i) => (
            <span key={i} className="chip">
              📎 {f.name}{" "}
              <b style={{ cursor: "pointer" }} onClick={() => setAtt((a) => a.filter((_, j) => j !== i))}>
                ✕
              </b>
            </span>
          ))}
        </div>
      )}
      <div className="composer">
        <label className="ghost clip" title="Attach invoice / PO / file">
          📎
          <input type="file" hidden accept=".pdf,.docx,.xlsx,.pptx,.png,.jpg,.jpeg" onChange={pick} disabled={busy || pending} />
        </label>
        <textarea
          rows={1}
          placeholder={pending ? "Pehle form confirm / reject karo" : "Type a message..."}
          value={input}
          disabled={pending}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button className="primary" onClick={send} disabled={busy || pending || (!input.trim() && !att.length)}>
          Send
        </button>
      </div>
    </div>
  );
}