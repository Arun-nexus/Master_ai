import { useEffect, useState } from "react";
import { api } from "./api";

const COLS = ["sr", "requirement", "required", "offered", "status", "remark"];
const cls = (s) => "st " + String(s || "").toLowerCase().replace(/\s+/g, "-");

function toCsv(rows) {
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  return "\ufeff" + [COLS.join(","), ...rows.map((r) => COLS.map((c) => esc(r[c])).join(","))].join("\n");
}

export default function Files() {
  const [products, setProducts] = useState([]);
  const [view, setView] = useState(null);
  const [spec, setSpec] = useState(null);
  const [match, setMatch] = useState(null);
  const [sheet, setSheet] = useState(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");

  const load = () => api.products().then(setProducts).catch((e) => setErr(e.message));
  useEffect(() => {
    load();
  }, []);

  async function open(n) {
    try {
      setView(await api.text(n));
    } catch (e) {
      setErr(e.message);
    }
  }

  async function del(n) {
    if (!window.confirm(`Delete ${n}?`)) return;
    try {
      await api.remove(n);
      if (view?.name === n) setView(null);
      load();
    } catch (e) {
      setErr(e.message);
    }
  }

  async function runMatch() {
    setErr("");
    setSheet(null);
    setMatch(null);
    setBusy("Searching...");
    const fd = new FormData();
    fd.append("file", spec);
    try {
      setMatch(await api.match(fd));
    } catch (e) {
      setErr(e.message);
    }
    setBusy("");
  }

  async function runSheet(product) {
    setErr("");
    setSheet(null);
    setBusy(`Building compliance sheet for ${product}...`);
    try {
      setSheet(await api.sheet(match.spec_id, product));
    } catch (e) {
      setErr(e.message);
    }
    setBusy("");
  }

  function download() {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([toCsv(sheet.rows)], { type: "text/csv;charset=utf-8" }));
    a.download = `compliance_${sheet.product}.csv`;
    a.click();
  }

  return (
    <div className="files">
      <aside className="side wide">
        <h4>Uploaded files</h4>
        <div className="list">
          {products.map((p) => (
            <div key={p.name} className="item static">
              <b>{p.name}</b>
              {p.sources.map((s, i) => (
                <small key={i}>
                  {s.kind}: {s.file}
                </small>
              ))}
              <div className="row">
                <button className="ghost sm" onClick={() => open(p.name)}>
                  View
                </button>
                <button className="ghost sm danger" onClick={() => del(p.name)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
          {products.length === 0 && <div className="empty">No files yet</div>}
        </div>
      </aside>
      <section className="panel">
        <h3>Compliance sheet</h3>
        <p className="desc">Upload a technical specification (image, PDF, Word, Excel, PPT). Best matching products are found by semantic search.</p>
        <div className="row">
          <input
            key={spec ? spec.name : "e"}
            type="file"
            accept=".pdf,.docx,.xlsx,.pptx,.png,.jpg,.jpeg"
            onChange={(e) => setSpec(e.target.files[0] || null)}
          />
          <button className="primary" onClick={runMatch} disabled={!spec || !!busy}>
            Find matching products
          </button>
        </div>
        {busy && <div className="note">{busy}</div>}
        {err && <div className="err">{err}</div>}
        {match && (
          <div className="cands">
            {match.candidates.length === 0 && <div className="empty">No products in store</div>}
            {match.candidates.map((c) => (
              <div key={c.name} className="cand">
                <div>
                  <b>{c.name}</b>
                  <span className="score">{(c.score * 100).toFixed(1)}%</span>
                  <small>{c.snippet}</small>
                </div>
                <button className="primary sm" onClick={() => runSheet(c.name)} disabled={!!busy}>
                  Compliance sheet
                </button>
              </div>
            ))}
          </div>
        )}
        {sheet && (
          <>
            <div className="row between">
              <h4>
                {sheet.product} — {sheet.rows.length} requirements
              </h4>
              <button className="ghost" onClick={download}>
                Download CSV
              </button>
            </div>
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    {COLS.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sheet.rows.map((r, i) => (
                    <tr key={i}>
                      {COLS.map((c) => (
                        <td key={c}>{c === "status" ? <span className={cls(r[c])}>{r[c]}</span> : r[c]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        {view && (
          <>
            <div className="row between">
              <h4>{view.name}</h4>
              <button className="ghost sm" onClick={() => setView(null)}>
                Close
              </button>
            </div>
            <pre className="result">{view.text}</pre>
          </>
        )}
      </section>
    </div>
  );
}
