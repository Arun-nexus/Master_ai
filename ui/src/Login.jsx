import { useState } from "react";
import { api } from "./api";

export default function Login({ onLogin }) {
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      const r = await api.login(u.trim(), p);
      onLogin({ token: r.access_token, username: r.username });
    } catch (x) {
      setErr(x.message);
    }
    setBusy(false);
  }

  return (
    <div className="login">
      <form className="card" onSubmit={submit}>
        <img src="/logo.jpg" alt="ABM" />
        <h2>Sign in with your ERP account</h2>
        <input placeholder="Username" value={u} onChange={(e) => setU(e.target.value)} autoFocus />
        <input placeholder="Password" type="password" value={p} onChange={(e) => setP(e.target.value)} />
        {err && <div className="err">{err}</div>}
        <button className="primary" disabled={busy || !u || !p}>
          {busy ? "Signing in..." : "Login"}
        </button>
      </form>
    </div>
  );
}
