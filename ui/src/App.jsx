import { useEffect, useState } from "react";
import { auth, expired } from "./api";
import Login from "./Login.jsx";
import Chat from "./Chat.jsx";
import Erp from "./Erp.jsx";
import Files from "./Files.jsx";

const TABS = [
  ["chat", "Product Chat"],
  ["erp", "ERP Tools"],
  ["files", "Files & Compliance"],
];

function initial() {
  const a = auth.get();
  if (a?.token && !expired(a.token)) return a;
  auth.clear();
  return null;
}

export default function App() {
  const [user, setUser] = useState(initial);
  const [tab, setTab] = useState("chat");

  useEffect(() => {
    const out = () => setUser(null);
    window.addEventListener("logout", out);
    return () => window.removeEventListener("logout", out);
  }, []);

  if (!user) {
    return (
      <Login
        onLogin={(a) => {
          auth.set(a);
          setUser(a);
        }}
      />
    );
  }

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <img src="/logo.jpg" alt="ABM" />
          <span>Kiwi</span>
        </div>
        <nav>
          {TABS.map(([k, l]) => (
            <button key={k} className={tab === k ? "tab on" : "tab"} onClick={() => setTab(k)}>
              {l}
            </button>
          ))}
        </nav>
        <div className="who">
          <span>{user.username}</span>
          <button
            className="ghost"
            onClick={() => {
              auth.clear();
              setUser(null);
            }}
          >
            Logout
          </button>
        </div>
      </header>
      <main>
        <div className={tab === "chat" ? "pane" : "pane hide"}>
          <Chat />
        </div>
        <div className={tab === "erp" ? "pane" : "pane hide"}>
          <Erp />
        </div>
        <div className={tab === "files" ? "pane" : "pane hide"}>
          <Files />
        </div>
      </main>
    </div>
  );
}
