import { useEffect, useState } from "react";
import { pingBackend } from "./api";

const GIVE_UP_MS = 60000; // show error only if backend is still down after this long
const RETRY_MS = 1500;

export default function Splash({ onReady }) {
  const [failed, setFailed] = useState(false);
  const [round, setRound] = useState(0);

  useEffect(() => {
    let dead = false;
    let timer;
    const t0 = Date.now();
    setFailed(false);

    const tick = async () => {
      const ok = await pingBackend();
      if (dead) return;
      if (ok) return onReady();
      if (Date.now() - t0 > GIVE_UP_MS) return setFailed(true);
      timer = setTimeout(tick, RETRY_MS);
    };
    tick();

    return () => {
      dead = true;
      clearTimeout(timer);
    };
  }, [round]);

  return (
    <div className="splash">
      <div className={"logo-orbit" + (failed ? " stopped" : "")}>
        <span className="ring r1" />
        <span className="ring r2" />
        <span className="ring r3" />
        <img src="/logo.jpg" alt="ABM" />
      </div>
      {failed ? (
        <>
          <div className="err splash-err">Server se connect nahi ho pa raha. Backend check karo aur dobara try karo.</div>
          <button className="primary" onClick={() => setRound((r) => r + 1)}>
            Retry
          </button>
        </>
      ) : (
        <div className="splash-txt">
          Starting Kiwi<span className="dots" />
        </div>
      )}
    </div>
  );
}