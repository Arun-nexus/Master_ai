import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the API (uvicorn app:app --port 8000) is proxied so the browser sees one origin.
const API = process.env.VITE_API || "http://127.0.0.1:8000";
const proxied = ["/auth", "/chat", "/erp", "/products", "/bid", "/download"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(proxied.map((p) => [p, { target: API, changeOrigin: true }])),
  },
});