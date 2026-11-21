import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => ({
  // GitHub Pages serves the demo from /<repo>/; set VITE_BASE when building it
  base: loadEnv(mode, ".", "VITE_").VITE_BASE ?? "/",
  plugins: [react(), tailwindcss()],
  server: { port: 5173, host: "127.0.0.1" },
}));
