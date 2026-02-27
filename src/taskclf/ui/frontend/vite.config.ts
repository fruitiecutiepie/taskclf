import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";

export default defineConfig({
  plugins: [solidPlugin()],
  build: {
    outDir: "../static",
    emptyOutDir: true,
    target: "esnext",
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8741",
      "/ws": { target: "ws://127.0.0.1:8741", ws: true },
    },
  },
});
