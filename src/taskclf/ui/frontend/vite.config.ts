import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";

const apiPort = process.env.TASKCLF_PORT || "8741";

export default defineConfig({
  plugins: [solidPlugin()],
  build: {
    outDir: "../static",
    emptyOutDir: true,
    target: "esnext",
  },
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api": `http://127.0.0.1:${apiPort}`,
      "/ws": { target: `ws://127.0.0.1:${apiPort}`, ws: true },
    },
  },
});
