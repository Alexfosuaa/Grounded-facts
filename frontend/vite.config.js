import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Vite configuration for the Svelte SPA.
//
// * `base: "./"` makes the built asset URLs relative, so the bundle works when
//   FastAPI serves it from the site root via StaticFiles.
// * The build lands in `dist/`, which the backend mounts at `/`.
// * In `npm run dev`, requests to `/api/*` are proxied to the FastAPI backend so
//   the SPA and the API share one origin during development too.
export default defineConfig({
  plugins: [svelte()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
