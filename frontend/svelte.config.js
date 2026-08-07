import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

// Enables standard preprocessing (e.g. modern JS) inside .svelte files.
export default {
  preprocess: vitePreprocess(),
};
