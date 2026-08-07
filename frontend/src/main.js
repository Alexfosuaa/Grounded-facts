import "./app.css";
import { mount } from "svelte";
import App from "./App.svelte";

// Svelte 5 mounts a component onto a DOM node with `mount()` (the old
// `new App({ target })` constructor style was removed).
const app = mount(App, {
  target: document.getElementById("app"),
});

export default app;
