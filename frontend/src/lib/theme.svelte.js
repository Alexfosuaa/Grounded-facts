// Light/dark theme, persisted to localStorage and reflected on <html> via a
// `data-theme` attribute that the CSS keys off. Kept in a `.svelte.js` module
// so any component can read/toggle it reactively through `themeStore`.

const STORAGE_KEY = "grounded-facts.theme";

// First run: honour a saved choice, otherwise follow the OS preference.
function initialTheme() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage unavailable — fall through to system preference */
  }
  const prefersDark =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

export const themeStore = $state({ theme: initialTheme() });

// Push the current theme onto <html> so the whole document restyles at once.
function apply(theme) {
  if (typeof document !== "undefined") {
    document.documentElement.setAttribute("data-theme", theme);
  }
}

apply(themeStore.theme);

export function toggleTheme() {
  themeStore.theme = themeStore.theme === "dark" ? "light" : "dark";
  apply(themeStore.theme);
  try {
    localStorage.setItem(STORAGE_KEY, themeStore.theme);
  } catch {
    /* best-effort persistence */
  }
}
