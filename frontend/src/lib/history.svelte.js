// A localStorage-backed history of looked-up facts, organized into named
// folders (like a bookmarks manager).
//
// Kept in a `.svelte.js` module so it can use Svelte 5 runes ($state) and be
// shared across components — any importer of `historyStore` re-renders when the
// data changes. Each item snapshots its facts so a past search restores
// instantly, with no network round-trip.

const STORAGE_KEY = "grounded-facts.history.v2";
const LEGACY_KEY = "grounded-facts.history.v1";
const MAX_ENTRIES = 60;

// Short unique id for folders.
function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

// Load persisted state, migrating the old flat v1 list into the v2 shape.
function loadInitial() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        items: Array.isArray(parsed.items) ? parsed.items : [],
        folders: Array.isArray(parsed.folders) ? parsed.folders : [],
      };
    }
    // Migrate v1: a bare array of items with no folders.
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const items = JSON.parse(legacy);
      if (Array.isArray(items)) {
        return { items: items.map((it) => ({ ...it, folderId: null })), folders: [] };
      }
    }
  } catch {
    /* corrupt/unavailable storage — start clean */
  }
  return { items: [], folders: [] };
}

// Reactive singleton. Components read `historyStore.items` / `.folders`.
export const historyStore = $state(loadInitial());

function persist() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ items: historyStore.items, folders: historyStore.folders }),
    );
  } catch {
    // Storage can be unavailable (private mode, quota) — history is a nicety,
    // so we silently carry on with just the in-memory copy.
  }
}

// A stable identity for a lookup so re-searching the same thing updates the
// existing entry (keeping its folder) instead of duplicating it.
function entryKey(topic, title) {
  return `${topic.trim().toLowerCase()}|${(title || "").trim().toLowerCase()}`;
}

export function addHistory({ topic, title = "", facts }) {
  if (!topic || !facts || facts.length === 0) return;
  const key = entryKey(topic, title);
  // Preserve the folder assignment of an existing entry for this lookup.
  const existing = historyStore.items.find((it) => it.key === key);
  const entry = {
    key,
    topic,
    title,
    facts,
    count: facts.length,
    timestamp: Date.now(),
    folderId: existing ? existing.folderId : null,
  };
  historyStore.items = [
    entry,
    ...historyStore.items.filter((it) => it.key !== key),
  ].slice(0, MAX_ENTRIES);
  persist();
}

export function removeHistory(key) {
  historyStore.items = historyStore.items.filter((it) => it.key !== key);
  persist();
}

export function clearHistory() {
  // Clear saved lookups but keep the (now empty) folders the user set up.
  historyStore.items = [];
  persist();
}

// --- folders ---------------------------------------------------------------
export function createFolder(name) {
  const clean = (name || "").trim() || "New folder";
  const folder = { id: uid(), name: clean };
  historyStore.folders = [...historyStore.folders, folder];
  persist();
  return folder.id;
}

export function renameFolder(id, name) {
  const clean = (name || "").trim();
  if (!clean) return;
  historyStore.folders = historyStore.folders.map((f) =>
    f.id === id ? { ...f, name: clean } : f,
  );
  persist();
}

export function deleteFolder(id) {
  // Drop the folder; its items fall back to "Ungrouped" rather than vanishing.
  historyStore.folders = historyStore.folders.filter((f) => f.id !== id);
  historyStore.items = historyStore.items.map((it) =>
    it.folderId === id ? { ...it, folderId: null } : it,
  );
  persist();
}

// Move a saved lookup into a folder (or back to "Ungrouped" with folderId null).
export function moveToFolder(key, folderId) {
  historyStore.items = historyStore.items.map((it) =>
    it.key === key ? { ...it, folderId: folderId || null } : it,
  );
  persist();
}
