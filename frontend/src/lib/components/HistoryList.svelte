<script>
  // Sidebar of recent lookups, organized into named folders (like bookmarks).
  // Clicking an item restores its facts instantly (parent re-renders, no fetch).
  import {
    historyStore,
    removeHistory,
    clearHistory,
    createFolder,
    renameFolder,
    deleteFolder,
    moveToFolder,
  } from "../history.svelte.js";

  let { onrestore } = $props();

  // Which folders are collapsed, and which folder is being renamed inline.
  let collapsed = $state(new Set());
  let editingId = $state(null);
  let editName = $state("");
  let addingFolder = $state(false);
  let newFolderName = $state("");

  // Build the display groups: a virtual "Ungrouped" bucket first, then each
  // user folder (shown even when empty so items can be dropped into it).
  let groups = $derived.by(() => {
    const byFolder = new Map();
    for (const it of historyStore.items) {
      const fid = it.folderId || "__none__";
      if (!byFolder.has(fid)) byFolder.set(fid, []);
      byFolder.get(fid).push(it);
    }
    const out = [
      { id: null, name: "Ungrouped", items: byFolder.get("__none__") || [], fixed: true },
    ];
    for (const f of historyStore.folders) {
      out.push({ id: f.id, name: f.name, items: byFolder.get(f.id) || [], fixed: false });
    }
    return out;
  });

  function toggle(id) {
    const key = id ?? "__none__";
    const next = new Set(collapsed);
    next.has(key) ? next.delete(key) : next.add(key);
    collapsed = next;
  }
  const isCollapsed = (id) => collapsed.has(id ?? "__none__");

  function startRename(folder) {
    editingId = folder.id;
    editName = folder.name;
  }
  function commitRename() {
    if (editingId) renameFolder(editingId, editName);
    editingId = null;
    editName = "";
  }
  function commitNewFolder() {
    if (newFolderName.trim()) createFolder(newFolderName);
    newFolderName = "";
    addingFolder = false;
  }

  // Human-friendly "3m ago" style timestamps.
  function ago(ts) {
    const s = Math.max(1, Math.round((Date.now() - ts) / 1000));
    if (s < 60) return `${s}s ago`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.round(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.round(h / 24)}d ago`;
  }
</script>

<aside class="history card">
  <div class="row between history-head">
    <h3>History</h3>
    <div class="history-tools">
      <button class="link" onclick={() => (addingFolder = !addingFolder)}>
        + Folder
      </button>
      {#if historyStore.items.length}
        <button class="link" onclick={clearHistory}>Clear</button>
      {/if}
    </div>
  </div>

  {#if addingFolder}
    <div class="folder-add">
      <input
        type="text"
        placeholder="Folder name"
        bind:value={newFolderName}
        onkeydown={(e) => {
          if (e.key === "Enter") commitNewFolder();
          if (e.key === "Escape") {
            addingFolder = false;
            newFolderName = "";
          }
        }}
      />
      <button class="mini primary" onclick={commitNewFolder}>Add</button>
    </div>
  {/if}

  {#if historyStore.items.length === 0 && historyStore.folders.length === 0}
    <p class="muted small">
      Your recent fact lookups will appear here. Create folders to organize them
      by theme.
    </p>
  {:else}
    <div class="folders">
      {#each groups as group (group.id ?? "__none__")}
        {#if !(group.fixed && group.items.length === 0)}
          <div class="folder">
            <div class="folder-head">
              <button
                class="folder-toggle"
                onclick={() => toggle(group.id)}
                aria-expanded={!isCollapsed(group.id)}
              >
                <span class="caret" class:open={!isCollapsed(group.id)} aria-hidden="true">▸</span>
                {#if editingId !== null && editingId === group.id}
                  <!-- svelte-ignore a11y_autofocus -->
                  <input
                    class="folder-rename"
                    bind:value={editName}
                    onclick={(e) => e.stopPropagation()}
                    onkeydown={(e) => {
                      if (e.key === "Enter") commitRename();
                      if (e.key === "Escape") {
                        editingId = null;
                      }
                    }}
                    onblur={commitRename}
                    autofocus
                  />
                {:else}
                  <span class="folder-name">{group.name}</span>
                {/if}
                <span class="folder-count">{group.items.length}</span>
              </button>
              {#if !group.fixed && editingId !== group.id}
                <div class="folder-actions">
                  <button class="icon-btn" title="Rename folder" aria-label="Rename folder" onclick={() => startRename(group)}>✎</button>
                  <button class="icon-btn" title="Delete folder" aria-label="Delete folder" onclick={() => deleteFolder(group.id)}>×</button>
                </div>
              {/if}
            </div>

            {#if !isCollapsed(group.id)}
              {#if group.items.length === 0}
                <p class="folder-empty">Empty — move a lookup here.</p>
              {:else}
                <ul class="history-list">
                  {#each group.items as item (item.key)}
                    <li>
                      <button class="history-item" onclick={() => onrestore?.(item)}>
                        <span class="hi-top">
                          <span class="hi-topic">{item.topic}</span>
                          <span class="hi-count">{item.count}</span>
                        </span>
                        <span class="hi-sub">
                          {#if item.title && item.title !== item.topic}
                            {item.title} ·
                          {/if}
                          {ago(item.timestamp)}
                        </span>
                      </button>
                      <div class="hi-actions">
                        <select
                          class="hi-move"
                          title="Move to folder"
                          aria-label="Move to folder"
                          value={item.folderId || ""}
                          onchange={(e) => moveToFolder(item.key, e.currentTarget.value)}
                        >
                          <option value="">Ungrouped</option>
                          {#each historyStore.folders as f}
                            <option value={f.id}>{f.name}</option>
                          {/each}
                        </select>
                        <button
                          class="hi-remove"
                          title="Remove"
                          aria-label="Remove from history"
                          onclick={() => removeHistory(item.key)}>×</button
                        >
                      </div>
                    </li>
                  {/each}
                </ul>
              {/if}
            {/if}
          </div>
        {/if}
      {/each}
    </div>
  {/if}
</aside>
