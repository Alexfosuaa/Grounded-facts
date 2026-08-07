<script>
  import { slide } from "svelte/transition";

  // Two-way bound source URL. Empty string = use the default (Wikipedia).
  // The parent passes `bind:url={sourceUrl}` and forwards it to /preview or /ask.
  let { url = $bindable("") } = $props();

  let open = $state(false);
  let active = $derived(url.trim().length > 0);
</script>

<div class="source-picker">
  <button
    type="button"
    class="source-toggle"
    class:active
    aria-expanded={open}
    onclick={() => (open = !open)}
  >
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
    {active ? "Custom source" : "Source: Wikipedia"}
    <span class="chev" aria-hidden="true">{open ? "▴" : "▾"}</span>
  </button>

  {#if open}
    <div class="source-body" transition:slide={{ duration: 160 }}>
      <input
        type="url"
        class="source-input"
        placeholder="Paste a public http(s) article link to ground on instead…"
        aria-label="Custom source URL"
        bind:value={url}
      />
      <p class="source-hint muted small">
        Leave blank to use Wikipedia (default). The app fetches the page, strips
        boilerplate, and grounds only on that source.
      </p>
      {#if active}
        <button type="button" class="link" onclick={() => (url = "")}>
          Reset to Wikipedia
        </button>
      {/if}
    </div>
  {/if}
</div>
