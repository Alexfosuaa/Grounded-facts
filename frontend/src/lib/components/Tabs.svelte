<script>
  // Accessible tab bar. Controlled by the parent: it owns which tab is active
  // and gets notified via `onselect`. Keeping it dumb makes the active tab easy
  // to drive from anywhere (e.g. deep-linking later).
  let { tabs, active, onselect } = $props();
</script>

<div class="tabs" role="tablist" aria-label="Sections">
  {#each tabs as tab}
    <button
      role="tab"
      class="tab"
      class:active={tab.id === active}
      aria-selected={tab.id === active}
      onclick={() => onselect?.(tab.id)}
    >
      <span class="tab-label">{tab.label}</span>
      {#if tab.badge != null}
        <span class="tab-badge">{tab.badge}</span>
      {/if}
    </button>
  {/each}
</div>
