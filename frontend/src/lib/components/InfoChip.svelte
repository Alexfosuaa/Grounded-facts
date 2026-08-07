<script>
  import { fade } from "svelte/transition";

  // A chip badge that reveals a short explanation on hover, focus, or click.
  // Used for the grounding / method / confidence badges so a reader can learn
  // what each term means without leaving the page. `label` is the visible chip
  // text; `text` is the explanation shown in the popover.
  let { label, text, cls = "", tipTitle = "" } = $props();

  let open = $state(false);
  let pinned = $state(false); // a click "pins" the popover; hover is transient

  function toggle() {
    pinned = !pinned;
    open = pinned;
  }
  function onKey(e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggle();
    } else if (e.key === "Escape") {
      open = false;
      pinned = false;
    }
  }
</script>

<span class="infochip-wrap">
  <button
    type="button"
    class="chip infochip {cls}"
    aria-label={tipTitle || label}
    aria-expanded={open}
    onmouseenter={() => (open = true)}
    onmouseleave={() => (open = pinned)}
    onfocus={() => (open = true)}
    onblur={() => {
      open = false;
      pinned = false;
    }}
    onclick={toggle}
    onkeydown={onKey}
  >
    {label}
    <svg class="infochip-i" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  </button>
  {#if open}
    <span class="infochip-pop" role="tooltip" transition:fade={{ duration: 100 }}>
      {text}
    </span>
  {/if}
</span>
