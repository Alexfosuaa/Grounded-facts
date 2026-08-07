<script>
  import { onMount } from "svelte";

  // Floating "back to top" control. It stays hidden until the user has scrolled
  // far enough that returning to the top is genuinely useful (long fact lists),
  // then smooth-scrolls the window back up.
  let visible = $state(false);
  const SHOW_AFTER = 500; // px scrolled before the button appears

  function onScroll() {
    visible = window.scrollY > SHOW_AFTER;
  }

  function toTop() {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
  }

  onMount(() => {
    onScroll(); // reflect the initial scroll position
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  });
</script>

<button
  class="back-to-top"
  class:visible
  onclick={toTop}
  disabled={!visible}
  aria-hidden={!visible}
  aria-label="Back to top"
  title="Back to top"
>
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M12 19V5M6 11l6-6 6 6" />
  </svg>
</button>
