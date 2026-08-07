<script>
  // One grounded fact, with its citation, grounding score, and a copy button.
  let { fact } = $props();

  // Colour the grounding chip by strength so strong support stands out at a
  // glance. Thresholds mirror the backend's default guard threshold (0.20).
  let strength = $derived(
    fact.grounding_score >= 0.35
      ? "good"
      : fact.grounding_score >= 0.2
        ? "mid"
        : "low",
  );
  let scoreClass = $derived(strength === "low" ? "" : `score-${strength}`);

  // Transient "Copied!" acknowledgement after copying the fact text.
  let copied = $state(false);
  async function copyFact() {
    try {
      await navigator.clipboard.writeText(
        `${fact.fact}\n— ${fact.source_title} (${fact.source_url || "Wikipedia"})`,
      );
      copied = true;
      setTimeout(() => (copied = false), 1400);
    } catch {
      /* clipboard blocked — silently ignore */
    }
  }
</script>

<div class="fact fact-{strength}">
  <div class="fact-body">
    <!-- Svelte auto-escapes interpolated text, so external source text is safe. -->
    <p>{fact.fact}</p>
    <div class="meta">
      <span class="chip {scoreClass}" title="Similarity to the retrieved evidence">
        grounding {fact.grounding_score.toFixed(2)}
      </span>
      <span class="chip">{fact.method}</span>
      <span class="fact-src">
        <svg class="fact-src-icon" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H11v16H5.5A1.5 1.5 0 0 1 4 18.5z" />
          <path d="M20 5.5A1.5 1.5 0 0 0 18.5 4H13v16h5.5a1.5 1.5 0 0 0 1.5-1.5z" />
        </svg>
        {#if fact.source_url}
          <a href={fact.source_url} target="_blank" rel="noopener">
            {fact.source_title}
          </a>
        {:else}
          {fact.source_title}
        {/if}
      </span>
    </div>
  </div>
  <button
    class="fact-copy"
    class:copied
    title="Copy fact with citation"
    aria-label="Copy fact"
    onclick={copyFact}
  >
    {#if copied}
      <!-- Check icon -->
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M5 12.5l4 4 10-10" />
      </svg>
    {:else}
      <!-- Copy icon -->
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="9" width="11" height="11" rx="2" />
        <path d="M5 15V6a2 2 0 0 1 2-2h8" />
      </svg>
    {/if}
  </button>
</div>
