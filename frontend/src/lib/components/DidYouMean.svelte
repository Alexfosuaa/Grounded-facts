<script>
  // "Did you mean?" follow-up for broad/ambiguous topics. Instead of guessing a
  // sense (planet vs element vs person), we surface Wikipedia's candidate pages
  // and let the user pick one; picking sets an explicit `title` for the fetch.
  let { topic, candidates, ambiguous, onpick, ondismiss } = $props();
</script>

<div class="did-you-mean" role="group" aria-label="Disambiguation">
  <p class="dym-lead">
    {#if ambiguous}
      <strong>“{topic}”</strong> could mean several things — which did you mean?
    {:else}
      Not what you meant? Try a more specific sense of <strong>“{topic}”</strong>:
    {/if}
  </p>
  <div class="dym-chips">
    {#each candidates as cand}
      <button
        class="dym-chip"
        title={cand.description}
        onclick={() => onpick?.(cand)}
      >
        <span class="dym-title">{cand.title}</span>
        {#if cand.description}
          <span class="dym-desc">{cand.description}</span>
        {/if}
      </button>
    {/each}
  </div>
  {#if ondismiss}
    <button class="link dym-dismiss" onclick={() => ondismiss?.()}>
      Dismiss
    </button>
  {/if}
</div>
