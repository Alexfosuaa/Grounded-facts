<script>
  // A single, understated "system" status line for the header. It keeps the
  // portfolio-worthy technical signal (local, offline-capable, vector backend,
  // dimensionality, generation mode) but presents it quietly with a status dot
  // instead of a loud row of jargon badges.
  let { info, infoError } = $props();

  // Compose the muted status text from the /api/info payload.
  let summary = $derived(
    info
      ? [
          "Running locally",
          `${(info.vector_backend || "").toUpperCase()} vectors`,
          `${info.embedding_dim}-dim`,
          info.generation_mode === "llm" ? "LLM generation" : "extractive",
        ]
          .filter(Boolean)
          .join("  ·  ")
      : "",
  );
</script>

<div class="sysline" aria-live="polite">
  {#if infoError}
    <span class="sys-dot err" aria-hidden="true"></span>
    <span class="sys-text">Backend offline</span>
  {:else if info}
    <span class="sys-dot ok" aria-hidden="true"></span>
    <span class="sys-text">{summary}</span>
  {/if}
</div>
