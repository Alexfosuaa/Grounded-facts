<script>
  import { api } from "../api.js";
  import { fly, fade } from "svelte/transition";

  // Example questions that resolve cleanly to single-topic articles, so the
  // first run demonstrates a grounded answer rather than an abstain.
  const EXAMPLES = [
    "What does photosynthesis produce?",
    "How do black holes form?",
    "Who developed the theory of relativity?",
    "What is DNA made of?",
  ];

  let question = $state("");
  let result = $state(null); // AnswerResponse from /api/ask, or null
  let status = $state({ msg: "", state: "" });
  let loading = $state(false);
  let reqSeq = 0; // guards against an older in-flight answer overwriting a newer one

  // Map the raw cosine confidence to a friendly label + chip colour. QA uses a
  // lower grounding bar than fact extraction (see QA_GROUNDING_THRESHOLD), so a
  // correct answer can still read as "Low" when the offline embedder is unsure.
  let confidence = $derived(result?.confidence ?? 0);
  let confLabel = $derived(
    confidence >= 0.45 ? "High" : confidence >= 0.3 ? "Medium" : "Low",
  );
  let confClass = $derived(
    confidence >= 0.45 ? "score-good" : confidence >= 0.3 ? "score-mid" : "",
  );

  // GET /api/ask and store the answer (or abstain) for rendering.
  async function ask() {
    const q = question.trim();
    if (!q) {
      status = { msg: "Type a question first.", state: "err" };
      return;
    }
    const myReq = ++reqSeq;
    loading = true;
    status = { msg: "", state: "" };
    result = null;
    try {
      const r = await api(`/ask?question=${encodeURIComponent(q)}`);
      if (myReq !== reqSeq) return; // a newer question superseded this one
      result = r;
    } catch (err) {
      if (myReq !== reqSeq) return;
      status = { msg: err.message, state: "err" };
    } finally {
      if (myReq === reqSeq) loading = false;
    }
  }

  function runExample(ex) {
    question = ex;
    ask();
  }
</script>

<section class="card">
  <div class="panel-head">
    <h2>Ask a question</h2>
    <p class="muted small">
      Get a single answer drawn straight from retrieved sources — with citations.
      If nothing in the sources supports an answer, the app says so instead of
      guessing.
    </p>
  </div>

  <div class="searchbar">
    <svg class="searchbar-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="M21 21l-4.3-4.3" />
    </svg>
    <input
      type="text"
      class="searchbar-input"
      placeholder="Ask anything — e.g. What does photosynthesis produce?"
      aria-label="Question"
      bind:value={question}
      onkeydown={(e) => e.key === "Enter" && !loading && ask()}
    />
  </div>
  <div class="search-controls end">
    <button class="primary cta" onclick={ask} disabled={loading}>
      {#if loading}
        <span class="spinner" aria-hidden="true"></span>
        Thinking…
      {:else}
        Ask
        <svg class="cta-arrow" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M5 12h14M13 6l6 6-6 6" />
        </svg>
      {/if}
    </button>
  </div>

  {#if status.msg}
    <div class="status {status.state}">{status.msg}</div>
  {/if}

  {#if !result && !loading && !status.msg}
    <!-- First-run empty state: sample questions so the panel is usable instantly. -->
    <div class="quickstart" in:fade={{ duration: 160 }}>
      <p class="qs-lead">Try a question:</p>
      <div class="qs-chips">
        {#each EXAMPLES as ex}
          <button class="qs-chip" onclick={() => runExample(ex)}>{ex}</button>
        {/each}
      </div>
    </div>
  {/if}

  {#if result}
    <div class="answer-block" transition:fade={{ duration: 140 }}>
      {#if result.grounded}
        <div class="answer" in:fly={{ y: 8, duration: 220 }}>
          <div class="answer-head">
            <span class="answer-tag">Answer</span>
            <span class="chip {confClass}" title="How closely the answer matches your question">
              Confidence: {confLabel}
            </span>
          </div>
          <p class="answer-text">{result.answer}</p>
          {#if result.source_url}
            <a class="answer-src" href={result.source_url} target="_blank" rel="noopener">
              Source: {result.source_title}
            </a>
          {/if}
        </div>
      {:else}
        <div class="answer abstain" in:fly={{ y: 8, duration: 220 }}>
          <div class="answer-head">
            <span class="answer-tag warn">No confident answer</span>
          </div>
          <p class="answer-text">
            The retrieved sources don't clearly answer that, so rather than guess
            here are the closest passages found. Try rephrasing or being more
            specific.
          </p>
        </div>
      {/if}

      {#if result.citations.length}
        <div class="citations">
          <p class="citations-label">
            {result.grounded ? "Sources" : "Closest passages"}
          </p>
          {#each result.citations as c}
            <div class="citation">
              <div class="citation-head">
                {#if c.source_url}
                  <a href={c.source_url} target="_blank" rel="noopener">{c.source_title}</a>
                {:else}
                  <span>{c.source_title}</span>
                {/if}
                <span class="chip" title="Retrieval similarity">match {c.score.toFixed(2)}</span>
              </div>
              <p class="citation-snippet">{c.snippet}</p>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</section>
