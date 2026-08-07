<script>
  import { api } from "../api.js";
  import { fly, fade } from "svelte/transition";
  import SourcePicker from "./SourcePicker.svelte";
  import InfoChip from "./InfoChip.svelte";

  // Example questions that resolve cleanly to single-topic articles, so the
  // first run demonstrates a grounded answer rather than an abstain.
  const EXAMPLES = [
    "What does photosynthesis produce?",
    "How do black holes form?",
    "Who developed the theory of relativity?",
    "What is DNA made of?",
  ];

  let question = $state("");
  let sourceUrl = $state(""); // optional custom source; blank = Wikipedia (default)
  let result = $state(null); // AnswerResponse from /api/ask, or null
  let status = $state({ msg: "", state: "" });
  let loading = $state(false);
  let answerIndex = $state(0); // which answer is shown: 0 = best, then alternatives
  let reqSeq = 0; // guards against an older in-flight answer overwriting a newer one

  // The best answer plus any runner-up "alternatives" as one cyclable list, so
  // "Try another answer" can step through them without another request.
  let answers = $derived(
    result && result.grounded
      ? [
          {
            answer: result.answer,
            confidence: result.confidence,
            source_title: result.source_title,
            source_url: result.source_url,
          },
          ...(result.alternatives ?? []),
        ]
      : [],
  );
  let current = $derived(answers[answerIndex] ?? null);

  // Keep the evidence panel in sync with the *shown* answer. The primary answer
  // shows the full retrieved passages; an alternative (which comes from a single
  // retrieved sentence) shows its own source so we never display citations that
  // belong to a different answer.
  let displayCitations = $derived(
    !result
      ? []
      : answerIndex === 0
        ? (result.citations ?? [])
        : current
          ? [
              {
                source_title: current.source_title,
                source_url: current.source_url,
                snippet: current.answer,
                score: current.confidence,
              },
            ]
          : [],
  );

  // Map the raw cosine confidence of the *shown* answer to a friendly label +
  // chip colour. QA uses a lower grounding bar than fact extraction (see
  // QA_GROUNDING_THRESHOLD), so a correct answer can still read as "Low" when
  // the embedder is unsure.
  let confidence = $derived(current?.confidence ?? 0);
  let confLabel = $derived(
    confidence >= 0.45 ? "High" : confidence >= 0.3 ? "Medium" : "Low",
  );
  let confClass = $derived(
    confidence >= 0.45 ? "score-good" : confidence >= 0.3 ? "score-mid" : "",
  );

  // Step to the next answer, wrapping back to the best one at the end.
  function tryAnother() {
    if (answers.length > 1) answerIndex = (answerIndex + 1) % answers.length;
  }

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
      const params = new URLSearchParams({ question: q });
      const src = sourceUrl.trim();
      if (src) params.set("source_url", src);
      const r = await api(`/ask?${params.toString()}`);
      if (myReq !== reqSeq) return; // a newer question superseded this one
      result = r;
      answerIndex = 0; // show the best answer first
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

  <SourcePicker bind:url={sourceUrl} />

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
            <InfoChip
              label={`Confidence: ${confLabel}`}
              cls={confClass}
              tipTitle="What confidence means"
              text="Confidence reflects how strongly the retrieved source supports this answer (retrieval similarity). Low means the match was weak, so the answer may be off — try 'Try another answer' or rephrase. The app abstains entirely when nothing clears the grounding bar."
            />
          </div>
          <p class="answer-text">{current?.answer}</p>
          {#if current?.source_url}
            <a class="answer-src" href={current.source_url} target="_blank" rel="noopener">
              Source: {current.source_title}
            </a>
          {/if}
          {#if answers.length > 1}
            <div class="answer-actions">
              <button class="try-another" onclick={tryAnother}>
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M23 4v6h-6M1 20v-6h6" />
                  <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                </svg>
                Try another answer
              </button>
              <span class="answer-count">{answerIndex + 1} of {answers.length}</span>
            </div>
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

      {#if displayCitations.length}
        <div class="citations">
          <p class="citations-label">
            {result.grounded ? "Sources" : "Closest passages"}
          </p>
          {#each displayCitations as c}
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
