<script>
  import { onMount } from "svelte";
  import { api } from "../api.js";
  import FactCard from "./FactCard.svelte";
  import DidYouMean from "./DidYouMean.svelte";
  import SourcePicker from "./SourcePicker.svelte";
  import { addHistory } from "../history.svelte.js";
  import { fly, fade } from "svelte/transition";

  // Curated starter topics — coherent, single-sense articles that show the app
  // off well on first run. Clicking one runs the lookup immediately.
  const EXAMPLES = [
    "Photosynthesis",
    "Black hole",
    "Machine learning",
    "DNA",
    "Great Barrier Reef",
    "Volcano",
  ];

  // --- form state ---
  let topic = $state("Photosynthesis");
  let maxFacts = $state(3);
  let sourceUrl = $state(""); // optional custom source; blank = Wikipedia (default)

  // Keep the requested fact count sane even if the number input is left blank
  // or pushed past the bounds (1–50). The article's length naturally limits the
  // result, so asking for the max just returns every grounded fact available.
  const MAX_FACTS = 50;
  function clampFacts(n) {
    const v = Math.round(Number(n) || 3);
    return Math.min(MAX_FACTS, Math.max(1, v));
  }
  // Stepper buttons for quick ±1 adjustments.
  const incFacts = () => (maxFacts = clampFacts(clampFacts(maxFacts) + 1));
  const decFacts = () => (maxFacts = clampFacts(clampFacts(maxFacts) - 1));

  // --- results state ---
  let facts = $state([]);
  let activeTitle = $state(""); // explicit page chosen via disambiguation
  let sourceTitle = $state("");
  let status = $state({ msg: "", state: "" });
  let loading = $state(false);
  let reqSeq = 0; // guards against a stale lookup overwriting a newer one

  // --- disambiguation state ---
  let candidates = $state([]);
  let ambiguous = $state(false);
  let showDidYouMean = $state(false);

  // --- retrieval-debug panel (loaded lazily the first time it is opened) ---
  let showDebug = $state(false);
  let debugHits = $state(null);
  let debugLoaded = $state(false);

  // Reset everything tied to a specific result set.
  function resetResults() {
    facts = [];
    sourceTitle = "";
    showDebug = false;
    debugHits = null;
    debugLoaded = false;
  }

  // Step 1 — the user asks about a topic. For broad/ambiguous topics we pause
  // and ask a follow-up question instead of guessing a sense; otherwise we go
  // straight to grounding facts for the primary article.
  async function curate() {
    if (!topic.trim()) {
      status = { msg: "Please enter a topic.", state: "err" };
      return;
    }
    const myReq = ++reqSeq;
    loading = true;
    showDidYouMean = false;
    activeTitle = "";
    resetResults();
    status = { msg: "Checking the topic…", state: "info" };
    // Capture the topic now so a keystroke during the await can't change which
    // article we ground against midway through.
    const forTopic = topic;

    try {
      // A custom source bypasses Wikipedia entirely, so disambiguation (a
      // Wikipedia-only concept) doesn't apply — ground on the given page.
      if (sourceUrl.trim()) {
        await runPreview(forTopic, "", myReq);
        return;
      }
      const dis = await api(`/disambiguate?topic=${encodeURIComponent(forTopic)}`);
      if (myReq !== reqSeq) return; // superseded by a newer lookup
      candidates = dis.candidates || [];
      ambiguous = !!dis.ambiguous;

      if (ambiguous && candidates.length > 1) {
        // Ask the follow-up question rather than committing to a guess.
        showDidYouMean = true;
        status = { msg: "", state: "" };
        loading = false;
        return;
      }
      // Unambiguous — ground facts for the primary article directly.
      await runPreview(forTopic, "", myReq);
    } catch (err) {
      // Disambiguation is best-effort; fall back to a direct preview.
      if (myReq !== reqSeq) return;
      await runPreview(forTopic, "", myReq);
    }
  }

  // Step 2 — fetch grounded facts, optionally pinned to a chosen page `title`.
  // `token` lets `curate()` share its request slot; a direct call starts a new one.
  async function runPreview(forTopic, title, token) {
    const myReq = token ?? ++reqSeq;
    loading = true;
    showDidYouMean = false;
    resetResults();
    status = { msg: "Retrieving sources and grounding facts…", state: "info" };
    try {
      const params = new URLSearchParams({
        topic: forTopic,
        max_facts: String(clampFacts(maxFacts)),
      });
      if (title) params.set("title", title);
      const src = sourceUrl.trim();
      if (src) params.set("source_url", src);
      const data = await api(`/preview?${params.toString()}`);
      if (myReq !== reqSeq) return; // a newer lookup is in flight — drop this one
      facts = data.facts || [];
      activeTitle = title;
      sourceTitle = facts[0]?.source_title || title || forTopic;

      if (facts.length === 0) {
        status = {
          msg: `No grounded facts found for “${forTopic}”. The sources may be temporarily unreachable — try again, or refine the topic.`,
          state: "err",
        };
      } else {
        status = { msg: "", state: "" };
        // Record the resolved source (e.g. "AI" → "Artificial intelligence")
        // so history shows the real article and dedups on it.
        addHistory({ topic: forTopic, title: sourceTitle, facts });
        syncUrl(forTopic, title);
      }
    } catch (err) {
      if (myReq !== reqSeq) return;
      status = { msg: err.message, state: "err" };
    } finally {
      if (myReq === reqSeq) loading = false;
    }
  }

  // Reflect the current lookup in the address bar so it can be bookmarked or
  // shared; opening such a link re-runs the same lookup (see onMount below).
  function syncUrl(forTopic, title) {
    try {
      const q = new URLSearchParams({ topic: forTopic });
      if (title) q.set("title", title);
      q.set("facts", String(clampFacts(maxFacts)));
      history.replaceState(null, "", `?${q.toString()}`);
    } catch {
      /* history API unavailable — non-critical */
    }
  }

  // Deep link support: ?topic=…&title=…&facts=… runs a lookup on load.
  onMount(() => {
    try {
      const p = new URLSearchParams(location.search);
      const t = p.get("topic");
      if (t && t.trim()) {
        topic = t;
        const f = p.get("facts");
        if (f) maxFacts = clampFacts(f);
        const title = p.get("title");
        if (title) runPreview(topic, title);
        else curate();
      }
    } catch {
      /* ignore malformed query strings */
    }
  });

  // The user picked a specific sense from the "Did you mean?" chips.
  function pickCandidate(cand) {
    topic = cand.title.replace(/\s*\(.*?\)\s*$/, "").trim() || topic;
    runPreview(topic, cand.title);
  }

  // Quick-start: run a lookup for a suggested example topic.
  function runExample(name) {
    topic = name;
    curate();
  }

  // Copy a shareable link to the current lookup.
  let linkCopied = $state(false);
  async function copyLink() {
    try {
      await navigator.clipboard.writeText(location.href);
      linkCopied = true;
      setTimeout(() => (linkCopied = false), 1400);
    } catch {
      /* clipboard blocked — ignore */
    }
  }

  // Restore a past lookup from history (facts are snapshotted — no network).
  export function restore(item) {
    topic = item.topic;
    activeTitle = item.title || "";
    facts = item.facts || [];
    sourceTitle = facts[0]?.source_title || item.title || item.topic;
    showDidYouMean = false;
    status = { msg: "", state: "" };
    showDebug = false;
    debugHits = null;
    debugLoaded = false;
  }

  // Offer the disambiguation chips on demand even when a topic wasn't flagged
  // ambiguous (e.g. the primary sense wasn't what the user wanted).
  async function otherSenses() {
    if (candidates.length === 0) {
      try {
        const dis = await api(`/disambiguate?topic=${encodeURIComponent(topic)}`);
        candidates = dis.candidates || [];
        ambiguous = !!dis.ambiguous;
      } catch {
        candidates = [];
      }
    }
    showDidYouMean = candidates.length > 0;
  }

  // Show/hide the raw retrieved chunks, fetching them on first open.
  async function toggleDebug() {
    showDebug = !showDebug;
    if (showDebug && !debugLoaded) {
      try {
        const params = new URLSearchParams({ topic, k: "5" });
        if (activeTitle) params.set("title", activeTitle);
        const data = await api(`/retrieve?${params.toString()}`);
        debugHits = data.hits || [];
      } catch {
        debugHits = [];
      } finally {
        debugLoaded = true;
      }
    }
  }
</script>

<section class="card">
  <div class="panel-head">
    <h2>Explore facts</h2>
    <p class="muted small">
      Enter any topic. Every fact is grounded against retrieved Wikipedia
      evidence and checked by the hallucination guard before it's shown.
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
      placeholder="Ask about any topic — e.g. Photosynthesis"
      aria-label="Topic"
      bind:value={topic}
      onkeydown={(e) => e.key === "Enter" && curate()}
    />
  </div>
  <div class="search-controls">
    <label class="facts-field">
      <span class="facts-label">Facts to show</span>
      <span class="stepper">
        <button
          type="button"
          class="step"
          onclick={decFacts}
          disabled={loading || clampFacts(maxFacts) <= 1}
          aria-label="Fewer facts">−</button>
        <input
          class="facts-count"
          type="number"
          min="1"
          max={MAX_FACTS}
          bind:value={maxFacts}
          aria-label="Number of facts"
        />
        <button
          type="button"
          class="step"
          onclick={incFacts}
          disabled={loading || clampFacts(maxFacts) >= MAX_FACTS}
          aria-label="More facts">+</button>
      </span>
    </label>
    <button class="primary cta" onclick={curate} disabled={loading}>
      {#if loading}
        <span class="spinner" aria-hidden="true"></span>
        Working…
      {:else}
        Get facts
        <svg class="cta-arrow" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M5 12h14M13 6l6 6-6 6" />
        </svg>
      {/if}
    </button>
  </div>

  <SourcePicker bind:url={sourceUrl} />

  <!-- Status + results below -->

  {#if showDidYouMean}
    <div transition:fade={{ duration: 120 }}>
      <DidYouMean
        {topic}
        {candidates}
        {ambiguous}
        onpick={pickCandidate}
        ondismiss={() => (showDidYouMean = false)}
      />
    </div>
  {/if}

  {#if status.msg}
    <div class="status {status.state}">{status.msg}</div>
  {/if}

  {#if loading}
    <!-- Lightweight skeletons so the panel doesn't jump while grounding runs. -->
    <div class="facts">
      {#each Array(clampFacts(maxFacts)) as _, i}
        <div class="fact skeleton" style="animation-delay:{i * 80}ms"></div>
      {/each}
    </div>
  {:else if facts.length}
    <div class="result-head">
      <span class="muted small">
        {facts.length} grounded fact{facts.length === 1 ? "" : "s"}
        {#if sourceTitle}· source: <strong>{sourceTitle}</strong>{/if}
      </span>
      <div class="result-actions">
        <button class="link" onclick={copyLink}>
          {linkCopied ? "Link copied" : "Copy link"}
        </button>
        <button class="link" onclick={otherSenses}>Other senses ▾</button>
      </div>
    </div>
    <div class="facts">
      {#each facts as fact, i (fact.fact)}
        <div in:fly={{ y: 8, duration: 220, delay: i * 60 }}>
          <FactCard {fact} />
        </div>
      {/each}
    </div>

    <div class="legend" aria-hidden="true">
      <span class="legend-item"><span class="dot good"></span> strong grounding</span>
      <span class="legend-item"><span class="dot mid"></span> moderate</span>
      <span class="legend-item"><span class="dot low"></span> weak</span>
    </div>

    <button class="link" onclick={toggleDebug}>
      {showDebug ? "Hide retrieval evidence ▴" : "Show retrieval evidence ▾"}
    </button>

    {#if showDebug}
      <div class="debug" transition:fade={{ duration: 120 }}>
        {#if debugHits === null}
          <p class="status info">Loading retrieved chunks…</p>
        {:else if debugHits.length === 0}
          <p class="status info">No chunks retrieved.</p>
        {:else}
          {#each debugHits as hit}
            <div class="hit">
              <span class="sc">{hit.score.toFixed(3)}</span> —
              <b>{hit.source_title}</b><br />
              {hit.text}
            </div>
          {/each}
        {/if}
      </div>
    {/if}
  {:else if !showDidYouMean}
    <!-- First-run / empty state: suggest topics so the app is usable instantly. -->
    <div class="quickstart" in:fade={{ duration: 160 }}>
      <p class="qs-lead">Try a topic to see grounded facts:</p>
      <div class="qs-chips">
        {#each EXAMPLES as name}
          <button class="qs-chip" onclick={() => runExample(name)}>{name}</button>
        {/each}
      </div>
    </div>
  {/if}
</section>
