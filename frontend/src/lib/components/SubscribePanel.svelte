<script>
  import { api } from "../api.js";

  // `onsubscribed` is a callback prop the parent passes so it can refresh the
  // subscriptions list after a new subscription is created.
  let { onsubscribed } = $props();

  let topic = $state("");
  let email = $state("");
  let cadence = $state("daily");
  let maxFacts = $state(3);
  let status = $state({ msg: "", state: "" });

  // POST /api/subscribe with the form values.
  async function subscribe() {
    if (!topic.trim() || !email.trim()) {
      status = { msg: "Topic and email are both required.", state: "err" };
      return;
    }
    // Clamp to the 2–5 range the API accepts so a stray value doesn't 422.
    const facts = Math.min(5, Math.max(2, Math.round(Number(maxFacts) || 3)));
    try {
      await api("/subscribe", {
        method: "POST",
        body: JSON.stringify({ topic, email, cadence, max_facts: facts }),
      });
      status = { msg: "Subscribed! You'll receive grounded digests.", state: "ok" };
      topic = "";
      email = "";
      onsubscribed?.();
    } catch (err) {
      status = { msg: err.message, state: "err" };
    }
  }
</script>

<section class="card">
  <div class="panel-head">
    <h2>Subscribe to a digest</h2>
    <p class="muted small">
      Get fresh grounded facts delivered on a schedule — curated for you across
      several related sources. No account needed.
    </p>
  </div>

  <div class="form">
    <label class="field">
      <span class="field-label">Topic</span>
      <input type="text" placeholder="e.g. Machine learning" bind:value={topic} />
    </label>
    <div class="field-row">
      <label class="field grow">
        <span class="field-label">Email</span>
        <input type="email" placeholder="you@example.com" bind:value={email} />
      </label>
      <label class="field">
        <span class="field-label">Cadence</span>
        <select bind:value={cadence}>
          <option value="daily">Daily</option>
          <option value="hourly">Hourly</option>
          <option value="weekly">Weekly</option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">Facts</span>
        <select bind:value={maxFacts}>
          <option value={2}>2</option>
          <option value={3}>3</option>
          <option value={4}>4</option>
          <option value={5}>5</option>
        </select>
      </label>
    </div>
    <div class="form-actions">
      <button class="primary cta" onclick={subscribe}>
        Subscribe
        <svg class="cta-arrow" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M5 12h14M13 6l6 6-6 6" />
        </svg>
      </button>
    </div>
  </div>

  {#if status.msg}
    <div class="status {status.state}">{status.msg}</div>
  {/if}
</section>
