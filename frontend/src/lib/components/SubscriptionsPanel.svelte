<script>
  import { api } from "../api.js";

  // `refreshToken` is a counter the parent bumps to request a reload (e.g. after
  // a new subscription is created elsewhere in the app). `oncount` reports the
  // current number of subscriptions back to the parent for the tab badge.
  let { refreshToken, oncount } = $props();

  let subs = $state([]);
  let status = $state({ msg: "", state: "" });
  let runStatus = $state({ msg: "", state: "" });
  let running = $state(false);
  let dryRun = $state(false); // email delivery is logged, not sent (demo mode)
  let preview = $state(null); // { id, subject, body } of the open digest preview
  let previewing = $state(null); // id whose digest is currently loading

  // GET /api/subscriptions and report the count back to the parent (tab badge).
  async function load() {
    try {
      const data = await api("/subscriptions");
      subs = data.subscriptions || [];
      oncount?.(subs.length);
    } catch (err) {
      status = { msg: err.message, state: "err" };
    }
  }

  // Detect demo/dry-run mode so the UI can explain why no real email arrives.
  async function loadInfo() {
    try {
      const info = await api("/info");
      dryRun = !!info.email_dry_run;
    } catch {
      /* non-critical */
    }
  }

  // DELETE /api/subscriptions/{id}, then reload the list.
  async function remove(id) {
    if (preview?.id === id) preview = null;
    try {
      await api(`/subscriptions/${id}`, { method: "DELETE" });
      await load();
    } catch (err) {
      runStatus = { msg: err.message, state: "err" };
    }
  }

  // Fetch (and toggle) the digest a subscriber would receive — no side effects,
  // so it works as a live preview even when email is in dry-run mode.
  async function previewDigest(sub) {
    if (preview?.id === sub.id) {
      preview = null;
      return;
    }
    previewing = sub.id;
    try {
      const d = await api(`/digest?topic=${encodeURIComponent(sub.topic)}`);
      preview = { id: sub.id, subject: d.subject, body: d.body };
      dryRun = !!d.dry_run;
    } catch (err) {
      runStatus = { msg: err.message, state: "err" };
    } finally {
      previewing = null;
    }
  }

  // POST /api/run-due to trigger a single delivery pass for due subscriptions.
  async function runDue() {
    running = true;
    runStatus = { msg: "Running delivery for due subscriptions…", state: "info" };
    try {
      const data = await api("/run-due", { method: "POST" });
      const note = dryRun ? " (demo mode: logged, not emailed)" : "";
      runStatus = {
        msg: `Processed ${data.processed} subscription(s); sent ${data.sent} digest(s).${note}`,
        state: "ok",
      };
    } catch (err) {
      runStatus = { msg: err.message, state: "err" };
    } finally {
      running = false;
    }
  }

  // Reload on mount and whenever the parent bumps refreshToken. Reading
  // `refreshToken` here registers it as a reactive dependency of the effect.
  $effect(() => {
    refreshToken;
    load();
    loadInfo();
  });
</script>

<section class="card">
  <div class="row between">
    <h2>Subscriptions</h2>
    <button class="secondary" onclick={runDue} disabled={running}>
      {running ? "Running…" : "Run delivery now"}
    </button>
  </div>

  {#if dryRun}
    <div class="demo-banner">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M4 4h16v16H4z" />
        <path d="M4 7l8 6 8-6" />
      </svg>
      <span>
        <b>Demo mode.</b> Digests are generated and shown here instead of being emailed.
        Set <code>SMTP_HOST</code> and restart to send real email. Use
        <b>Preview digest</b> to see exactly what a subscriber receives.
      </span>
    </div>
  {/if}

  {#if runStatus.msg}
    <div class="status {runStatus.state}">{runStatus.msg}</div>
  {/if}

  {#if status.msg}
    <div class="status {status.state}">{status.msg}</div>
  {/if}

  <div class="subs">
    {#if subs.length === 0}
      <p class="status info">No subscriptions yet.</p>
    {:else}
      {#each subs as sub}
        <div class="sub-item">
          <div class="sub">
            <span class="desc">
              <b>{sub.topic}</b> → {sub.email} · {sub.cadence}
            </span>
            <div class="sub-actions">
              <button
                class="secondary sub-preview"
                onclick={() => previewDigest(sub)}
                disabled={previewing === sub.id}
              >
                {previewing === sub.id
                  ? "Loading…"
                  : preview?.id === sub.id
                    ? "Hide digest"
                    : "Preview digest"}
              </button>
              <button
                class="sub-delete"
                onclick={() => remove(sub.id)}
                aria-label="Delete subscription">Delete</button
              >
            </div>
          </div>
          {#if preview?.id === sub.id}
            <div class="digest-preview">
              <div class="digest-subject">{preview.subject}</div>
              <pre class="digest-body">{preview.body}</pre>
            </div>
          {/if}
        </div>
      {/each}
    {/if}
  </div>
</section>
