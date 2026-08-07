<script>
  import { onMount } from "svelte";
  import { api } from "./lib/api.js";
  import { themeStore, toggleTheme } from "./lib/theme.svelte.js";
  import PipelineBadges from "./lib/components/PipelineBadges.svelte";
  import Tabs from "./lib/components/Tabs.svelte";
  import CuratePanel from "./lib/components/CuratePanel.svelte";
  import AskPanel from "./lib/components/AskPanel.svelte";
  import HistoryList from "./lib/components/HistoryList.svelte";
  import SubscribePanel from "./lib/components/SubscribePanel.svelte";
  import SubscriptionsPanel from "./lib/components/SubscriptionsPanel.svelte";
  import BackToTop from "./lib/components/BackToTop.svelte";

  // Pipeline backends (embedder, vector store, generation mode) shown in the
  // header. Loaded once when the app mounts.
  let info = $state(null);
  let infoError = $state(false);

  // Which section is visible. Panels stay mounted (hidden via CSS) so their
  // state — e.g. the facts you just looked up — survives switching tabs.
  let active = $state("explore");

  // Live subscription count, reported by SubscriptionsPanel, shown as a tab
  // badge. A counter bumped after subscribing tells the panel to reload.
  let subsCount = $state(null);
  let subsRefreshToken = $state(0);

  // Reference to the Explore panel so History can restore a past lookup into it.
  let curatePanel = $state(null);

  let tabs = $derived([
    { id: "explore", label: "Explore" },
    { id: "ask", label: "Ask" },
    { id: "subscribe", label: "Subscribe" },
    { id: "subscriptions", label: "Subscriptions", badge: subsCount },
  ]);

  function restoreFromHistory(item) {
    active = "explore";
    curatePanel?.restore(item);
  }

  onMount(async () => {
    try {
      info = await api("/info");
    } catch {
      infoError = true;
    }
  });
</script>

<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <span class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 40 40" width="38" height="38" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="gf-logo" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#6366f1" />
              <stop offset="1" stop-color="#a855f7" />
            </linearGradient>
          </defs>
          <rect x="1" y="1" width="38" height="38" rx="11" fill="url(#gf-logo)" />
          <!-- Check = verified; underline = grounded in a source. -->
          <path d="M11 20.5l5.2 5.2L29 13" fill="none" stroke="#fff" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round" />
          <path d="M12.5 30.5h15" stroke="rgba(255,255,255,0.6)" stroke-width="2.4" stroke-linecap="round" />
        </svg>
      </span>
      <div class="brand-text">
        <span class="brand-name">Grounded Facts</span>
        <PipelineBadges {info} {infoError} />
      </div>
    </div>
    <button
      class="theme-toggle"
      onclick={toggleTheme}
      title="Toggle light / dark theme"
      aria-label="Toggle theme"
    >
      {#if themeStore.theme === "dark"}
        <!-- Sun icon (switch to light) -->
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4.2" />
          <path d="M12 2.5v2M12 19.5v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2.5 12h2M19.5 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" />
        </svg>
      {:else}
        <!-- Moon icon (switch to dark) -->
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M20 14.5A8 8 0 1 1 9.5 4a6.3 6.3 0 0 0 10.5 10.5z" />
        </svg>
      {/if}
    </button>
  </div>
</header>

<nav class="tabbar">
  <Tabs {tabs} {active} onselect={(id) => (active = id)} />
</nav>

<main>
  <!-- Explore: search + results alongside a history sidebar. -->
  <div class="tabpanel" hidden={active !== "explore"}>
    <div class="explore-grid">
      <div class="explore-main">
        <CuratePanel bind:this={curatePanel} />
      </div>
      <HistoryList onrestore={restoreFromHistory} />
    </div>
  </div>

  <!-- Ask: single grounded answer to a free-form question, with citations. -->
  <div class="tabpanel" hidden={active !== "ask"}>
    <AskPanel />
  </div>

  <!-- Subscribe: create a recurring digest. -->
  <div class="tabpanel" hidden={active !== "subscribe"}>
    <SubscribePanel
      onsubscribed={() => {
        subsRefreshToken += 1;
        active = "subscriptions";
      }}
    />
  </div>

  <!-- Subscriptions: manage existing digests and trigger delivery. -->
  <div class="tabpanel" hidden={active !== "subscriptions"}>
    <SubscriptionsPanel
      refreshToken={subsRefreshToken}
      oncount={(n) => (subsCount = n)}
    />
  </div>
</main>

<footer>
  <span
    >Grounded Facts · FastAPI + vector search + hallucination guard · built with
    Svelte</span
  >
</footer>

<BackToTop />
