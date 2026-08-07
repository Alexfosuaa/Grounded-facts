from backend.rag import fetcher


class FakePage:
    """Minimal stand-in for a wikipedia.page object."""

    def __init__(self, title, content="Photosynthesis is a process. " * 20):
        self.title = title
        self.url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        self.content = content


def test_fetch_sources_follows_redirect_to_primary(monkeypatch):
    # A redirect/abbreviation ("AI") must resolve to Wikipedia's primary article
    # ("Artificial intelligence"), never a minor namesake ("Ai" the chimpanzee).
    def fake_page(title, auto_suggest=False):
        resolved = "Artificial intelligence" if title == "AI" else title
        return FakePage(resolved)

    monkeypatch.setattr(fetcher.wikipedia, "page", fake_page)
    # Search would rank the namesake highly; it must not be consulted here.
    monkeypatch.setattr(
        fetcher.wikipedia,
        "search",
        lambda topic, results=6: ["Ai", "Artificial intelligence"],
    )
    sources = fetcher.fetch_sources("AI")
    assert [s["title"] for s in sources] == ["Artificial intelligence"]


def test_fetch_sources_disambiguation_falls_back_to_search(monkeypatch):
    # A genuinely ambiguous bare title ("Mercury") raises DisambiguationError; we
    # must NOT silently pick an option — we skip the bare title and ground on the
    # first concrete search result instead.
    def fake_page(title, auto_suggest=False):
        if title == "Mercury":
            raise fetcher.wikipedia.exceptions.DisambiguationError(
                "Mercury", ["Mercury (planet)", "Mercury (element)"]
            )
        return FakePage(title)

    monkeypatch.setattr(fetcher.wikipedia, "page", fake_page)
    monkeypatch.setattr(
        fetcher.wikipedia,
        "search",
        lambda topic, results=6: ["Mercury", "Mercury (planet)", "Mercury (element)"],
    )
    sources = fetcher.fetch_sources("Mercury")
    assert [s["title"] for s in sources] == ["Mercury (planet)"]


def test_fetch_sources_grounds_on_single_article(monkeypatch):
    # The straightforward case: the topic is its own primary-topic article.
    monkeypatch.setattr(
        fetcher.wikipedia, "page", lambda title, auto_suggest=False: FakePage(title)
    )
    sources = fetcher.fetch_sources("Photosynthesis")
    assert [s["title"] for s in sources] == ["Photosynthesis"]


def test_fetch_sources_with_explicit_title(monkeypatch):
    # When a title is pinned (user picked it from disambiguation), fetch exactly
    # that page — even a qualified one — and nothing else.
    monkeypatch.setattr(
        fetcher.wikipedia, "page", lambda title, auto_suggest=False: FakePage(title)
    )
    sources = fetcher.fetch_sources(
        "Photosynthesis", title="Photosynthesis (board game)"
    )
    assert len(sources) == 1
    assert sources[0]["title"] == "Photosynthesis (board game)"


def test_fetch_candidates(monkeypatch):
    monkeypatch.setattr(
        fetcher.wikipedia, "search", lambda topic, results=6: ["Alpha", "Beta"]
    )
    monkeypatch.setattr(
        fetcher.wikipedia,
        "summary",
        lambda t, sentences=1, auto_suggest=False: f"desc {t}",
    )
    candidates = fetcher.fetch_candidates("x")
    assert candidates == [
        {"title": "Alpha", "description": "desc Alpha"},
        {"title": "Beta", "description": "desc Beta"},
    ]


# --- Custom (non-Wikipedia) source fetching --------------------------------


def test_is_safe_public_url_rejects_unsafe():
    # SSRF guard: private/loopback/link-local/metadata IPs and non-http(s)
    # schemes must all be rejected. IP literals need no DNS, so this is offline.
    for url in [
        "http://127.0.0.1/admin",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://8.8.8.8/",
        "",
        "not-a-url",
    ]:
        assert fetcher._is_safe_public_url(url) is False, url


def test_is_safe_public_url_allows_public_ip():
    # A public IP literal resolves without DNS, keeping the test offline.
    assert fetcher._is_safe_public_url("http://8.8.8.8/") is True


def test_extract_readable_strips_boilerplate():
    html = """
    <html><head><title>My Page</title></head>
    <body>
      <nav>Home About Contact</nav>
      <script>var secret = 1;</script>
      <article><h1>Heading</h1><p>The main content is here and readable.</p></article>
      <footer>Copyright 2024</footer>
    </body></html>
    """
    src = fetcher._extract_readable(html, "https://example.com/a", 9000)
    assert src is not None
    assert src["title"] == "My Page"
    assert src["url"] == "https://example.com/a"
    assert "main content is here" in src["text"]
    assert "Home About" not in src["text"]  # nav stripped
    assert "secret" not in src["text"]  # script stripped
    assert "Copyright" not in src["text"]  # footer stripped


def test_fetch_url_source_reads_public_page(monkeypatch):
    # Bypass the network by faking a safe URL and a streamed HTTP response.
    monkeypatch.setattr(fetcher, "_is_safe_public_url", lambda url: True)

    class FakeResp:
        encoding = "utf-8"
        is_redirect = False
        headers: dict = {}

        def raise_for_status(self):
            pass

        def close(self):
            pass

        def iter_content(self, chunk_size=16384):
            yield (
                b"<html><head><title>T</title></head><body>"
                b"<article><p>Readable content here.</p></article></body></html>"
            )

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    src = fetcher.fetch_url_source("https://example.com/a")
    assert src is not None
    assert src["title"] == "T"
    assert "Readable content here" in src["text"]


def test_is_safe_public_url_rejects_ambiguous():
    # Parser/client disagreements that enable SSRF bypasses must be rejected up
    # front, regardless of the (public-looking) host they appear to point at.
    assert fetcher._is_safe_public_url("http://127.0.0.1:80\\@8.8.8.8/") is False
    assert fetcher._is_safe_public_url("http://8.8.8.8@127.0.0.1/") is False


def test_fetch_url_source_rejects_redirect_to_internal(monkeypatch):
    # A public URL that 302-redirects to an internal address must be refused: the
    # guard re-validates every hop, not just the originally supplied URL.
    import requests

    class RedirectResp:
        is_redirect = True
        headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

        def close(self):
            pass

    real_guard = fetcher._is_safe_public_url
    # Only the original URL is treated as safe; the redirect target (link-local
    # cloud-metadata address) is judged by the real guard and must be rejected.
    monkeypatch.setattr(
        fetcher,
        "_is_safe_public_url",
        lambda u: u == "https://example.com/a" or real_guard(u),
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: RedirectResp())
    assert fetcher.fetch_url_source("https://example.com/a") is None


def test_fetch_url_source_rejects_unsafe_without_network(monkeypatch):
    # An unsafe URL must be rejected *before* any network call is attempted.
    import requests

    def boom(*a, **k):
        raise AssertionError("network must not be touched for an unsafe URL")

    monkeypatch.setattr(requests, "get", boom)
    assert fetcher.fetch_url_source("http://127.0.0.1/") is None
