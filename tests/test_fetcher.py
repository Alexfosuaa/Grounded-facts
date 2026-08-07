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
