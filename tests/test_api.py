from fastapi.testclient import TestClient

from backend import api
from backend.rag import curator, fetcher


def _client(temp_db):
    # temp_db has already pointed db.DB_PATH at an isolated file.
    return TestClient(api.app)


def test_health(temp_db):
    with _client(temp_db) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_info(temp_db):
    with _client(temp_db) as client:
        r = client.get("/api/info")
        assert r.status_code == 200
        body = r.json()
        assert "embedder" in body and "vector_backend" in body


def test_preview(temp_db, monkeypatch):
    fake = [
        {
            "fact": "A grounded fact.",
            "source_title": "Src",
            "source_url": "http://x",
            "grounding_score": 0.5,
            "method": "extractive",
        }
    ]
    monkeypatch.setattr(curator, "curate_facts", lambda *a, **k: fake)
    with _client(temp_db) as client:
        r = client.get("/api/preview", params={"topic": "Physics", "max_facts": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["topic"] == "Physics"
        assert body["facts"][0]["fact"] == "A grounded fact."


def test_digest_preview(temp_db, monkeypatch):
    # The digest endpoint composes the email a subscriber would receive without
    # any side effects, so it should surface a subject, a rendered body, and the
    # facts — plus the dry-run flag so the UI can explain demo mode.
    fake = [
        {
            "fact": "A grounded fact.",
            "source_title": "Src",
            "source_url": "http://x",
            "grounding_score": 0.5,
            "method": "extractive",
        }
    ]
    monkeypatch.setattr(curator, "curate_facts", lambda *a, **k: fake)
    with _client(temp_db) as client:
        r = client.get("/api/digest", params={"topic": "Physics", "max_facts": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["subject"] == "Your facts about Physics"
        assert "A grounded fact." in body["body"]
        assert body["facts"][0]["fact"] == "A grounded fact."
        assert isinstance(body["dry_run"], bool)


def test_digest_requires_topic(temp_db):
    with _client(temp_db) as client:
        r = client.get("/api/digest", params={"topic": "   "})
        assert r.status_code == 400


def test_digest_uses_multiple_sources(temp_db, monkeypatch):
    # The digest curates on the subscriber's behalf across several sources, so it
    # must ask curate_facts for more than one article (max_sources > 1).
    captured = {}

    def fake_curate(topic, **kwargs):
        captured.update(kwargs)
        return [
            {
                "fact": "A grounded fact.",
                "source_title": "Src",
                "source_url": "http://x",
                "grounding_score": 0.5,
                "method": "extractive",
            }
        ]

    monkeypatch.setattr(curator, "curate_facts", fake_curate)
    with _client(temp_db) as client:
        r = client.get("/api/digest", params={"topic": "Physics"})
        assert r.status_code == 200
        assert captured.get("max_sources", 1) > 1


def test_ask_answers(temp_db, monkeypatch):
    # The ask endpoint returns a grounded answer with citations. We stub the
    # curator so the test stays offline and deterministic.
    canned = {
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis converts light energy into chemical energy.",
        "grounded": True,
        "confidence": 0.61,
        "source_title": "Photosynthesis",
        "source_url": "http://x",
        "citations": [
            {
                "source_title": "Photosynthesis",
                "source_url": "http://x",
                "snippet": "Photosynthesis is a process...",
                "score": 0.7,
            }
        ],
    }
    monkeypatch.setattr(curator, "answer_question", lambda *a, **k: canned)
    with _client(temp_db) as client:
        r = client.get("/api/ask", params={"question": "What is photosynthesis?"})
        assert r.status_code == 200
        body = r.json()
        assert body["grounded"] is True
        assert body["answer"].startswith("Photosynthesis converts")
        assert body["citations"][0]["source_title"] == "Photosynthesis"


def test_ask_abstains(temp_db, monkeypatch):
    # When nothing is grounded, the endpoint reports grounded=False with no answer.
    canned = {
        "question": "asdfqwer nonsense",
        "answer": None,
        "grounded": False,
        "confidence": 0.0,
        "source_title": "",
        "source_url": "",
        "citations": [],
    }
    monkeypatch.setattr(curator, "answer_question", lambda *a, **k: canned)
    with _client(temp_db) as client:
        r = client.get("/api/ask", params={"question": "asdfqwer nonsense"})
        assert r.status_code == 200
        body = r.json()
        assert body["grounded"] is False
        assert body["answer"] is None


def test_ask_requires_question(temp_db):
    with _client(temp_db) as client:
        r = client.get("/api/ask", params={"question": "   "})
        assert r.status_code == 400


def test_subscribe_lifecycle(temp_db):
    with _client(temp_db) as client:
        r = client.post(
            "/api/subscribe",
            json={"email": "user@example.com", "topic": "Physics", "cadence": "daily"},
        )
        assert r.status_code == 200

        r = client.get("/api/subscriptions")
        subs = r.json()["subscriptions"]
        assert len(subs) == 1
        sub_id = subs[0]["id"]
        # The list endpoint must not leak the full address.
        assert subs[0]["email"] != "user@example.com"
        assert "@example.com" in subs[0]["email"]

        r = client.delete(f"/api/subscriptions/{sub_id}")
        assert r.status_code == 200
        assert client.get("/api/subscriptions").json()["subscriptions"] == []


def test_subscribe_rejects_bad_cadence(temp_db):
    with _client(temp_db) as client:
        r = client.post(
            "/api/subscribe",
            json={"email": "user@example.com", "topic": "Physics", "cadence": "yearly"},
        )
        assert r.status_code == 400


def test_subscribe_custom_max_facts(temp_db):
    # The subscriber can choose how many facts each digest carries; it is echoed
    # back and persisted on the subscription row.
    with _client(temp_db) as client:
        r = client.post(
            "/api/subscribe",
            json={
                "email": "user@example.com",
                "topic": "Physics",
                "cadence": "daily",
                "max_facts": 5,
            },
        )
        assert r.status_code == 200
        assert r.json()["max_facts"] == 5
        subs = client.get("/api/subscriptions").json()["subscriptions"]
        assert subs[0]["max_facts"] == 5


def test_subscribe_rejects_out_of_range_max_facts(temp_db):
    # Only 2–5 is allowed, so a digest email stays focused.
    with _client(temp_db) as client:
        for bad in (1, 6):
            r = client.post(
                "/api/subscribe",
                json={"email": "u@example.com", "topic": "Physics", "max_facts": bad},
            )
            assert r.status_code == 422


def test_run_due_with_no_subscriptions(temp_db):
    with _client(temp_db) as client:
        r = client.post("/api/run-due")
        assert r.status_code == 200
        assert r.json()["processed"] == 0


def test_disambiguate(temp_db, monkeypatch):
    fake = [
        {"title": "Photosynthesis", "description": "A biological process."},
        {"title": "Photosynthesis (board game)", "description": "A 2017 board game."},
    ]
    monkeypatch.setattr(fetcher, "fetch_candidates", lambda *a, **k: fake)
    monkeypatch.setattr(fetcher, "is_ambiguous", lambda *a, **k: True)
    with _client(temp_db) as client:
        r = client.get("/api/disambiguate", params={"topic": "Photosynthesis"})
        assert r.status_code == 200
        body = r.json()
        assert body["topic"] == "Photosynthesis"
        assert body["ambiguous"] is True
        assert body["candidates"][1]["title"] == "Photosynthesis (board game)"


def test_preview_with_custom_source(temp_db, monkeypatch):
    # A source_url is fetched and injected so Wikipedia is bypassed; curate_facts
    # must receive the fetched source.
    fake_src = {"title": "Blog", "url": "https://ex.com/a", "text": "Some text."}
    monkeypatch.setattr(fetcher, "fetch_url_source", lambda url, **k: fake_src)
    captured = {}

    def fake_curate(topic, **kwargs):
        captured["sources"] = kwargs.get("sources")
        return [
            {
                "fact": "From the blog.",
                "source_title": "Blog",
                "source_url": "https://ex.com/a",
                "grounding_score": 0.5,
                "method": "extractive",
            }
        ]

    monkeypatch.setattr(curator, "curate_facts", fake_curate)
    with _client(temp_db) as client:
        r = client.get(
            "/api/preview",
            params={"topic": "X", "source_url": "https://ex.com/a"},
        )
        assert r.status_code == 200
        assert captured["sources"] == [fake_src]
        assert r.json()["facts"][0]["source_title"] == "Blog"


def test_preview_rejects_unfetchable_source(temp_db, monkeypatch):
    # When the URL can't be fetched safely we return 400 rather than silently
    # falling back to Wikipedia — the user explicitly asked for that source.
    monkeypatch.setattr(fetcher, "fetch_url_source", lambda url, **k: None)
    with _client(temp_db) as client:
        r = client.get(
            "/api/preview",
            params={"topic": "X", "source_url": "http://127.0.0.1/"},
        )
        assert r.status_code == 400


def test_ask_with_custom_source(temp_db, monkeypatch):
    fake_src = {"title": "Blog", "url": "https://ex.com/a", "text": "Some text."}
    monkeypatch.setattr(fetcher, "fetch_url_source", lambda url, **k: fake_src)
    captured = {}

    def fake_answer(question, **kwargs):
        captured["sources"] = kwargs.get("sources")
        return {
            "question": question,
            "answer": "An answer.",
            "grounded": True,
            "confidence": 0.6,
            "source_title": "Blog",
            "source_url": "https://ex.com/a",
            "citations": [],
            "alternatives": [],
        }

    monkeypatch.setattr(curator, "answer_question", fake_answer)
    with _client(temp_db) as client:
        r = client.get(
            "/api/ask",
            params={"question": "Q?", "source_url": "https://ex.com/a"},
        )
        assert r.status_code == 200
        assert captured["sources"] == [fake_src]
        assert r.json()["alternatives"] == []
