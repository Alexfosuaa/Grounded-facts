import datetime

from backend.rag import curator
from backend.services import worker


def test_worker_digest_curates_multiple_sources(temp_db, monkeypatch):
    # Each delivered digest must curate across several sources (max_sources > 1),
    # matching the /digest preview so subscribers get multi-source facts.
    db = temp_db
    past = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    db.add_subscription("u@e.com", "Physics", "daily", past)
    sub = db.list_subscriptions()[0]

    captured = {}

    def fake_curate(topic, **kwargs):
        captured.update(kwargs)
        return [
            {
                "fact": "A grounded fact.",
                "source_title": "Src",
                "source_url": "",
                "grounding_score": 0.5,
                "method": "extractive",
            }
        ]

    # Stub the slow/external bits so we exercise only the source-count wiring.
    monkeypatch.setattr(curator, "curate_facts", fake_curate)
    monkeypatch.setattr(
        worker.dedup, "filter_new_facts", lambda sid, facts: (facts, [[0.1]])
    )
    monkeypatch.setattr(worker.emailer, "try_send", lambda *a, **k: None)
    monkeypatch.setattr(worker.db, "add_sent_fact", lambda *a, **k: None)

    result = worker.process_subscription(sub)
    assert result["status"] == "sent"
    assert captured.get("max_sources", 1) > 1


def test_worker_uses_subscription_max_facts(temp_db, monkeypatch):
    # The subscriber's chosen fact count must flow into curation, not a hardcoded 3.
    db = temp_db
    past = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    db.add_subscription("u@e.com", "Physics", "daily", past, max_facts=5)
    sub = db.list_subscriptions()[0]

    captured = {}

    def fake_curate(topic, **kwargs):
        captured.update(kwargs)
        return [
            {
                "fact": "A grounded fact.",
                "source_title": "Src",
                "source_url": "",
                "grounding_score": 0.5,
                "method": "extractive",
            }
        ]

    monkeypatch.setattr(curator, "curate_facts", fake_curate)
    monkeypatch.setattr(
        worker.dedup, "filter_new_facts", lambda sid, facts: (facts, [[0.1]])
    )
    monkeypatch.setattr(worker.emailer, "try_send", lambda *a, **k: None)
    monkeypatch.setattr(worker.db, "add_sent_fact", lambda *a, **k: None)

    worker.process_subscription(sub)
    assert captured.get("max_facts") == 5


def test_worker_surfaces_send_error(temp_db, monkeypatch):
    # When SMTP delivery fails, the result must carry the real reason (so a bad
    # Gmail App Password shows up in /run-due and logs instead of a silent 0).
    db = temp_db
    past = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    db.add_subscription("u@e.com", "Physics", "daily", past)
    sub = db.list_subscriptions()[0]

    monkeypatch.setattr(
        curator,
        "curate_facts",
        lambda topic, **kwargs: [
            {
                "fact": "A grounded fact.",
                "source_title": "Src",
                "source_url": "",
                "grounding_score": 0.5,
                "method": "extractive",
            }
        ],
    )
    monkeypatch.setattr(
        worker.dedup, "filter_new_facts", lambda sid, facts: (facts, [[0.1]])
    )
    monkeypatch.setattr(
        worker.emailer,
        "try_send",
        lambda *a, **k: "SMTPAuthenticationError: (535, b'Bad credentials')",
    )

    result = worker.process_subscription(sub)
    assert result["status"] == "send_failed"
    assert "SMTPAuthenticationError" in result["error"]


def test_format_email_lists_each_source(temp_db):
    # The rendered digest body should attribute every fact to its own source, so
    # a multi-source digest visibly cites more than one article.
    facts = [
        {
            "fact": "F1.",
            "source_title": "Alpha",
            "source_url": "a",
            "grounding_score": 0.5,
        },
        {
            "fact": "F2.",
            "source_title": "Beta",
            "source_url": "b",
            "grounding_score": 0.4,
        },
    ]
    body = worker.format_email("Topic", facts)
    assert "Alpha" in body
    assert "Beta" in body


def test_run_forever_stops_on_event(monkeypatch):
    # The embedded scheduler (used by the API lifespan) must run a delivery pass
    # and then exit promptly when its stop_event is set, so shutting the web
    # service down doesn't leak a thread or hang.
    import threading

    stop = threading.Event()
    calls = {"n": 0}

    def fake_run_once(now=None):
        calls["n"] += 1
        stop.set()  # ask the loop to stop after this single pass
        return {"processed": 0}

    monkeypatch.setattr(worker.db, "init_db", lambda: None)
    monkeypatch.setattr(worker, "run_once", fake_run_once)

    worker.run_forever(poll_interval=0, stop_event=stop)

    assert calls["n"] == 1
