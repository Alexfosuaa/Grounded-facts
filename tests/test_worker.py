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
    monkeypatch.setattr(worker.emailer, "send_email", lambda *a, **k: True)
    monkeypatch.setattr(worker.db, "add_sent_fact", lambda *a, **k: None)

    result = worker.process_subscription(sub)
    assert result["status"] == "sent"
    assert captured.get("max_sources", 1) > 1


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
