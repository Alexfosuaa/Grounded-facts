import datetime

from backend.services import dedup


def _facts():
    return [
        {
            "fact": "Photosynthesis releases oxygen as a byproduct.",
            "source_title": "P",
            "source_url": "",
        },
        {
            "fact": "Chlorophyll absorbs light in the blue and red spectrum.",
            "source_title": "P",
            "source_url": "",
        },
    ]


def test_first_send_keeps_all(temp_db, embedder):
    db = temp_db
    db.add_subscription(
        "a@b.com", "Photosynthesis", "daily", datetime.datetime.utcnow()
    )
    sub_id = db.list_subscriptions()[0]["id"]
    kept, embs = dedup.filter_new_facts(sub_id, _facts(), embedder)
    assert len(kept) == 2
    assert len(embs) == 2


def test_second_send_filters_duplicates(temp_db, embedder):
    db = temp_db
    db.add_subscription(
        "a@b.com", "Photosynthesis", "daily", datetime.datetime.utcnow()
    )
    sub_id = db.list_subscriptions()[0]["id"]

    kept, embs = dedup.filter_new_facts(sub_id, _facts(), embedder)
    for fact, emb in zip(kept, embs, strict=True):
        db.add_sent_fact(sub_id, fact["fact"], emb)

    kept2, _ = dedup.filter_new_facts(sub_id, _facts(), embedder)
    assert kept2 == []


def test_intra_batch_dedup(temp_db, embedder):
    db = temp_db
    db.add_subscription("a@b.com", "T", "daily", datetime.datetime.utcnow())
    sub_id = db.list_subscriptions()[0]["id"]
    duplicated = _facts() + [_facts()[0]]  # same first fact twice in one batch
    kept, _ = dedup.filter_new_facts(sub_id, duplicated, embedder)
    assert len(kept) == 2
