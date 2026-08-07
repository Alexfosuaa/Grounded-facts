import datetime


def test_subscription_crud(temp_db):
    db = temp_db
    db.add_subscription("a@b.com", "Physics", "daily", datetime.datetime.utcnow())
    subs = db.list_subscriptions()
    assert len(subs) == 1
    assert subs[0]["topic"] == "Physics"

    db.delete_subscription(subs[0]["id"])
    assert db.list_subscriptions() == []


def test_due_subscriptions(temp_db):
    db = temp_db
    past = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    future = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    db.add_subscription("due@b.com", "A", "daily", past)
    db.add_subscription("later@b.com", "B", "daily", future)

    due = db.get_due_subscriptions(datetime.datetime.utcnow())
    emails = {d["email"] for d in due}
    assert "due@b.com" in emails
    assert "later@b.com" not in emails


def test_sent_facts_roundtrip(temp_db):
    db = temp_db
    db.add_subscription("a@b.com", "A", "daily", datetime.datetime.utcnow())
    sub_id = db.list_subscriptions()[0]["id"]
    db.add_sent_fact(sub_id, "a fact", [0.1, 0.2, 0.3])
    db.add_sent_fact(sub_id, "another fact", [0.4, 0.5, 0.6])
    embs = db.get_sent_fact_embeddings(sub_id)
    assert len(embs) == 2
    assert embs[0] == [0.1, 0.2, 0.3]
