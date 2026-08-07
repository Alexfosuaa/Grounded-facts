"""Standalone delivery worker.

This is the scheduler that was previously embedded inside the Streamlit script
(where every browser rerun risked spawning a duplicate scheduler). Pulling it
into its own process makes delivery reliable and independently deployable
(see ``docker-compose.yml``), while :func:`run_once` stays importable so the API,
the UI "Run due now" button, and the tests can all trigger a single pass.
"""

from __future__ import annotations

# Load .env before importing modules that read configuration at import time.
from dotenv import load_dotenv

load_dotenv()

import datetime
import time
from typing import Dict, List, Optional

from backend.rag import curator
from backend.services import db, dedup, emailer

CADENCE_DELTAS = {
    "hourly": datetime.timedelta(hours=1),
    "daily": datetime.timedelta(days=1),
    "weekly": datetime.timedelta(weeks=1),
}


def format_email(topic: str, facts: List[Dict]) -> str:
    """Render the plain-text digest body: each fact with its source and score."""
    lines = [f"Here are {len(facts)} fresh fact(s) about {topic}:", ""]
    for i, fact in enumerate(facts, 1):
        lines.append(f"{i}. {fact['fact']}")
        source = fact.get("source_title") or "Wikipedia"
        url = fact.get("source_url") or ""
        score = fact.get("grounding_score")
        cite = f"   Source: {source}"
        if url:
            cite += f" — {url}"
        if score is not None:
            cite += f" (grounding {score})"
        lines.append(cite)
        lines.append("")
    lines.append("To unsubscribe, open the app and remove your subscription.")
    return "\n".join(lines)


def process_subscription(sub: Dict, now: Optional[datetime.datetime] = None) -> Dict:
    """Curate, dedupe, and (attempt to) deliver facts for a single subscription."""
    now = now or datetime.datetime.utcnow()
    topic = sub["topic"]
    delta = CADENCE_DELTAS.get(sub.get("cadence", "daily"), CADENCE_DELTAS["daily"])

    # Claim the row *before* any slow curation/sending so a concurrent worker
    # poll and an API /run-due can't both deliver it. If another pass already
    # advanced it, we don't win the claim and simply skip.
    if not db.claim_due_subscription(sub["id"], now + delta, now):
        return {"id": sub["id"], "topic": topic, "status": "skipped", "sent": 0}

    try:
        # Curate across several related articles so the digest draws on multiple
        # sources rather than a single page.
        facts = curator.curate_facts(
            topic, max_facts=3, max_sources=curator.DIGEST_MAX_SOURCES
        )
        kept, embeddings = dedup.filter_new_facts(sub["id"], facts)

        if not kept:
            return {
                "id": sub["id"],
                "topic": topic,
                "status": "no_new_facts",
                "sent": 0,
            }

        subject = f"Your facts about {topic}"
        body = format_email(topic, kept)
        if emailer.send_email(sub["email"], subject, body):
            for fact, emb in zip(kept, embeddings, strict=True):
                db.add_sent_fact(sub["id"], fact["fact"], emb)
            return {
                "id": sub["id"],
                "topic": topic,
                "status": "sent",
                "sent": len(kept),
            }

        # Delivery failed. The row was already advanced by the claim, so it is
        # retried on the next cadence rather than immediately (at-most-once).
        return {"id": sub["id"], "topic": topic, "status": "send_failed", "sent": 0}
    except Exception as exc:  # one bad subscription must not kill the whole pass
        print(f"[worker] error processing subscription {sub['id']}: {exc}")
        return {"id": sub["id"], "topic": topic, "status": "error", "sent": 0}


def run_once(now: Optional[datetime.datetime] = None) -> Dict:
    """Process every due subscription exactly once. Returns a summary."""
    now = now or datetime.datetime.utcnow()
    due = db.get_due_subscriptions(now)
    results = [process_subscription(sub, now) for sub in due]
    return {
        "processed": len(results),
        "sent": sum(1 for r in results if r["status"] == "sent"),
        "no_new_facts": sum(1 for r in results if r["status"] == "no_new_facts"),
        "failed": sum(1 for r in results if r["status"] == "send_failed"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }


def run_forever(poll_interval: int = 60) -> None:  # pragma: no cover - long running
    db.init_db()
    print(f"[worker] started; polling every {poll_interval}s")
    while True:
        summary = run_once()
        if summary["processed"]:
            print(f"[worker] {datetime.datetime.utcnow().isoformat()} {summary}")
        time.sleep(poll_interval)


if __name__ == "__main__":  # pragma: no cover
    import os

    run_forever(int(os.getenv("SCHED_POLL_INTERVAL", "60")))
