from backend.rag import curator


def test_curate_returns_grounded_facts(embedder, sample_sources):
    facts = curator.curate_facts(
        "Photosynthesis", max_facts=3, sources=sample_sources, embedder=embedder
    )
    assert 1 <= len(facts) <= 3
    for f in facts:
        assert f["fact"]
        assert f["method"] == "extractive"  # no API key in the test env
        assert f["grounding_score"] >= curator._grounding_threshold()
        assert f["source_title"] == "Photosynthesis"


def test_retrieve_scores_descending(embedder, sample_sources):
    hits = curator.retrieve(
        "Photosynthesis", sources=sample_sources, embedder=embedder, k=4
    )
    assert hits
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_guard_drops_unsupported_fact(embedder, sample_sources):
    hits = curator.retrieve("Photosynthesis", sources=sample_sources, embedder=embedder)
    off_topic = [
        {
            "fact": "The New York Stock Exchange crashed in October 1929.",
            "source_title": "x",
            "source_url": "",
        }
    ]
    kept = curator._apply_guard(
        off_topic, hits, embedder, curator._grounding_threshold(), "extractive"
    )
    assert kept == []


def test_guard_keeps_supported_fact(embedder, sample_sources):
    hits = curator.retrieve("Photosynthesis", sources=sample_sources, embedder=embedder)
    supported = [
        {
            "fact": "Photosynthesis releases oxygen as a byproduct.",
            "source_title": "Photosynthesis",
            "source_url": "",
        }
    ]
    kept = curator._apply_guard(
        supported, hits, embedder, curator._grounding_threshold(), "extractive"
    )
    assert len(kept) == 1
    assert kept[0]["grounding_score"] >= curator._grounding_threshold()


def test_empty_sources_returns_no_facts(embedder):
    assert curator.curate_facts("Nothing", sources=[], embedder=embedder) == []


def test_is_coherent_fact_filters_out_noise():
    # Good: a self-contained, declarative statement.
    assert curator._is_coherent_fact(
        "Photosynthesis converts light energy into chemical energy stored in glucose."
    )
    # Dangling reference + attribution/opinion.
    assert not curator._is_coherent_fact(
        "He also stated that the expansion set detracted from the game."
    )
    # Opinion + quotation (typical review sentence).
    assert not curator._is_coherent_fact(
        'Matt said the game was "easy to learn" but dry.'
    )
    # Dangling pronoun start.
    assert not curator._is_coherent_fact("It takes place in the chloroplasts of cells.")
    # Too short to be informative.
    assert not curator._is_coherent_fact("Short fact.")


def test_answer_question_prefers_affirmative_over_negated(embedder):
    # A source where a negated sentence shares the affirmative question's
    # keywords. The negation filter must keep it from becoming the answer.
    # (min_grounding is forced low so the assertion targets the *filter*, not
    # the noisy absolute score of the tiny hashing-embedder fixture.)
    sources = [
        {
            "title": "Photosynthesis",
            "url": "",
            "text": (
                "Photosynthesis produces oxygen and glucose for the plant. "
                "Some archaea perform anoxygenic photosynthesis that does not "
                "produce oxygen at all."
            ),
        }
    ]
    res = curator.answer_question(
        "What does photosynthesis produce?",
        sources=sources,
        title="Photosynthesis",
        embedder=embedder,
        min_grounding=-1.0,
    )
    assert res["grounded"] is True
    # The negated distractor must not be surfaced as the answer.
    assert curator._NEGATION_RE.search(res["answer"].lower()) is None
    assert "produces" in res["answer"].lower()


def test_answer_question_abstains_below_threshold(embedder, sample_sources):
    # With an unreachable grounding bar nothing qualifies, so the guard abstains
    # instead of guessing — the QA counterpart of the fact hallucination guard.
    res = curator.answer_question(
        "What does photosynthesis produce?",
        sources=sample_sources,
        title="Photosynthesis",
        embedder=embedder,
        min_grounding=0.99,
    )
    assert res["grounded"] is False
    assert res["answer"] is None


def test_answer_question_returns_alternatives(embedder):
    # Several distinct, coherent sentences that all answer the question should
    # produce a primary answer plus ranked "try another answer" alternatives.
    # (min_grounding forced low so every candidate qualifies with the noisy
    # hashing fixture — we're testing the alternatives plumbing, not scores.)
    sources = [
        {
            "title": "Photosynthesis",
            "url": "",
            "text": (
                "Photosynthesis produces oxygen for the atmosphere around us. "
                "Photosynthesis produces glucose that feeds the growing plant. "
                "Photosynthesis produces chemical energy stored inside sugars."
            ),
        }
    ]
    res = curator.answer_question(
        "What does photosynthesis produce?",
        sources=sources,
        title="Photosynthesis",
        embedder=embedder,
        min_grounding=-1.0,
    )
    assert res["grounded"] is True
    assert isinstance(res["alternatives"], list)
    assert len(res["alternatives"]) >= 1

    def _norm(s):
        return " ".join(s.lower().split())

    primary = _norm(res["answer"])
    for alt in res["alternatives"]:
        # Alternatives are genuinely different from the primary and each other.
        assert _norm(alt["answer"]) != primary
        assert alt["answer"]
        assert "confidence" in alt


def test_answer_question_abstain_includes_empty_alternatives(embedder, sample_sources):
    # The abstain path must still carry an (empty) alternatives list so the API
    # response shape is consistent whether or not an answer was found.
    res = curator.answer_question(
        "What does photosynthesis produce?",
        sources=sample_sources,
        title="Photosynthesis",
        embedder=embedder,
        min_grounding=0.99,
    )
    assert res["grounded"] is False
    assert res["alternatives"] == []


def test_extractive_facts_are_coherent(embedder):
    # A source mixing a clean fact with dangling/opinion noise; only the clean,
    # self-contained sentence should survive into the candidates.
    sources = [
        {
            "title": "Photosynthesis",
            "url": "",
            "text": (
                "Photosynthesis is the process that converts light energy into "
                "chemical energy in plants. He also stated that the board game was "
                'fun. A reviewer said it was "great looking" but dry.'
            ),
        }
    ]
    facts = curator.curate_facts(
        "Photosynthesis", max_facts=3, sources=sources, embedder=embedder
    )
    assert facts, "expected at least one coherent fact"
    for f in facts:
        assert not f["fact"].startswith("He ")
        assert '"' not in f["fact"]
