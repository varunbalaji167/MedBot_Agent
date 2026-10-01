"""
Tests for rag_core.py.

Runs with ONLY numpy + the standard library -- no faiss, torch,
transformers, or streamlit required. That's the point: these tests verify
the actual decision logic (the in-scope/out-of-scope gate, risk-detection
fallback direction, chunk/page attribution, KB dedup) mechanically, in an
environment that can't even install the full production ML stack.

What this suite does NOT prove: that MiniLM embeddings + Phi-3/flan-t5 give
good answers to real medical questions. That depends on the actual models
and needs a real run with network access and the full dependency set (see
the bottom of this file for how to do that once you have both). What it
DOES prove: the control flow around those models -- the parts that decide
whether to trust a retrieval, whether to say "I don't know", and how risk
detection fails safely -- behaves correctly regardless of which embedding/
generation model ends up plugged in.
"""

import unittest
import numpy as np

import rag_core


# ---------------------------------------------------------------------------
# A tiny deterministic "embedding" model for tests: bag-of-words counts over
# a fixed vocabulary, L2-normalized. Two texts that share vocabulary get high
# cosine similarity; texts about unrelated topics get low similarity. This
# lets us test retrieval/confidence behavior precisely without needing a
# real semantic embedding model.
# ---------------------------------------------------------------------------

VOCAB = [
    "diabetes", "insulin", "blood", "sugar", "glucose",
    "asthma", "inhaler", "wheezing", "lungs", "breathing",
    "migraine", "headache", "light", "nausea", "aura",
]


def fake_embed_fn(texts: list[str]) -> np.ndarray:
    vecs = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for i, text in enumerate(texts):
        lower = text.lower()
        for j, word in enumerate(VOCAB):
            vecs[i, j] = lower.count(word)
    return rag_core.l2_normalize(vecs)


def build_fake_kb():
    """A 2-document, 2-chunk-each fake KB: one about diabetes, one about
    asthma. Nothing about migraines -- used as the deliberately
    out-of-scope topic in tests below."""
    chunks = [
        {"text": "Diabetes is managed with insulin and monitoring blood sugar levels.",
         "source": "diabetes.pdf", "page": 1, "doc_hash": "hash_a"},
        {"text": "High blood glucose over time can damage blood vessels and nerves.",
         "source": "diabetes.pdf", "page": 2, "doc_hash": "hash_a"},
        {"text": "Asthma causes wheezing and difficulty breathing, often treated with an inhaler.",
         "source": "asthma.pdf", "page": 1, "doc_hash": "hash_b"},
        {"text": "Avoiding triggers helps reduce asthma attacks affecting the lungs.",
         "source": "asthma.pdf", "page": 2, "doc_hash": "hash_b"},
    ]
    index = rag_core.NumpyBruteForceIndex(dim=len(VOCAB))
    chunks_by_id = {}
    embeddings = fake_embed_fn([c["text"] for c in chunks])
    ids = np.arange(len(chunks))
    index.add_with_ids(embeddings, ids)
    for cid, c in zip(ids.tolist(), chunks):
        chunks_by_id[cid] = c
    return index, chunks_by_id


class FakeGenerator:
    """Records whether it was called and returns a canned answer. Can be
    configured to raise, to simulate the LLM/API being unavailable."""

    def __init__(self, answer="Insulin helps regulate blood sugar in diabetes.", raise_on_call=False):
        self.answer = answer
        self.raise_on_call = raise_on_call
        self.calls = []

    def __call__(self, system_prompt, user_prompt, max_new_tokens=300, temperature=0.3):
        self.calls.append((system_prompt, user_prompt, max_new_tokens, temperature))
        if self.raise_on_call:
            raise RuntimeError("Simulated generator failure")
        return self.answer


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------


class TestConfidenceGate(unittest.TestCase):
    def test_clamped_to_10_98_range(self):
        self.assertEqual(rag_core.compute_confidence(-1.0), 10.0)
        self.assertEqual(rag_core.compute_confidence(0.0), 10.0)
        self.assertEqual(rag_core.compute_confidence(1.0), 98.0)
        self.assertEqual(rag_core.compute_confidence(2.5), 98.0)  # shouldn't happen with cosine sim, but must not blow past 98

    def test_mid_range_score(self):
        self.assertEqual(rag_core.compute_confidence(0.55), 55.0)

    def test_should_answer_gate(self):
        # Written relative to the configured threshold so it survives tuning.
        t = rag_core.CONFIDENCE_THRESHOLD
        self.assertTrue(rag_core.should_answer_from_context(t, True))
        self.assertTrue(rag_core.should_answer_from_context(t + 40, True))
        self.assertFalse(rag_core.should_answer_from_context(t - 0.1, True))
        self.assertFalse(rag_core.should_answer_from_context(99.0, False))  # no chunks -> never answer, regardless of confidence number


# ---------------------------------------------------------------------------
# The core behavior the user asked about directly: in-scope -> grounded
# answer with confidence; out-of-scope -> "not covered", and the LLM must
# never even be called.
# ---------------------------------------------------------------------------


class TestRetrievalAndAnswerScope(unittest.TestCase):
    def setUp(self):
        self.index, self.chunks_by_id = build_fake_kb()

    def test_in_scope_question_returns_correct_source_and_high_confidence(self):
        gen = FakeGenerator(answer="Diabetes is managed with insulin.")
        result = rag_core.generate_answer(
            self.index, self.chunks_by_id, fake_embed_fn, gen,
            "How is diabetes managed with insulin?",
        )
        self.assertEqual(result["answer"], "Diabetes is managed with insulin.")
        self.assertEqual(result["mode"], "llm")
        self.assertGreaterEqual(result["confidence"], rag_core.CONFIDENCE_THRESHOLD)
        # UI-facing relevance is the calibrated value, in the modest 55-90 band.
        self.assertGreaterEqual(result["display_confidence"], 55.0)
        self.assertLessEqual(result["display_confidence"], 90.0)
        self.assertTrue(len(result["sources"]) >= 1)
        self.assertEqual(result["sources"][0]["source"], "diabetes.pdf")
        self.assertEqual(len(gen.calls), 1)  # generator WAS called, since this is in-scope

    def test_out_of_scope_question_says_not_covered_and_never_calls_llm(self):
        # "migraine" and its related vocabulary appear nowhere in the fake KB
        # (only diabetes/asthma content), so retrieval similarity should be
        # near zero -- below the confidence threshold.
        gen = FakeGenerator(raise_on_call=True)  # if this gets called at all, the test fails loudly
        result = rag_core.generate_answer(
            self.index, self.chunks_by_id, fake_embed_fn, gen,
            "What causes a migraine aura with nausea and light sensitivity?",
        )
        self.assertEqual(result["answer"], rag_core.NOT_COVERED_MSG)
        self.assertEqual(result["mode"], "not_covered")
        self.assertEqual(result["sources"], [])
        self.assertLess(result["confidence"], rag_core.CONFIDENCE_THRESHOLD)
        self.assertEqual(result["display_confidence"], 0.0)  # no relevance figure shown beside "I don't know"
        self.assertEqual(len(gen.calls), 0)  # <-- the load-bearing assertion: LLM never invoked

    def test_generous_k_does_not_pad_context_with_unrelated_documents(self):
        # This is the exact regression found in manual testing: with a small
        # KB (4 chunks total: 2 diabetes, 2 asthma) and k=4, EVERY chunk
        # qualifies for the k slot count regardless of relevance, unless the
        # relative-score cutoff prunes the unrelated ones.
        gen = FakeGenerator(answer="Diabetes is managed with insulin and monitoring blood sugar.")
        result = rag_core.generate_answer(
            self.index, self.chunks_by_id, fake_embed_fn, gen,
            "How is diabetes managed with insulin?", k=4,
        )
        sources_from = {s["source"] for s in result["sources"]}
        self.assertIn("diabetes.pdf", sources_from)
        self.assertNotIn("asthma.pdf", sources_from)  # must be pruned despite k=4 allowing it

    def test_model_refusal_falls_back_to_extractive_kb_passage(self):
        # When retrieval SUCCEEDS but the model refuses (returns just
        # "UNAVAILABLE") -- e.g. the HF API is down and the weak local
        # fallback misfires -- the app must NOT falsely claim the topic isn't
        # in the knowledge base. It must surface the retrieved passage
        # directly, so an in-scope topic (diabetes) still gets a grounded,
        # KB-sourced answer. This is the exact failure that made asthma report
        # "not covered" in the wild.
        gen = FakeGenerator(answer="UNAVAILABLE")
        result = rag_core.generate_answer(
            self.index, self.chunks_by_id, fake_embed_fn, gen, "How is diabetes managed?",
        )
        self.assertNotEqual(result["answer"], rag_core.NOT_COVERED_MSG)
        self.assertEqual(result["mode"], "extractive")
        self.assertTrue(len(result["sources"]) >= 1)
        self.assertEqual(result["sources"][0]["source"], "diabetes.pdf")
        # the answer is literal KB text, not a hallucination
        self.assertIn("insulin", result["answer"].lower())
        self.assertEqual(len(gen.calls), 1)  # model WAS called (retrieval was in-scope)

    def test_empty_model_output_also_falls_back_to_extractive(self):
        gen = FakeGenerator(answer="")
        result = rag_core.generate_answer(
            self.index, self.chunks_by_id, fake_embed_fn, gen, "How is diabetes managed?",
        )
        self.assertEqual(result["mode"], "extractive")
        self.assertNotEqual(result["answer"], rag_core.NOT_COVERED_MSG)
        self.assertTrue(len(result["sources"]) >= 1)

    def test_unavailable_mentioned_within_a_real_answer_is_not_treated_as_refusal(self):
        # Regression test for a real bug found in manual testing: a
        # reasoning-capable model (openai/gpt-oss-20b) produced visible
        # chain-of-thought discussing the possibility of saying
        # "UNAVAILABLE" per the system prompt's own instructions, even
        # though its actual verdict was to answer normally from clearly
        # relevant, high-confidence context. A bare substring check
        # ("UNAVAILABLE" in answer) discarded that perfectly good answer.
        verbose_answer = (
            "The user is asking about diabetes management. Context mentions "
            "insulin and blood sugar monitoring. This is not a case where I "
            "should say UNAVAILABLE, since the context clearly covers it. "
            "Diabetes is managed with insulin and monitoring blood sugar levels."
        )
        gen = FakeGenerator(answer=verbose_answer)
        result = rag_core.generate_answer(
            self.index, self.chunks_by_id, fake_embed_fn, gen, "How is diabetes managed with insulin?",
        )
        self.assertEqual(result["answer"], verbose_answer)  # must NOT be replaced with NOT_COVERED_MSG
        self.assertTrue(len(result["sources"]) >= 1)

    def test_retrieval_query_separates_retrieval_from_the_prompt_question(self):
        # retrieval_query drives which chunk is retrieved; the question is what
        # the generator is asked -- a patient profile in the question must not
        # change retrieval.
        gen = FakeGenerator(answer="ok")
        result = rag_core.generate_answer(
            self.index, self.chunks_by_id, fake_embed_fn, gen,
            question="please explain next steps",
            retrieval_query="diabetes insulin blood sugar",
        )
        self.assertTrue(any(s["source"] == "diabetes.pdf" for s in result["sources"]))
        _, user_prompt, _, _ = gen.calls[0]
        self.assertIn("please explain next steps", user_prompt)

    def test_empty_kb_never_calls_llm(self):
        empty_index = rag_core.NumpyBruteForceIndex(dim=len(VOCAB))
        gen = FakeGenerator(raise_on_call=True)
        result = rag_core.generate_answer(empty_index, {}, fake_embed_fn, gen, "Anything at all?")
        self.assertEqual(result["answer"], rag_core.NOT_COVERED_MSG)
        self.assertEqual(len(gen.calls), 0)


# ---------------------------------------------------------------------------
# Chunking / page attribution / hashing
# ---------------------------------------------------------------------------


class TestChunkingAndHashing(unittest.TestCase):
    def test_hash_is_deterministic_and_content_sensitive(self):
        a = rag_core.compute_hash(b"hello world")
        b = rag_core.compute_hash(b"hello world")
        c = rag_core.compute_hash(b"hello there")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_chunk_page_attribution(self):
        # Page 1 has 10 words, page 2 has 10 words. chunk_size=8, overlap=2.
        page1 = " ".join(f"p1w{i}" for i in range(10))
        page2 = " ".join(f"p2w{i}" for i in range(10))
        chunks = rag_core.chunk_document([page1, page2], "doc.pdf", "hash1", chunk_size=8, overlap=2)

        # First chunk starts at word 0 -> page 1.
        self.assertEqual(chunks[0]["page"], 1)
        self.assertTrue(chunks[0]["text"].startswith("p1w0"))

        # Some later chunk should start on page 2 once we've advanced past
        # word index 10 (start = 0, 6, 12, ... with step chunk_size-overlap=6).
        pages_seen = {c["page"] for c in chunks}
        self.assertIn(2, pages_seen)

        # Every chunk carries the filename/hash metadata needed for citation.
        for c in chunks:
            self.assertEqual(c["source"], "doc.pdf")
            self.assertEqual(c["doc_hash"], "hash1")

    def test_quoted_section_headers_become_separate_chunks(self):
        # Topics delimited by quoted headers ("COPD":, "Asthma":) must not blend
        # into one chunk, even when they'd fit together or cross a page.
        pages = [
            '"COPD": COPD obstructs airflow. Symptoms: cough, wheezing. ,',
            '"Asthma": Asthma inflames the airways. Symptoms: wheezing, tightness.',
        ]
        chunks = rag_core.chunk_document(pages, "kb.pdf", "h", chunk_size=150, overlap=20)
        texts = [c["text"] for c in chunks]
        self.assertTrue(any("COPD" in t and "Asthma" not in t for t in texts))
        self.assertTrue(any("Asthma" in t and "COPD" not in t for t in texts))
        for t in texts:
            self.assertFalse("COPD" in t and "Asthma" in t)  # never both in one chunk

    def test_headerless_text_falls_back_to_word_chunks(self):
        chunks = rag_core.chunk_document(["plain text with no quoted headers at all"], "k.pdf", "h")
        self.assertEqual(len(chunks), 1)
        self.assertIn("plain text", chunks[0]["text"])

    def test_chunks_never_span_pages(self):
        # Per-page chunking: a chunk must contain words from exactly one page,
        # so a chunk can't blend two conditions (one condition per page here).
        page1 = " ".join(f"p1w{i}" for i in range(10))
        page2 = " ".join(f"p2w{i}" for i in range(10))
        chunks = rag_core.chunk_document([page1, page2], "doc.pdf", "h", chunk_size=8, overlap=2)
        for c in chunks:
            has_p1 = "p1w" in c["text"]
            has_p2 = "p2w" in c["text"]
            self.assertFalse(has_p1 and has_p2)  # never both pages in one chunk
            if has_p1:
                self.assertEqual(c["page"], 1)
            if has_p2:
                self.assertEqual(c["page"], 2)


# ---------------------------------------------------------------------------
# Risk triage: keyword fast-path never depends on the LLM; LLM failure falls
# back to the keyword MODERATE list, never silently to LOW.
# ---------------------------------------------------------------------------


class TestRiskTriage(unittest.TestCase):
    def test_high_risk_keyword_short_circuits_before_any_classify_call(self):
        calls = []

        def classify_fn(text):
            calls.append(text)
            return "LOW"  # even if the LLM would say LOW, keyword match must win

        result = rag_core.assess_risk("I have severe chest pain radiating to my arm", classify_fn=classify_fn)
        self.assertEqual(result["level"], "HIGH")
        self.assertEqual(calls, [])  # classify_fn must never be reached

    def test_llm_unavailable_falls_back_to_moderate_keyword_not_low(self):
        def classify_fn(text):
            raise RuntimeError("API down")

        result = rag_core.assess_risk("I've had a persistent headache and mild fever", classify_fn=classify_fn)
        self.assertEqual(result["level"], "MODERATE")

    def test_llm_unavailable_and_no_keyword_match_falls_back_to_low(self):
        def classify_fn(text):
            return None

        result = rag_core.assess_risk("I have a small paper cut on my finger", classify_fn=classify_fn)
        self.assertEqual(result["level"], "LOW")

    def test_llm_can_upgrade_a_non_keyword_phrase_to_high(self):
        # No literal HIGH_RISK phrase appears here, but the LLM's own
        # judgement should still be able to flag it.
        def classify_fn(text):
            return "HIGH"

        result = rag_core.assess_risk("crushing pressure across my chest that won't go away", classify_fn=classify_fn)
        self.assertEqual(result["level"], "HIGH")

    def test_no_classify_fn_provided_still_uses_keyword_lists(self):
        result = rag_core.assess_risk("I have a mild headache today", classify_fn=None)
        self.assertEqual(result["level"], "MODERATE")


# ---------------------------------------------------------------------------
# Conversational memory: fails open on any generator error/malformed output.
# ---------------------------------------------------------------------------


class TestCondenseQuestion(unittest.TestCase):
    def test_no_history_returns_follow_up_unchanged(self):
        gen = FakeGenerator(raise_on_call=True)
        result = rag_core.condense_question([], "What about children?", gen)
        self.assertEqual(result, "What about children?")
        self.assertEqual(len(gen.calls), 0)

    def test_generator_failure_falls_back_to_raw_follow_up(self):
        history = [
            {"role": "user", "content": "What is the dosage of ibuprofen for adults?", "plain_text": "What is the dosage of ibuprofen for adults?"},
            {"role": "assistant", "content": "...", "plain_text": "Typically 200-400mg every 4-6 hours for adults."},
        ]

        def failing_gen(system_prompt, user_prompt, max_new_tokens=300):
            raise RuntimeError("network error")

        result = rag_core.condense_question(history, "What about children?", failing_gen)
        self.assertEqual(result, "What about children?")

    def test_successful_condensation_is_used(self):
        history = [
            {"role": "user", "content": "x", "plain_text": "What is the dosage of ibuprofen for adults?"},
            {"role": "assistant", "content": "x", "plain_text": "200-400mg every 4-6 hours."},
        ]
        gen = FakeGenerator(answer="What is the dosage of ibuprofen for children?")
        result = rag_core.condense_question(history, "What about children?", gen)
        self.assertEqual(result, "What is the dosage of ibuprofen for children?")
        self.assertEqual(len(gen.calls), 1)

    def test_condensation_uses_zero_temperature(self):
        # This is a rewrite task, not a creative one -- sampling variance
        # here previously caused the exact same input to occasionally get
        # rewritten differently across runs, which showed up as an
        # intermittent, hard-to-reproduce retrieval regression.
        history = [{"role": "user", "content": "x", "plain_text": "What is COPD?"}]
        gen = FakeGenerator(answer="What are the symptoms of asthma?")
        rag_core.condense_question(history, "What are the symptoms of asthma?", gen)
        self.assertEqual(len(gen.calls), 1)
        _, _, _, temperature = gen.calls[0]
        self.assertEqual(temperature, 0.0)


# ---------------------------------------------------------------------------
# Index backend: removal actually removes only the targeted document's chunks
# ---------------------------------------------------------------------------


class TestIndexRemoval(unittest.TestCase):
    def test_remove_ids_only_affects_targeted_chunks(self):
        index, chunks_by_id = build_fake_kb()
        self.assertEqual(index.ntotal, 4)

        # Remove the two diabetes chunks (ids 0, 1), keep asthma (ids 2, 3).
        index.remove_ids(np.array([0, 1], dtype="int64"))
        self.assertEqual(index.ntotal, 2)

        gen = FakeGenerator(answer="Use an inhaler for asthma symptoms.")
        result = rag_core.generate_answer(
            index, chunks_by_id, fake_embed_fn, gen, "How is asthma treated with an inhaler?",
        )
        self.assertGreaterEqual(result["confidence"], rag_core.CONFIDENCE_THRESHOLD)
        self.assertEqual(result["sources"][0]["source"], "asthma.pdf")


# ---------------------------------------------------------------------------
# Risk triage now uses the structured severity/conditions the form collects.
# ---------------------------------------------------------------------------


class TestRiskWithStructuredSignals(unittest.TestCase):
    def test_high_severity_is_never_low(self):
        # A vague description that would otherwise score LOW must not read LOW
        # at 8/10 severity -- the exact bug (8/10 asthma case -> LOW).
        result = rag_core.assess_risk("I have been feeling generally off", severity=8)
        self.assertEqual(result["level"], "MODERATE")

    def test_chronic_condition_floors_to_moderate(self):
        result = rag_core.assess_risk(
            "I have been feeling generally off", severity=3,
            conditions=["Asthma", "Hypertension"],
        )
        self.assertEqual(result["level"], "MODERATE")

    def test_low_severity_no_conditions_stays_low(self):
        result = rag_core.assess_risk(
            "I have a small paper cut on my finger", severity=2, conditions=["None"],
        )
        self.assertEqual(result["level"], "LOW")

    def test_floors_never_downgrade_a_high_emergency(self):
        # HIGH keyword wins before any floor logic; low severity can't soften it.
        result = rag_core.assess_risk("I have chest pain", severity=1, conditions=["None"])
        self.assertEqual(result["level"], "HIGH")

    def test_freetext_condition_outside_the_set_does_not_floor(self):
        # A free-text "Other" condition not in SERIOUS_CONDITIONS leaves the
        # deterministic floor to the LLM (here absent), so it stays LOW.
        result = rag_core.assess_risk("mild thing", severity=3, conditions=["lupus"])
        self.assertEqual(result["level"], "LOW")

    def test_structured_signals_are_passed_to_classifier(self):
        seen = {}

        def classify_fn(text):
            seen["text"] = text
            return "LOW"

        rag_core.assess_risk(
            "short of breath", severity=9, conditions=["Asthma"], classify_fn=classify_fn,
        )
        self.assertIn("9/10", seen["text"])
        self.assertIn("Asthma", seen["text"])


# ---------------------------------------------------------------------------
# Intent routing: an info question must not be treated as a symptom/emergency.
# ---------------------------------------------------------------------------


class TestDetectIntent(unittest.TestCase):
    def test_info_question_with_symptom_noun_is_info(self):
        self.assertEqual(rag_core.detect_intent("What is chest pain?"), "info_query")
        self.assertEqual(rag_core.detect_intent("What causes asthma"), "info_query")
        self.assertEqual(rag_core.detect_intent("symptoms of diabetes"), "info_query")

    def test_first_person_self_report_is_symptom(self):
        self.assertEqual(rag_core.detect_intent("I have chest pain"), "symptom_query")
        self.assertEqual(rag_core.detect_intent("I've been feeling dizzy for two days"), "symptom_query")

    def test_self_report_wins_even_when_phrased_as_question(self):
        self.assertEqual(
            rag_core.detect_intent("what should I do, I have severe chest pain?"),
            "symptom_query",
        )

    def test_keyword_phrase_lookup_is_info_not_emergency(self):
        # noun-phrase info lookups must not trip the symptom/emergency path
        self.assertEqual(rag_core.detect_intent("chest pain treatment"), "info_query")
        self.assertEqual(rag_core.detect_intent("cough remedies"), "info_query")

    def test_bare_symptom_phrase_still_routes_to_symptom(self):
        # a bare complaint with no info cue stays on the safe symptom path
        self.assertEqual(rag_core.detect_intent("chest pain"), "symptom_query")


# ---------------------------------------------------------------------------
# Display confidence: UI-only calibration, independent of the gate.
# ---------------------------------------------------------------------------


class TestDisplayConfidence(unittest.TestCase):
    def test_band_endpoints_and_clamping(self):
        self.assertEqual(rag_core.display_confidence(0.30), 55.0)
        self.assertEqual(rag_core.display_confidence(0.70), 90.0)
        self.assertEqual(rag_core.display_confidence(0.10), 55.0)  # clamped up
        self.assertEqual(rag_core.display_confidence(0.95), 90.0)  # clamped down

    def test_strong_minilm_match_reads_as_strong(self):
        # cosine 0.55 displayed as a weak "55%" was the misleading behavior;
        # it should now read meaningfully higher.
        self.assertGreater(rag_core.display_confidence(0.55), 55.0)
        self.assertLess(rag_core.display_confidence(0.55), 90.0)


# ---------------------------------------------------------------------------
# Golden set: the "clear-cut" cases from the manual test plan, wired as
# deterministic checks. No ML stack needed -- detect_intent and the risk
# keyword/floor layer are pure, and scope uses the fake bag-of-words KB. Add a
# row to extend coverage.
# ---------------------------------------------------------------------------


# (text -> expected intent)
INTENT_GOLDEN = [
    ("What is COPD?", "info_query"),
    ("What are the symptoms of asthma?", "info_query"),
    ("How is diabetes treated?", "info_query"),
    ("What causes hypertension?", "info_query"),
    ("What is chest pain?", "info_query"),          # info, NOT emergency (the fix)
    ("chest pain treatment", "info_query"),
    ("cough remedies", "info_query"),
    ("symptoms of diabetes", "info_query"),
    ("I have chest pain", "symptom_query"),
    ("I have been feeling dizzy and tired for 2 days", "symptom_query"),
    ("I feel suicidal", "symptom_query"),
    ("I am coughing blood", "symptom_query"),
    ("my chest hurts", "symptom_query"),
    ("chest pain", "symptom_query"),                 # bare -> safe symptom default
    ("what should I do, I have severe chest pain?", "symptom_query"),
]

# (text, severity, conditions -> expected level), with classify_fn=None so this
# exercises the deterministic keyword + floor layers (not the LLM).
RISK_GOLDEN = [
    ("I have chest pain", None, None, "HIGH"),
    ("I have difficulty breathing", None, None, "HIGH"),
    ("I feel suicidal", None, None, "HIGH"),
    ("I am coughing blood", None, None, "HIGH"),
    ("I have a mild headache", None, None, "MODERATE"),     # MODERATE keyword
    ("I have a small paper cut", None, None, "LOW"),
    ("I have a cough", 9, ["None"], "MODERATE"),            # severity floor
    ("I have a cough", 3, ["Asthma"], "MODERATE"),          # condition floor
    ("I have a cough", 2, ["None"], "LOW"),                 # neither
    ("I feel a bit off", 8, None, "MODERATE"),              # severity floor only
    ("I have a cough", 3, ["lupus"], "LOW"),                # free-text, not serious
]

# (text -> is it in-scope for the fake diabetes/asthma KB?)
SCOPE_GOLDEN = [
    ("How is diabetes managed with insulin?", True),
    ("asthma wheezing and inhaler use", True),
    ("What causes a migraine aura with nausea?", False),    # not in the fake KB
    ("capital of france", False),
    ("random unrelated query about cars", False),
]


class TestGoldenSet(unittest.TestCase):
    def test_intent_routing(self):
        for text, expected in INTENT_GOLDEN:
            with self.subTest(text=text):
                self.assertEqual(rag_core.detect_intent(text), expected)

    def test_risk_floors_and_keywords(self):
        for text, severity, conditions, expected in RISK_GOLDEN:
            with self.subTest(text=text, severity=severity, conditions=conditions):
                result = rag_core.assess_risk(text, severity=severity, conditions=conditions)
                self.assertEqual(result["level"], expected)

    def test_scope_decision(self):
        index, chunks_by_id = build_fake_kb()
        for text, in_scope in SCOPE_GOLDEN:
            with self.subTest(text=text, in_scope=in_scope):
                gen = FakeGenerator(answer="A grounded answer from the knowledge base.")
                result = rag_core.generate_answer(index, chunks_by_id, fake_embed_fn, gen, text)
                if in_scope:
                    self.assertNotEqual(result["mode"], "not_covered")
                    self.assertTrue(len(result["sources"]) >= 1)
                else:
                    self.assertEqual(result["mode"], "not_covered")
                    self.assertEqual(result["sources"], [])
                    self.assertEqual(len(gen.calls), 0)  # generator never called out-of-scope


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# To validate actual answer QUALITY (not just this control flow) once you
# have network access and the full stack installed:
#
#   1. pip install -r requirements.txt
#   2. Run the Streamlit app locally with a real PDF loaded.
#   3. Build a small golden set: 10-15 questions you know the PDF answers,
#      plus 5-10 you know it does NOT answer, with the expected source page
#      for each "should answer" case.
#   4. For each, check: did it retrieve the expected page? Is confidence
#      reasonably high for in-scope and low for out-of-scope? Does the
#      generated answer actually match the source text (not hallucinate)?
#
# That's a manual-but-structured eval pass -- worth turning into a scripted
# one later (e.g. asserting retrieved page == expected page for each golden
# question), but it needs the real embedding model, so it can't run in an
# environment without the ML stack installed.
# ---------------------------------------------------------------------------