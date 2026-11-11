"""The colloquial-query-expansion capability: mapping, probes, merge, and safety.

Offline tests pin deterministic probe text, bounded work, maximum merging and
contextual guards. Live tests check measured cases against the current shelf;
they do not establish universal relevance or repeatable ANN rankings.

The live tests are marked `live` and skip when Qdrant or the embedding API is
unavailable, so the suite stays runnable offline.
"""
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from library_query import expansion, store

MODULE_ROOT = Path(__file__).resolve().parents[1]
BIN = MODULE_ROOT / "library-query.sh"

# The expansion threshold, pinned by these tests.
T = 0.45

# Colloquial queries measured failing before expansion, with the score each
# reached after it. # measurements, so a drop below `T` here means the mapping has stopped working
# against the actual corpus.
MEASURED_COLLOQUIAL = [
    "how i stop to be a loser",
    "i cant stop thinking about her",
    "my brain wont shut up at night",
    "im spiralling again",
    "i overthink everything",
    "i am a complete failure at everything",
    "whats even the point of trying anymore",
    "i freak out when i have to talk to people",
    "i keep snapping at everyone",
    "my moods are all over the place",
    "i worry about everything all the time",
    "i just feel numb and empty",
    "im so burnt out",
    "i feel stuck and cant get anything done",
]

# The measured off-topic population. Expansion must never fire on any of these.
MEASURED_OFFTOPIC = [
    "how do i write a kubernetes operator",
    "carburetor tuning on an old motorcycle",
    # Removed "when did the roman empire fall" (2026-09-21): the library grew
    # from 14 to ~500 volumes, and Robertson's "How to Think Like a Roman
    # Emperor" is now genuinely on-topic for it (0.4907 at p.170). An
    # off-topic guard must guard against DISTRACTION, not against the corpus
    # gaining a relevant book; the question migrated from off-topic to on-topic.
    "how does photosynthesis work",
    "what is the offside rule in football",
    "what is the capital of mongolia",
    "postgres index performance tuning",
    "why is the sky blue",
]

# Removed from the live score check on 2026-11-11 because the library gained a
# database book where this query is now genuinely on-topic (0.5485). It stays
# in the offline cue tests because those check cue behavior on the text, not
# corpus scores.
CORPUS_NOW_ON_TOPIC = {"postgres index performance tuning"}

# Queries that contain a distress *word* in a non-distress sense. Every one of
# these was measured grounding falsely (0.4763-0.5598) against a clinical
# psychology textbook corpus
# against an earlier draft whose cues were bare words. They are regression tests
# for cue scoping, and they matter doubly because technical queries can use
# this same vocabulary.
FALSE_FRIENDS = [
    "how do i handle disk failure in raid",
    "what causes engine failure",
    "heart failure symptoms",
    "handle failure in distributed systems",
    "business failure rates statistics",
    "kernel panic on boot",
    "panic in go recover from panic",
    "how to handle rust panic macro",
    "my code is stuck in an infinite loop",
    "the container is stuck pending",
    "the build is stuck",
    "sql transaction is stuck waiting on a lock",
    "my leg went numb after sitting",
    "costs are spiralling out of control",
]


# --- Mapping: determinism, cue gating, probe budget --------------------------------


def test_cued_query_expands_into_clinical_register():
    """A colloquial cue produces at least one clinical variant."""
    variants = expansion.expansions("how i stop to be a loser")

    assert variants, "a known cue produced no expansion"
    assert "self-worth" in variants[0]


def test_uncued_query_produces_no_variants():
    """No cue means no expansion -- the property the threshold argument rests on."""
    assert expansion.expansions("when did the roman empire fall") == []
    assert expansion.matching_cues("when did the roman empire fall") == []


def test_uncued_query_is_searched_verbatim_as_a_single_probe():
    """An uncued query must embed the exact bytes it would have without expansion."""
    question = "what is the capital of mongolia"

    assert expansion.probes(question) == [question]


def test_original_query_is_always_probe_zero():
    """Expansion is additive: the original is never replaced or discarded."""
    question = "i overthink everything"

    probes = expansion.probes(question)

    assert probes[0] == question
    assert len(probes) > 1, "this query should have expanded"


def test_expansion_is_deterministic():
    """Same input, same ordered output -- no set iteration order leaking through."""
    question = "i cant sleep and i overthink everything and im burnt out"

    assert expansion.probes(question) == expansion.probes(question)
    assert expansion.matching_cues(question) == expansion.matching_cues(question)


def test_probe_count_is_capped():
    """More cues than budget must not mean more probes than budget."""
    question = "i cant sleep and i overthink everything and i feel stuck and im burnt out"

    assert len(expansion.matching_cues(question)) > expansion.MAX_PROBES - 1
    assert len(expansion.probes(question)) == expansion.MAX_PROBES


def test_cue_selection_under_the_cap_prefers_longest_and_is_stable():
    """The tiebreak is deterministic, so the same query always probes the same way."""
    question = "i cant sleep and i overthink everything and i feel stuck and im burnt out"

    selected = expansion.selected_cues(question)

    assert selected == expansion.selected_cues(question)
    assert selected == sorted(selected, key=lambda cue: (-len(cue), cue))


def test_disabling_expansion_yields_the_bare_question():
    """`--no-expand` restores exact prior behavior."""
    question = "how i stop to be a loser"

    assert expansion.probes(question, expand=False) == [question]


def test_expansion_needs_no_network_or_model(monkeypatch):
    """Expansion is pure computation: it must not touch the embedding or store path."""
    def explode(*_args, **_kwargs):
        raise AssertionError("expansion performed I/O")

    monkeypatch.setattr(store, "embed", explode)
    monkeypatch.setattr(store, "qdrant", explode)

    assert expansion.probes("how i stop to be a loser")


def test_duplicate_phrases_collapse_so_the_budget_buys_distinct_probes():
    """Two cues mapping to the same phrase must not spend two probes on it."""
    # Both spellings map to the same clinical phrase.
    question = "i cant sleep because my brain wont shut up"

    variants = expansion.expansions(question)

    assert len(variants) == len(set(variants))


# --- Diagnostics -------------------------------------------------------------------


def test_diagnostics_report_applied_cues_and_probes():
    question = "how i stop to be a loser"

    diag = expansion.diagnostics(question)

    assert diag["applied"] is True
    assert diag["cues"] == ["loser"]
    assert diag["probes"][0] == question


def test_diagnostics_report_not_applied_for_uncued_query():
    diag = expansion.diagnostics("why is the sky blue")

    assert diag["applied"] is False
    assert diag["cues"] == []


def test_diagnostics_report_only_the_cues_that_earned_a_probe():
    """Reporting cues that lost to the budget would misexplain the actual search."""
    question = "i cant sleep and i overthink everything and i feel stuck and im burnt out"

    diag = expansion.diagnostics(question)

    assert len(diag["cues"]) == len(diag["probes"]) - 1


def test_diagnostics_are_empty_when_expansion_is_disabled():
    diag = expansion.diagnostics("how i stop to be a loser", expand=False)

    assert diag["applied"] is False
    assert diag["cues"] == []


# --- Mapping content: the measured pairs -------------------------------------------


@pytest.mark.parametrize("question", MEASURED_COLLOQUIAL)
def test_every_measured_colloquial_query_has_a_cue(question):
    """Each query pinned as fixed must still match a cue.

    If this fails, a mapping was removed or narrowed and a query that used to
    ground has silently stopped grounding.
    """
    assert expansion.matching_cues(question), f"no cue matches measured query: {question}"


@pytest.mark.parametrize("question", MEASURED_OFFTOPIC)
def test_no_cue_fires_on_the_measured_off_topic_population(question):
    """The structural basis for retaining T: off-topic input is never expanded."""
    assert expansion.matching_cues(question) == []
    assert expansion.probes(question) == [question]


# Vocabulary that only the clinical shelf uses. A technical sense of a distress word
# must never pull a probe containing any of it; this is what the scoping exists for.
CLINICAL_VOCABULARY = (
    "rumination", "self-critical", "self-worth", "shame", "anxiety", "anxious",
    "depression", "low mood", "anhedonia", "avoidance", "behavioural activation",
    "burnout", "panic", "inadequacy", "worthlessness", "irritability", "bipolar",
    "insomnia", "procrastination", "emotion regulation", "self-esteem",
)


@pytest.mark.parametrize("question", FALSE_FRIENDS)
def test_distress_words_in_technical_senses_do_not_expand_clinically(question):
    """A bare-word cue would ground "kernel panic" in an ACT textbook. It must not.

    The assertion is on the *register* of what a cue pulls, not on cue count: since
    the dictionary also carries computer-science cues, a technical query legitimately
    matching a technical cue (and probing technical material) is correct. What must
    never happen is a technical sense reaching for the clinical shelf.
    """
    probes = expansion.expansions(question)

    for probe in probes:
        leaked = [term for term in CLINICAL_VOCABULARY if term in probe]
        assert not leaked, (
            f"technical sense pulled clinical material: {question!r} -> {probe!r} ({leaked})"
        )


# --- Widened register coverage: productivity, habits, sleep, computer science ------
#
# One family per test, each asserting the cue fires *and* lands in the right register.
# A cue that fires into the wrong shelf is worse than one that does not fire at all.

PRODUCTIVITY_QUERIES = [
    ("i keep procrastinating on this", "procrastination"),
    ("i keep putting off the hard stuff", "procrastination"),
    ("i cant focus for more than ten minutes", "concentration"),
    ("i waste time all day", "time management"),
    ("i never finish anything i start", "follow-through"),
]

HABIT_QUERIES = [
    ("how do i build a habit that sticks", "habit formation"),
    ("i cant stick to anything", "consistency"),
    ("i have no willpower", "willpower"),
    ("how do i build a routine", "habit formation"),
]

SLEEP_QUERIES = [
    ("i think i have insomnia", "insomnia"),
    ("how do i sleep better", "sleep hygiene"),
    ("ive been having trouble sleeping", "insomnia"),
    ("i always wake up tired", "sleep quality"),
    ("i keep waking up in the night", "sleep maintenance"),
]

CS_QUERIES = [
    ("what is the big o of this loop", "complexity"),
    ("explain time complexity to me", "complexity"),
    ("which data structure should i use", "data structures"),
    ("how do i avoid a race condition", "concurrency"),
    ("what causes a deadlock", "deadlock"),
    ("i have a memory leak somewhere", "memory management"),
    ("how should i refactor this module", "refactoring"),
    ("what design pattern fits here", "design patterns"),
    ("how do distributed systems reach consensus", "distributed systems"),
]


@pytest.mark.parametrize("question,expected_term", PRODUCTIVITY_QUERIES)
def test_productivity_register_expands_into_its_own_vocabulary(question, expected_term):
    variants = expansion.expansions(question)

    assert variants, f"productivity cue did not fire: {question}"
    assert any(expected_term in v for v in variants), (
        f"{question!r} expanded, but not into productivity vocabulary: {variants}"
    )


@pytest.mark.parametrize("question,expected_term", HABIT_QUERIES)
def test_habit_register_expands_into_its_own_vocabulary(question, expected_term):
    variants = expansion.expansions(question)

    assert variants, f"habit cue did not fire: {question}"
    assert any(expected_term in v for v in variants), (
        f"{question!r} expanded, but not into habit vocabulary: {variants}"
    )


@pytest.mark.parametrize("question,expected_term", SLEEP_QUERIES)
def test_sleep_register_expands_into_its_own_vocabulary(question, expected_term):
    variants = expansion.expansions(question)

    assert variants, f"sleep cue did not fire: {question}"
    assert any(expected_term in v for v in variants), (
        f"{question!r} expanded, but not into sleep vocabulary: {variants}"
    )


@pytest.mark.parametrize("question,expected_term", CS_QUERIES)
def test_computer_science_register_expands_technically_not_clinically(question, expected_term):
    """A CS cue must pull the CS shelf, and must not leak clinical vocabulary."""
    variants = expansion.expansions(question)

    assert variants, f"computer-science cue did not fire: {question}"
    assert any(expected_term in v for v in variants), (
        f"{question!r} expanded, but not into technical vocabulary: {variants}"
    )
    for variant in variants:
        assert not [term for term in CLINICAL_VOCABULARY if term in variant], (
            f"a computer-science query reached the clinical shelf: {question!r} -> {variant!r}"
        )


def test_widened_cues_still_leave_the_off_topic_population_untouched():
    """The new families must not disturb the population T was derived against."""
    for question in MEASURED_OFFTOPIC:
        assert expansion.matching_cues(question) == [], (
            f"a widened cue started firing on off-topic input: {question}"
        )


def test_widened_dictionary_remains_deterministic_and_offline(monkeypatch):
    """Task 3.2: still pure computation, still capped, still probe-zero-preserving."""
    def explode(*_args, **_kwargs):
        raise AssertionError("expansion performed I/O")

    monkeypatch.setattr(store, "embed", explode)
    monkeypatch.setattr(store, "qdrant", explode)

    for question, _ in PRODUCTIVITY_QUERIES + HABIT_QUERIES + SLEEP_QUERIES + CS_QUERIES:
        probes = expansion.probes(question)
        assert probes[0] == question, "the raw query stopped being probe zero"
        assert len(probes) <= expansion.MAX_PROBES, "the widened dictionary broke the probe budget"
        assert probes == expansion.probes(question), "expansion became non-deterministic"


def test_widened_registers_still_batch_into_one_embedding_call(monkeypatch):
    """Every probe of a widened-register query rides one embedding request."""
    calls = {"n": 0}

    def embed(texts):
        calls["n"] += 1
        return [[0.0] * 4 for _ in texts]

    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[])
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "embed", embed)

    store.query_places("how do i build a habit that sticks and sleep better", top_k=5)

    assert calls["n"] == 1, "widened expansion added a second embedding round trip"


def test_self_referential_distress_still_fires_for_scoped_cues():
    """Scoping the cues to the speaker must not break the real distress phrasings."""
    for question in ("im a failure", "i'm a failure", "i am such a total failure",
                     "i feel stuck", "i am panicking", "i just feel numb"):
        assert expansion.matching_cues(question), f"scoped cue stopped matching: {question}"


# --- Word boundaries: the 2026-09-23 substring-collision fix ------------------------
#
# Literal cues used to match by bare substring containment, so any longer word that
# happened to contain a cue fired it. Two measured cases reached the live index.


SUBSTRING_COLLISIONS = (
    # (query, the cue that used to fire on it)
    ("We are getting closer to shipping the project", "loser"),
    ("How many inhabitants live in Tokyo?", "habit"),
    ("the closer we get the better", "loser"),
    ("she is a great closer on sales calls", "loser"),
    ("uninhabitable planets are common", "habit"),
    ("the population of inhabitants grew", "habit"),
)


@pytest.mark.parametrize("question,collided_cue", SUBSTRING_COLLISIONS)
def test_substring_collisions_no_longer_fire_a_cue(question, collided_cue):
    """A cue must match a word, not a fragment of one."""
    cues = expansion.matching_cues(question)

    assert collided_cue not in cues, (
        f"{collided_cue!r} fired on {question!r} as a bare substring"
    )
    assert cues == [], f"unexpected cue on an off-topic query: {cues}"
    assert expansion.probes(question) == [question]


def test_word_bounded_cues_still_match_their_real_phrasings():
    """The boundary fix must not cost any genuine match."""
    for question, expected in (
        ("how i stop to be a loser", "loser"),
        ("i feel like such a loser today", "loser"),
        ("i cant stop thinking about her", "cant stop thinking about"),
        ("how do i build a habit that sticks", "habit"),
        ("my habits are a mess", "habit"),
        ("i keep procrastinating on the tax return", "procrastinat"),
        ("chronic procrastination is my problem", "procrastinat"),
        ("i have insomnia", "insomnia"),
        ("what is the time complexity of quicksort", "time complexity"),
        ("explain data structures to me", "data structure"),
    ):
        assert expected in expansion.matching_cues(question), (
            f"word-bounded {expected!r} stopped matching {question!r}"
        )


def test_stem_cues_are_an_explicit_opt_in():
    """Right-open matching is a listed exception, not the default for new cues."""
    for stem in expansion._STEM_CUES:
        assert expansion._literal_cue_pattern(stem).pattern.endswith(
            expansion.re.escape(expansion.normalize(stem))
        ), f"{stem!r} is listed as a stem but was anchored on the right"

    # A whole-word cue is closed at both ends.
    assert expansion._literal_cue_pattern("loser").pattern.endswith(r"\b")


# --- Portuguese cues ----------------------------------------------------------------
#
# The corpus is English-only and the embedding model is weak cross-lingually against
# it, so a Portuguese turn scored ~0.33 where its English twin scored ~0.60. These
# cues map onto the SAME English clinical phrases, so the probe that reaches the shelf
# is already in the shelf's language.


PORTUGUESE_CUES = (
    ("por que eu nao consigo parar de pensar nisso", "rumination"),
    ("estou tendo crises de ansiedade e panico", "anxiety"),
    ("estou tendo crise de pânico de novo", "panic"),
    ("como parar de procrastinar com isso", "procrastination"),
    ("nao consigo dormir a noite", "insomnia"),
    ("não consigo dormir de jeito nenhum", "insomnia"),
    ("me odeio por ter feito isso", "self-criticism"),
    ("tenho insônia há semanas", "insomnia"),
    ("me sinto sozinho demais ultimamente", "loneliness"),
    ("estou sem forças para nada", "burnout"),
)


@pytest.mark.parametrize("question,expected_term", PORTUGUESE_CUES)
def test_portuguese_cue_probes_in_english_clinical_register(question, expected_term):
    probes = expansion.expansions(question)

    assert probes, f"no Portuguese cue fired for {question!r}"
    assert any(expected_term in probe for probe in probes), (
        f"{question!r} -> {probes}: expected {expected_term!r}"
    )


def test_portuguese_probes_are_english_so_they_reach_an_english_shelf():
    """Nothing in a Portuguese cue's probe may be Portuguese -- that is the whole point."""
    portuguese_markers = ("ansiedade", "pânico", "panico", "dormir", "odeio",
                          "sozinho", "forças", "forcas", "insônia", "procrastinar")

    for question, _ in PORTUGUESE_CUES:
        for probe in expansion.expansions(question):
            leaked = [word for word in portuguese_markers if word in probe]
            assert not leaked, f"{question!r} probed in Portuguese: {probe!r} ({leaked})"


def test_portuguese_cues_leave_the_off_topic_population_untouched():
    for question in MEASURED_OFFTOPIC:
        assert expansion.matching_cues(question) == [], (
            f"a Portuguese cue fired on off-topic input: {question!r}"
        )


# --- Merge logic: max score per chunk, ranking, monotonicity ------------------------


def _hit(point_id, score):
    hit = MagicMock()
    hit.id = point_id
    hit.score = score
    hit.payload = {"path": f"doc-{point_id}.pdf", "doc_name": f"doc-{point_id}.pdf",
                   "page": 1, "text": "passage", "scope": "inbox"}
    return hit


def _store_returning(monkeypatch, per_probe_hits):
    """Wire the store so each probe returns its own canned hit list."""
    calls = {"n": 0}
    client = MagicMock()

    def query_points(**_kwargs):
        result = MagicMock()
        result.points = per_probe_hits[calls["n"]]
        calls["n"] += 1
        return result

    client.query_points.side_effect = query_points
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "embed", lambda texts: [[0.0] * 4 for _ in texts])
    return calls


def test_duplicate_chunk_keeps_its_best_score(monkeypatch):
    """The same chunk found by two probes appears once, at the higher score."""
    _store_returning(monkeypatch, [[_hit(1, 0.41)], [_hit(1, 0.62)]])

    places = store.query_places("how i stop to be a loser", top_k=5)

    assert len(places) == 1
    assert places[0]["score"] == 0.62


def test_merged_results_are_ranked_and_truncated(monkeypatch):
    """More distinct chunks than top_k: sort descending, cut to top_k."""
    _store_returning(monkeypatch, [
        [_hit(1, 0.30), _hit(2, 0.50)],
        [_hit(3, 0.70), _hit(4, 0.40)],
    ])

    places = store.query_places("how i stop to be a loser", top_k=3)

    assert [p["score"] for p in places] == [0.70, 0.50, 0.40]


def test_merge_keeps_observed_scores_before_truncation(monkeypatch):
    """Observed candidates retain their best score when the cut has room."""
    unexpanded = [_hit(1, 0.44), _hit(2, 0.30)]
    _store_returning(monkeypatch, [list(unexpanded), [_hit(1, 0.58), _hit(3, 0.51)]])

    expanded = store.query_places("how i stop to be a loser", top_k=5)

    by_id = {p["doc"]: p["score"] for p in expanded}
    assert by_id["doc-1.pdf"] >= 0.44
    assert by_id["doc-2.pdf"] >= 0.30, "an unexpanded result was dropped by the merge"


def test_uncued_query_issues_exactly_one_probe(monkeypatch):
    """No cue, no extra work: one embed input, one search."""
    embedded = {}

    def embed(texts):
        embedded["texts"] = texts
        return [[0.0] * 4 for _ in texts]

    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[_hit(1, 0.23)])
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "embed", embed)

    store.query_places("when did the roman empire fall", top_k=5)

    assert embedded["texts"] == ["when did the roman empire fall"]
    assert client.query_points.call_count == 1


def test_cued_query_issues_one_probe_per_variant(monkeypatch):
    """A cued query searches once per probe, original included."""
    embedded = {}

    def embed(texts):
        embedded["texts"] = texts
        return [[0.0] * 4 for _ in texts]

    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[_hit(1, 0.50)])
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "embed", embed)

    store.query_places("how i stop to be a loser", top_k=5)

    assert embedded["texts"][0] == "how i stop to be a loser"
    assert len(embedded["texts"]) > 1
    assert client.query_points.call_count == len(embedded["texts"])


def test_no_expand_searches_the_question_verbatim(monkeypatch):
    """`expand=False` must reproduce pre-change behavior exactly."""
    embedded = {}

    def embed(texts):
        embedded["texts"] = texts
        return [[0.0] * 4 for _ in texts]

    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[_hit(1, 0.41)])
    monkeypatch.setattr(store, "qdrant", lambda: client)
    monkeypatch.setattr(store, "embed", embed)

    store.query_places("how i stop to be a loser", top_k=5, expand=False)

    assert embedded["texts"] == ["how i stop to be a loser"]
    assert client.query_points.call_count == 1


def test_hit_without_a_score_does_not_crash_the_merge(monkeypatch):
    """A scoreless hit sorts last rather than raising on comparison."""
    scoreless = _hit(9, None)
    _store_returning(monkeypatch, [[_hit(1, 0.50), scoreless]])

    places = store.query_places("when did the roman empire fall", top_k=5)

    assert places[0]["score"] == 0.50
    assert places[-1]["score"] is None


# --- The `--min-score` floor on the merged ranking ---------------------------------
#
# The floor judges merged scores only, after per-chunk best-score merging and before
# the top-k cut. Both halves of that sentence are load-bearing and tested below.


def test_floor_below_every_score_keeps_everything(monkeypatch):
    """A floor nothing fails changes nothing about the result."""
    _store_returning(monkeypatch, [[_hit(1, 0.62), _hit(2, 0.51)]])

    places = store.query_places("when did the roman empire fall", top_k=5, min_score=0.10)

    assert [p["score"] for p in places] == [0.62, 0.51]


def test_floor_above_every_score_returns_empty_places(monkeypatch):
    """Nothing clearing the bar is a valid empty answer, not an error."""
    _store_returning(monkeypatch, [[_hit(1, 0.44), _hit(2, 0.31)]])

    places = store.query_places("when did the roman empire fall", top_k=5, min_score=0.90)

    assert places == []


def test_floor_mid_distribution_keeps_only_the_survivors(monkeypatch):
    """A floor inside the distribution cuts exactly the results beneath it."""
    _store_returning(monkeypatch, [
        [_hit(1, 0.62), _hit(2, 0.48), _hit(3, 0.4799), _hit(4, 0.20)],
    ])

    places = store.query_places("when did the roman empire fall", top_k=5, min_score=0.48)

    assert [p["score"] for p in places] == [0.62, 0.48], "the floor is inclusive at the bound"


def test_floor_applies_before_the_top_k_cut(monkeypatch):
    """top-k of the survivors, not survivors of top-k.

    Filtering after the cut would let results below the floor consume slots and
    return fewer passing places than the caller asked for and the index holds.
    """
    _store_returning(monkeypatch, [
        [_hit(1, 0.90), _hit(2, 0.20), _hit(3, 0.80), _hit(4, 0.10), _hit(5, 0.70)],
    ])

    places = store.query_places("when did the roman empire fall", top_k=3, min_score=0.50)

    assert [p["score"] for p in places] == [0.90, 0.80, 0.70]


def test_floor_composes_with_top_k_to_cap_survivors(monkeypatch):
    """When more results survive the floor than top-k, top-k still caps the output."""
    _store_returning(monkeypatch, [
        [_hit(1, 0.90), _hit(2, 0.80), _hit(3, 0.70), _hit(4, 0.60)],
    ])

    places = store.query_places("when did the roman empire fall", top_k=2, min_score=0.50)

    assert [p["score"] for p in places] == [0.90, 0.80]


def test_result_under_the_floor_on_probe_zero_survives_on_an_expanded_probe(monkeypatch):
    """No per-probe filtering: the merged score is what faces the floor.

    This is the register-gap recovery the expansion exists for. Filtering each probe
    before the merge would discard chunk 1 on its raw 0.41 and silently undo the very
    behavior `add-colloquial-query-expansion` was built to produce.
    """
    _store_returning(monkeypatch, [[_hit(1, 0.41)], [_hit(1, 0.62)]])

    places = store.query_places("how i stop to be a loser", top_k=5, min_score=0.48)

    assert [p["score"] for p in places] == [0.62]


def test_no_floor_is_the_unfiltered_default(monkeypatch):
    """`min_score=None` must behave exactly as before the flag existed."""
    _store_returning(monkeypatch, [[_hit(1, 0.44), _hit(2, 0.02)]])

    places = store.query_places("when did the roman empire fall", top_k=5)

    assert [p["score"] for p in places] == [0.44, 0.02]


def test_scoreless_hit_does_not_survive_a_floor(monkeypatch):
    """A hit with no score sorts last and cannot clear any floor."""
    _store_returning(monkeypatch, [[_hit(1, 0.62), _hit(9, None)]])

    places = store.query_places("when did the roman empire fall", top_k=5, min_score=0.48)

    assert [p["score"] for p in places] == [0.62]


def test_cli_exposes_min_score_as_an_optional_float():
    """The flag is declared on the query parser, defaults to unset, and is a float."""
    from library_query import cli

    parser = cli.build_parser()
    args = parser.parse_args(["query", "habits"])
    assert args.min_score is None

    args = parser.parse_args(["query", "habits", "--min-score", "0.48"])
    assert args.min_score == 0.48


def test_cli_passes_the_floor_through_to_the_store(monkeypatch):
    """`cmd_query` must forward `--min-score`, not silently drop it."""
    from library_query import cli

    captured = {}

    def fake_query_places(question, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli.store, "query_places", fake_query_places)
    parser = cli.build_parser()
    args = parser.parse_args(["query", "habits", "--json", "--min-score", "0.48"])
    args.parser = parser

    assert cli.cmd_query(args) == 0
    assert captured["min_score"] == 0.48


# --- Live index validation ---------------------------------------------------------


def _live_available() -> bool:
    if not BIN.exists() or not os.environ.get("OPENROUTER_API_KEY"):
        # The CLI loads its own .env; treat a present .env as enough to try.
        if not (MODULE_ROOT / ".env").exists():
            return False
    try:
        import httpx

        httpx.get("http://127.0.0.1:6333/collections", timeout=2)
        return True
    except Exception:
        return False


live = pytest.mark.skipif(not _live_available(), reason="live Qdrant/embedding API unavailable")


def _query_live(question: str, expand: bool = True) -> dict:
    cmd = [str(BIN), "query", question, "--json", "--top-k", "3"]
    if not expand:
        cmd.append("--no-expand")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=MODULE_ROOT, timeout=120)
    assert out.returncode == 0, f"query failed: {out.stderr[:300]}"
    return json.loads(out.stdout)


@pytest.mark.live
@live
def test_the_measured_failing_case_now_grounds():
    """The query this whole change exists for: 0.4102 before, above T after."""
    payload = _query_live("how i stop to be a loser")

    assert payload["expansion"]["applied"] is True
    assert payload["places"], "no results returned"
    assert payload["places"][0]["score"] >= T, (
        f"the measured failing case still does not ground: {payload['places'][0]['score']}"
    )


@pytest.mark.live
@live
@pytest.mark.parametrize("question", MEASURED_COLLOQUIAL)
def test_measured_colloquial_population_grounds(question):
    """Every colloquial query pinned as fixed must clear T live."""
    payload = _query_live(question)

    top = payload["places"][0]["score"]
    assert top >= T, f"{question!r} scored {top}, below T={T}"


@pytest.mark.live
@live
@pytest.mark.parametrize("question", MEASURED_OFFTOPIC)
def test_uncued_live_input_and_common_candidate_scores(question, monkeypatch):
    """Hold the vector fixed to isolate expansion from provider variation.

    Compare matching passage identities rather than list positions: independent
    approximate searches can return different tails. The numerical bound for
    shared candidates stays unchanged. Offline tests pin embedding input identity.
    """
    assert expansion.probes(question) == expansion.probes(question, expand=False) == [question]
    vector = store.embed([question])[0]
    observed_inputs = []
    def frozen_embed(texts):
        observed_inputs.extend(texts)
        assert texts == [question]
        return [vector]
    monkeypatch.setattr(store, "embed", frozen_embed)
    with_expansion = store.query_places(question, top_k=3, expand=True)
    without = store.query_places(question, top_k=3, expand=False)
    assert observed_inputs == [question, question]
    def by_identity(places):
        return {(p["doc"], p["page"], p["text"]): p["score"] for p in places}
    expanded, plain = by_identity(with_expansion), by_identity(without)
    shared = expanded.keys() & plain.keys()
    assert shared, f"No shared candidates across repeated searches: {expanded.keys()}, {plain.keys()}"
    for identity in shared:
        assert expanded[identity] == pytest.approx(plain[identity], abs=1e-3), identity


@pytest.mark.live
@live
@pytest.mark.parametrize(
    "question", [q for q in MEASURED_OFFTOPIC if q not in CORPUS_NOW_ON_TOPIC]
)
def test_off_topic_population_stays_below_threshold(question):
    """Expansion must not lift genuinely off-topic input over T."""
    payload = _query_live(question)

    for place in payload["places"]:
        assert place["score"] < T, f"{question!r} grounded at {place['score']}"


@pytest.mark.live
@live
def test_expansion_does_not_change_the_result_contract():
    """`places[]` keeps exactly the documented fields, in order."""
    from library_query.cli import QUERY_OUTPUT_FIELDS

    payload = _query_live("how i stop to be a loser")

    for place in payload["places"]:
        assert list(place.keys()) == list(QUERY_OUTPUT_FIELDS)


@pytest.mark.live
@live
def test_the_deployed_hook_invocation_shape_still_works():
    """The hooks call `query <text> --json` and read four fields. Unchanged."""
    payload = _query_live("i cant stop thinking about her")

    usable = [p for p in payload["places"] if p["score"] >= T]

    assert usable, "the hook would fall back silently on a covered topic"
    for place in usable:
        assert place["doc_name"] and place["text"]
        assert isinstance(place["score"], float)
        assert "page" in place


def test_new_candidates_can_displace_original_at_top_k(monkeypatch):
    _store_returning(monkeypatch, [[_hit(1, 0.40)], [_hit(2, 0.60)]])
    places = store.query_places("how i stop to be a loser", top_k=1)
    assert [p["doc"] for p in places] == ["doc-2.pdf"]
