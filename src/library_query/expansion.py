"""Colloquial -> clinical query expansion, applied before embedding.

The corpus is written in clinical/textbook register; users write in colloquial
distress ("how i stop to be a loser"). Those do not embed near each other, so
genuinely on-topic questions fall under the grounding threshold while the book
that answers them sits indexed and unread.

This module translates register using a curated mapping and conservative local
eligibility guards. Coverage changes need cue/context regressions and measured
shelf checks; vocabulary matches alone cannot establish intent or relevance.

Two properties matter more than coverage, and both are structural:

- **Cue-gated.** No matching cue means no expansion at all -- the query is
  embedded byte-for-byte as it is today. This keeps embedding
  input unchanged. Independent live embeddings/searches may still vary;
  similarity floors do not establish semantic relevance.
- **Never concatenated.** Variants are separate probes, embedded independently
  (see `store.query_places`). Appending clinical terms to the user's text both
  dilutes real matches and drags unrelated queries toward the corpus -- measured
  at 0.4484 for "when did the roman empire fall", a hair under the threshold.

No language model or network is used to compute variants. Callers can disable
expansion for context-resolved questions; retrieval still embeds and searches.
"""
from __future__ import annotations

import re

# The original query plus at most three variants. Each probe is an embedding
# call and a search, so the worst case stays bounded for an interactive path.
MAX_PROBES = 4

#: Input-transform version: part of the embedding cache key. Bump when the way
#: probe text is produced from a question changes, so vectors produced under an
#: older transform are never reused as if equivalent.
TRANSFORM_VERSION = 2

# Cue (word-bounded phrase in the normalized query) -> clinical register phrase to
# probe with.
#
# Phrases are tuned against the live index, not guessed: each was measured to pull
# its colloquial counterpart above T = 0.45 and onto the correct shelf.
#
# Literal cues match on WORD BOUNDARIES, not bare substrings (see
# `_literal_cue_pattern`): "closer" must not fire "loser". Cues that are meant as
# truncated stems are listed in `_STEM_CUES` and stay open on the right.
COLLOQUIAL_TO_CLINICAL: dict[str, str] = {
    # Self-worth / self-criticism. "loser" embeds near money-failure language, so
    # the raw query ranks the wealth shelf above the psychology shelf; this pulls
    # it back onto the workbook.
    "loser": "self-critical thinking negative self-judgment shame inadequacy self-worth",
    "worthless": "feelings of worthlessness low self-esteem negative self-evaluation",
    # Self-referential only. A bare "failure" cue was measured firing on "disk
    # failure in raid" and "engine failure", grounding them at 0.4983 on a
    # workbook -- precisely the false grounding T exists to prevent. The cue has
    # to carry the speaker, so it is matched by regex allowing the modifiers
    # people actually use ("i am such a complete failure").
    r"\b(?:i'm|im|i am)\b.{0,24}?\bfailure\b":
        "self-critical thinking negative self-judgment shame inadequacy self-worth",
    "not good enough": "feelings of inadequacy low self-worth negative self-evaluation",
    "hate myself": "self-criticism self-directed hostility shame low self-worth",
    # Rumination. The corpus's strongest rumination coverage and the widest
    # measured gap: "i cant stop thinking about her" moves 0.2925 -> 0.6020.
    "cant stop thinking about": "rumination about a person intrusive repetitive thoughts",
    "can't stop thinking about": "rumination about a person intrusive repetitive thoughts",
    "overthink": "overthinking rumination repetitive negative thinking",
    # Costs spiral and loops dwell; bind these to the speaker too.
    r"\b(?:i'm|im|i am|i keep|feel)\b.{0,16}?\bspiral(?:l)?ing\b":
        "rumination spiral repetitive negative thinking escalation",
    r"\b(?:i'm|im|i am|i keep|stop)\b.{0,16}?\bdwelling\b":
        "rumination dwelling on the past repetitive negative thinking",
    "going round in circles": "rumination repetitive negative thinking perseveration",
    # Three spellings of one thought: contracted, apostrophised, and expanded.
    # Normalization folds punctuation but not "won't" -> "will not".
    "brain wont shut up": "rumination at night sleep onset insomnia worry in bed",
    "brain won't shut up": "rumination at night sleep onset insomnia worry in bed",
    "brain will not shut up": "rumination at night sleep onset insomnia worry in bed",
    "cant sleep": "rumination at night sleep onset insomnia worry in bed",
    "can't sleep": "rumination at night sleep onset insomnia worry in bed",
    "up all night": "rumination at night sleep onset insomnia worry in bed",
    # Worry / anxiety.
    "worry about everything": "generalized worry anxiety repetitive negative thinking",
    "worried about everything": "generalized worry anxiety repetitive negative thinking",
    "freak out": "social anxiety avoidance exposure anxious arousal",
    # Speaker-bound, like the `failure` cue and for the same measured reason:
    # bare "panic" fires on "kernel panic on boot" and Go's panic/recover, which
    # scored 0.5321 against a textbook in the measured corpus. Technical
    # callers query this same tool with exactly that vocabulary, so these cannot be loose.
    r"\b(?:i'm|im|i am|i)\b.{0,16}?\bpanic": "panic anxiety anxious arousal avoidance exposure",
    "panic attack": "panic anxiety anxious arousal avoidance exposure",
    r"\b(?:i'm|im|i am|feel|feeling)\b.{0,16}?\banxious\b": "anxiety avoidance exposure anxious arousal",
    # Mood.
    "point of trying": "hopelessness low mood depression loss of motivation",
    # "my leg went numb" is not anhedonia; bind the cue to feeling, not sensation.
    r"\b(?:i'm|im|i am|i feel|feeling|i just feel)\b.{0,16}?\bnumb\b":
        "emotional numbness anhedonia low mood depression",
    "moods are all over": "mood instability bipolar mood episodes mood charting",
    "mood swings": "mood instability bipolar mood episodes mood charting",
    # Activation / avoidance.
    # Code, containers and builds all get "stuck"; only a person gets stuck in
    # the sense the corpus means. Measured: "the build is stuck" scored 0.5598
    # against the rumination material before the cue was bound to the speaker.
    r"\b(?:i'm|im|i am|i feel|feeling)\b.{0,16}?\bstuck\b":
        "avoidance behavioural activation procrastination motivation",
    "cant get anything done": "avoidance behavioural activation procrastination motivation",
    "burnt out": "burnout exhaustion depletion recovery",
    "burned out": "burnout exhaustion depletion recovery",
    # Irritability.
    "snapping at": "irritability anger emotion regulation",
    # --- Productivity / procrastination ------------------------------------------
    # Colloquial work-avoidance language. Kept off bare "focus"/"distracted", which
    # a technical query uses freely ("focus the input", "distracted by the logs").
    "procrastinat": "procrastination task avoidance behavioural activation motivation",
    "keep putting off": "procrastination task avoidance delay aversive task initiation",
    "put it off": "procrastination task avoidance delay aversive task initiation",
    "cant focus": "attention focus concentration distraction deep work",
    "can't focus": "attention focus concentration distraction deep work",
    "cant concentrate": "attention focus concentration distraction deep work",
    "can't concentrate": "attention focus concentration distraction deep work",
    "waste time": "time management attention distraction deep work priorities",
    "wasting time": "time management attention distraction deep work priorities",
    "too much to do": "workload overwhelm prioritisation time management",
    "never finish anything": "task completion follow-through procrastination motivation",
    # --- Habits ------------------------------------------------------------------
    # "habit" is near-unambiguous in this corpus; the CS register says "convention"
    # or "pattern" instead, so no speaker binding is needed here.
    "habit": "habit formation cue routine reward behaviour change repetition",
    "stick to": "habit formation consistency adherence behaviour change",
    "give up on": "motivation persistence relapse behaviour change",
    "willpower": "self-control willpower ego depletion behaviour change",
    "self discipline": "self-control discipline habit formation behaviour change",
    "build a routine": "habit formation routine cue reward consistency",
    # --- Sleep -------------------------------------------------------------------
    # Distinct from the existing night-rumination cues: these are sleep-hygiene
    # queries rather than worry-in-bed ones, and the corpus separates them.
    "insomnia": "insomnia sleep onset sleep maintenance sleep hygiene",
    "sleep better": "sleep hygiene sleep quality circadian rhythm sleep restriction",
    "trouble sleeping": "insomnia sleep onset sleep hygiene sleep quality",
    "wake up tired": "sleep quality non-restorative sleep circadian rhythm fatigue",
    "keep waking up": "sleep maintenance insomnia nocturnal awakening sleep quality",
    "oversleep": "hypersomnia sleep inertia circadian rhythm sleep schedule",
    # --- Computer science ---------------------------------------------------------
    # These deliberately expand *into* the technical register, never the clinical
    # one. They exist so a CS question reaches the CS shelf, and they are the
    # counterweight to the false-friend scoping above: "my code is stuck" must pull
    # debugging material, not self-help material.
    "big o": "algorithmic complexity asymptotic analysis running time big-O notation",
    "time complexity": "algorithmic complexity asymptotic analysis running time",
    "data structure": "data structures arrays trees hash tables algorithm design",
    "algorithm": "algorithm design analysis correctness complexity",
    "race condition": "concurrency race condition synchronization mutual exclusion",
    "deadlock": "concurrency deadlock lock ordering synchronization",
    "memory leak": "memory management allocation garbage collection resource leak",
    "refactor": "refactoring code structure design maintainability technical debt",
    "design pattern": "software design patterns abstraction modularity architecture",
    "type system": "type systems static typing type inference program correctness",
    "distributed system": "distributed systems consistency fault tolerance consensus replication",
    "database index": "database indexing query planning b-tree storage engine",
    # --- Portuguese ---------------------------------------------------------------
    # The corpus is entirely English and `text-embedding-3-small` is weak
    # cross-lingually against it: "por que eu nao consigo parar de pensar nisso"
    # scored 0.3320 where its English twin scored 0.6020. Nothing here translates the
    # *query* -- these cues map straight onto the SAME English clinical phrases the
    # English cues use, so the probe that reaches the shelf is already in the shelf's
    # language. That is why one row per concept is enough and no PT corpus is needed.
    #
    # Both spellings of every accented word are listed: users type with and without
    # accents, and normalization folds punctuation but not diacritics.
    "ansiedade": "anxiety avoidance exposure anxious arousal",
    "crise de panico": "panic anxiety anxious arousal avoidance exposure",
    "crise de pânico": "panic anxiety anxious arousal avoidance exposure",
    "nao consigo dormir": "insomnia sleep onset sleep maintenance sleep hygiene",
    "não consigo dormir": "insomnia sleep onset sleep maintenance sleep hygiene",
    "insonia": "insomnia sleep onset sleep maintenance sleep hygiene",
    "insônia": "insomnia sleep onset sleep maintenance sleep hygiene",
    "procrastin": "procrastination task avoidance behavioural activation motivation",
    "parar de pensar": "rumination about a person intrusive repetitive thoughts",
    "me odeio": "self-criticism self-directed hostility shame low self-worth",
    "sozinho demais": "loneliness social isolation belonging connection",
    "sem forcas": "burnout exhaustion depletion recovery",
    "sem forças": "burnout exhaustion depletion recovery",
}

_NON_WORD = re.compile(r"[^\w\s']+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and strip punctuation so cues match real, messily-typed input."""
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def _is_regex(cue: str) -> bool:
    """Whether a cue is a regex rather than a literal phrase.

    Regex cues exist for the handful of mappings that must bind to the speaker --
    "failure" alone fires on disk and engine failure -- and are marked by regex
    metacharacters no literal cue in this mapping contains.
    """
    return any(token in cue for token in (r"\b", "(?:", "{", "|"))


#: Literal cues that are deliberately truncated stems rather than whole words, and so
#: must stay open at the right-hand edge: "procrastinat" has to reach both
#: "procrastinating" and "procrastination"; "habit" has to reach "habits"; "algorithm"
#: has to reach "algorithms". These get a leading `\b` only.
#:
#: Everything not listed here is anchored at BOTH ends. Keeping the exceptions explicit
#: is the point: a new cue is whole-word by default, so nobody has to remember that
#: substring matching is dangerous -- it simply is not available unless asked for.
_STEM_CUES: frozenset[str] = frozenset({
    "overthink",        # overthinking, overthinks
    "procrastinat",     # procrastinating, procrastination
    "habit",            # habits
    "insomnia",         # insomniac
    "oversleep",        # oversleeping, overslept (prefix only)
    "algorithm",        # algorithms, algorithmic
    "refactor",         # refactoring, refactored
    "data structure",   # data structures
    "design pattern",   # design patterns
    "type system",      # type systems
    "distributed system",  # distributed systems
    "database index",   # database indexes, database indexing
    "race condition",   # race conditions
    "memory leak",      # memory leaks
    "deadlock",         # deadlocks
    "ansiedade",        # ansiedades (PT)
    "procrastin",       # procrastinar, procrastinando, procrastinação (PT)
})

#: Compiled word-boundary patterns for literal cues, built once on first use.
_LITERAL_CUE_RE: dict[str, re.Pattern[str]] = {}


def _literal_cue_pattern(cue: str) -> re.Pattern[str]:
    """A word-boundary matcher for one literal cue.

    Bare substring containment was the original implementation and it was wrong in a
    way that fired constantly: "closer" contains "loser", so a project-shipping update
    pulled shame and self-condemnation material, and "inhabitants" contains "habit", so
    a question about Tokyo's population pulled habit-formation loops. Anchoring to `\\b`
    removes that whole class of collision while still matching every phrasing that
    genuinely shares the cue's wording.

    The trailing boundary is dropped for the stems in `_STEM_CUES`, which are truncated
    on purpose and have to stay open on the right.
    """
    pattern = _LITERAL_CUE_RE.get(cue)
    if pattern is None:
        suffix = "" if cue in _STEM_CUES else r"\b"
        pattern = re.compile(rf"\b{re.escape(normalize(cue))}{suffix}")
        _LITERAL_CUE_RE[cue] = pattern
    return pattern


# Known technical uses of clinical vocabulary. This is a conservative lexical
# guard, not an intent classifier; callers can always disable expansion.
_TECHNICAL_SLEEP = re.compile(
    r"\b(?:api|http|rest|endpoint)\b|"
    r"\binsomnia\s+(?:desktop\s+)?(?:client|app|workspace|plugin)\b|"
    r"\bsend\s+(?:a\s+|an\s+|my\s+)?requests?\b"
)
_SELF_REFERENCE = re.compile(r"\b(?:i|i'm|im|me|my|myself)\b")
_NEGATION = re.compile(r"\b(?:not|never|don't|dont|doesn't|doesnt|isn't|isnt|no longer)\b")
_MENTION = re.compile(r"\b(?:word|term|phrase|quote|quoted|means|meaning|definition|translate)\b")
_QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”|`[^`\n]*`|(?<!\w)'[^'\n]*'(?!\w)")
_TECHNICAL_CUES = frozenset({
    "big o", "time complexity", "data structure", "algorithm", "race condition",
    "deadlock", "memory leak", "refactor", "design pattern", "type system",
    "distributed system", "database index",
})


def _cue_matches(cue: str, normalized: str) -> bool:
    """Match eligible cue occurrences, excluding explicit false-friend contexts."""
    pattern = re.compile(cue) if _is_regex(cue) else _literal_cue_pattern(cue)
    for match in pattern.finditer(normalized):
        prefix = normalized[max(0, match.start() - 64):match.start()]
        if cue not in _TECHNICAL_CUES:
            # Limit negation to the immediately preceding phrase. The cue's own
            # negation ("not good enough", "can't sleep") remains meaningful.
            local = prefix.split(" but ")[-1].split(" however ")[-1]
            if _NEGATION.search(" ".join(local.split()[-4:])):
                continue
            if _is_regex(cue) and _NEGATION.search(match.group()):
                continue
        if cue == "loser":
            references = list(_SELF_REFERENCE.finditer(prefix))
            if not references or _NEGATION.search(prefix[references[-1].end():]):
                continue
            if re.search(r"\bloser\s+(?:bracket|tree)\b", normalized):
                continue
        if cue in {"insomnia", "insonia", "insônia"} and _TECHNICAL_SLEEP.search(normalized):
            continue
        return True
    return False


def matching_cues(question: str) -> list[str]:
    """Return the cues present in `question`, longest first.

    Longest-first is the deterministic tiebreak the probe budget relies on: a
    longer cue is the more specific reading, so it is the one worth spending a
    probe on when several match.
    """
    # Quotation alone can describe a real feeling. Mask quoted text only in
    # metalinguistic questions, preserving self-referential quoted distress.
    cue_text = _QUOTED.sub(" ", question) if _MENTION.search(normalize(question)) else question
    normalized = normalize(cue_text)
    original = normalize(question)
    found = [cue for cue in COLLOQUIAL_TO_CLINICAL
             if _cue_matches(cue, original if cue in _TECHNICAL_CUES else normalized)]
    return sorted(found, key=lambda cue: (-len(cue), cue))


def selected_cues(question: str) -> list[str]:
    """The cues that actually earn a probe, after the budget is applied.

    Distinct from `matching_cues`: a query can match more cues than there are
    probes to spend, and the diagnostic block should report the ones that were
    really used rather than every one that happened to match.
    """
    chosen: list[str] = []
    phrases: list[str] = []
    for cue in matching_cues(question):
        phrase = COLLOQUIAL_TO_CLINICAL[cue]
        if phrase in phrases:
            continue
        phrases.append(phrase)
        chosen.append(cue)
        if len(phrases) >= MAX_PROBES - 1:
            break
    return chosen


def expansions(question: str) -> list[str]:
    """Clinical-register variants for `question`, capped by the probe budget.

    Empty when no cue matches -- the caller then searches the query verbatim,
    preserving uncued embedding input. Duplicate phrases (several cues sharing
    a mapping) collapse, so the budget is spent on distinct probes.
    """
    return [COLLOQUIAL_TO_CLINICAL[cue] for cue in selected_cues(question)]


def probes(question: str, expand: bool = True) -> list[str]:
    """The full probe set: the original query first, then any variants.

    The original is always probe 0 and is never replaced. Maximum merging
    preserves observed candidate scores before truncation; final top-k can
    displace original-probe candidates.
    """
    if not expand:
        return [question]
    return [question, *expansions(question)]


def diagnostics(question: str, expand: bool = True) -> dict:
    """The `expansion` block reported in `query --json`.

    Expansion is on by default, so it has to be visible: this is how a surprising
    result can be traced back to the cue that caused it.
    """
    cues = selected_cues(question) if expand else []
    probe_set = probes(question, expand=expand)
    return {
        "applied": len(probe_set) > 1,
        "cues": cues,
        "probes": probe_set,
    }
