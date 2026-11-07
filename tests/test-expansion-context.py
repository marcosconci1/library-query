"""Contextual false-friend regressions and bounded top-k semantics."""
import pytest
from library_query import expansion

@pytest.mark.parametrize("question", [
    "What does a loser bracket mean in a tournament?",
    "I am implementing a loser tree sort",
    "I am not a loser", "I am not really such a complete loser", "I don't feel like a loser",
    'What does the word "loser" mean?',
    'Translate the phrase "I am a loser"',
    "What does the word 'insomnia' mean?",
    "Why does the Insomnia app fail to send requests?",
    "How do I send a REST request in Insomnia?",
    "I don't have insomnia", "I never overthink things",
    "I am not anxious", "I don't panic on stage",
])
def test_contextual_negatives_have_no_clinical_probes(question):
    assert expansion.expansions(question) == []
    assert expansion.probes(question) == [question]

@pytest.mark.parametrize("question", [
    "how i stop to be a loser", "i feel like such a loser today",
    'I feel like "a loser" today', "i have insomnia",
    "i am anxious", "i can't sleep", "i am not good enough",
    "I don't panic, but I have insomnia",
    "I feel like a loser after the tournament",
    "I request advice on insomnia",
])
def test_legitimate_distress_keeps_recall(question):
    assert expansion.expansions(question)

def test_transform_version_invalidates_old_probe_cache():
    assert expansion.TRANSFORM_VERSION == 2


def test_quoted_technical_terms_keep_technical_probes():
    assert expansion.expansions('What does the term "deadlock" mean?') == [
        "concurrency deadlock lock ordering synchronization"
    ]
