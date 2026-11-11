"""Selected, cut-off passages are shown whole when the output caps allow it."""
import importlib.util
import json
import time
from pathlib import Path

import pytest

PLUGIN = Path(__file__).parents[1] / 'integrations/hermes/library-query/__init__.py'
spec = importlib.util.spec_from_file_location('library_filter_expand_tests', PLUGIN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def candidate(i, cut=1600, truncated=True):
    return ({'source_id': f'doc{i}', 'title': f'Title {i}', 'authors': ['Author']},
            {'evidence_id': f'e{i}', 'chunk_index': i, 'text': str(i) * cut, 'text_truncated': truncated,
             'location': {'kind': 'page', 'value': i + 1}})


@pytest.fixture(autouse=True)
def all_selected(monkeypatch):
    monkeypatch.setattr(m, 'judge', lambda scope, question, values: ('model', [True] * len(values)))


def run(values, expand):
    data = json.loads(m.bundle('scope', 'q', values, time.monotonic(), expand=expand))
    return data, [p for s in data['sources'] for p in s['passages']]


def test_selected_cut_passage_is_replaced_by_full_chunk():
    seen = []
    def expand(p):
        seen.append(p['evidence_id'])
        return p['text'] + 'END'
    data, passages = run([candidate(0)], expand)
    assert seen == ['e0']
    assert passages[0]['text'].endswith('END')
    assert passages[0]['text_truncated'] is False
    assert data['receipt']['expanded_count'] == 1


def test_complete_passages_are_never_looked_up():
    data, passages = run([candidate(0, truncated=False)], lambda p: pytest.fail('must not expand'))
    assert passages[0]['text_truncated'] is False
    assert 'expanded_count' not in data['receipt']


def test_expansion_never_drops_a_selected_passage():
    data, passages = run([candidate(0), candidate(1), candidate(2)], lambda p: p['text'] + 'x' * 2000)
    assert len(passages) == 3
    assert sum(len(p['text']) for p in passages) <= m.MAX_TEXT + m.EXPAND_HEADROOM
    assert passages[0]['text_truncated'] is False
    assert any(p['text_truncated'] for p in passages)  # later ones stay cut once the headroom is used


def test_modest_expansion_fits_all_three():
    data, passages = run([candidate(0), candidate(1), candidate(2)], lambda p: p['text'] + 'x' * 400)
    assert len(passages) == 3
    assert all(not p['text_truncated'] for p in passages)
    assert data['receipt']['expanded_count'] == 3


def test_judge_budget_in_receipt_is_unchanged():
    data, _ = run([candidate(0)], lambda p: p['text'] + 'x')
    assert m.FULL_BUDGET['max_text'] == 4800 and m.FULL_BUDGET['max_excerpt_chars'] == 1600


def test_failed_lookup_leaves_the_passage_as_judged():
    data, passages = run([candidate(0)], lambda p: None)
    assert passages[0]['text'] == '0' * 1600
    assert passages[0]['text_truncated'] is True
    assert 'expanded_count' not in data['receipt']


def test_shorter_or_equal_text_is_ignored():
    data, passages = run([candidate(0)], lambda p: 'short')
    assert passages[0]['text'] == '0' * 1600


def test_response_cap_falls_back_to_the_cut_excerpt():
    big = ({'source_id': 'doc0', 'title': 'T' * 5200, 'authors': ['A']},
           {'evidence_id': 'e0', 'chunk_index': 0, 'text': 'a' * 1600, 'text_truncated': True,
            'location': {'kind': 'page', 'value': 1}})
    raw = m.bundle('scope', 'q', [big], time.monotonic(), expand=lambda p: p['text'] + 'b' * 2000)
    assert len(raw) <= 12000
    data = json.loads(raw)
    for s in data['sources']:
        for p in s['passages']:
            assert p['text'] == 'a' * 1600 or len(p['text']) > 1600


def test_expand_passage_reads_the_matching_chunk(monkeypatch):
    calls = []
    def cli(command, args=None, request=None):
        calls.append((command, args))
        return {'passages': [{'chunk_index': 4, 'text': 'other'}, {'chunk_index': 5, 'text': 'the whole chunk'}]}
    monkeypatch.setattr(m, 'call_cli', cli)
    assert m.expand_passage({'evidence_id': 'e', 'chunk_index': 5}) == 'the whole chunk'
    assert calls[0][0] == 'passage' and calls[0][1]['before'] == 0 and calls[0][1]['after'] == 0


def test_expand_passage_swallows_errors(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError('retrieval_process_timeout')
    monkeypatch.setattr(m, 'call_cli', boom)
    assert m.expand_passage({'evidence_id': 'e', 'chunk_index': 5}) is None


def test_shared_evidence_id_different_chunks_expand_independently():
    p1 = {'evidence_id': 'same_eid', 'chunk_index': 0, 'text': 'first_chunk_text' * 50, 'text_truncated': True,
          'location': {'kind': 'page', 'value': 1}}
    p2 = {'evidence_id': 'same_eid', 'chunk_index': 1, 'text': 'second_chunk_text' * 50, 'text_truncated': True,
          'location': {'kind': 'page', 'value': 2}}
    values = [
        ({'source_id': 'doc0', 'title': 'Title 0', 'authors': ['Author']}, p1),
        ({'source_id': 'doc0', 'title': 'Title 0', 'authors': ['Author']}, p2),
    ]

    def expand(p):
        if p.get('chunk_index') == 0:
            return 'EXPANDED_CHUNK_0_' + p['text']
        elif p.get('chunk_index') == 1:
            return 'EXPANDED_CHUNK_1_' + p['text']
        return None

    data, passages = run(values, expand)
    assert len(passages) == 2
    assert passages[0]['text'].startswith('EXPANDED_CHUNK_0_')
    assert passages[1]['text'].startswith('EXPANDED_CHUNK_1_')
    assert data['receipt']['expanded_count'] == 2


def test_worker_exception_does_not_abort_bundle():
    p1 = {'evidence_id': 'e1', 'chunk_index': 1, 'text': 'text_1' * 100, 'text_truncated': True,
          'location': {'kind': 'page', 'value': 1}}
    p2 = {'evidence_id': 'e2', 'chunk_index': 2, 'text': 'text_2' * 100, 'text_truncated': True,
          'location': {'kind': 'page', 'value': 2}}
    values = [
        ({'source_id': 'doc1', 'title': 'Doc 1', 'authors': ['Author']}, p1),
        ({'source_id': 'doc2', 'title': 'Doc 2', 'authors': ['Author']}, p2),
    ]

    def expand(p):
        if p.get('evidence_id') == 'e1':
            raise RuntimeError('worker_crash')
        return 'EXPANDED_' + p['text']

    data, passages = run(values, expand)
    assert len(passages) == 2
    # p1 failed, so it falls back to judged cut excerpt (not expanded)
    assert passages[0]['text'] == p1['text']
    assert passages[0]['text_truncated'] is True
    # p2 succeeded and expanded
    assert passages[1]['text'] == 'EXPANDED_' + p2['text']
    assert passages[1]['text_truncated'] is False
    assert data['receipt']['expanded_count'] == 1


def test_malformed_cli_response_degrades_softly(monkeypatch):
    monkeypatch.setattr(m, 'call_cli', lambda *a, **kw: {'passages': None})
    res = m.expand_passage({'evidence_id': 'e', 'chunk_index': 1})
    assert res is None

    monkeypatch.setattr(m, 'call_cli', lambda *a, **kw: {'passages': ['not-a-dict', 123, None]})
    assert m.expand_passage({'evidence_id': 'e', 'chunk_index': 1}) is None

    monkeypatch.setattr(m, 'call_cli', lambda *a, **kw: None)
    assert m.expand_passage({'evidence_id': 'e', 'chunk_index': 1}) is None
    assert m.expand_passage(None) is None
    assert m.expand_passage("not-a-dict") is None

    # In bundle:
    p1 = {'evidence_id': 'e1', 'chunk_index': 1, 'text': 'text_1' * 100, 'text_truncated': True,
          'location': {'kind': 'page', 'value': 1}}
    values = [({'source_id': 'doc1', 'title': 'Doc 1', 'authors': ['Author']}, p1)]
    data, passages = run(values, expand=m.expand_passage)
    assert len(passages) == 1
    assert passages[0]['text'] == p1['text']
    assert passages[0]['text_truncated'] is True
    assert 'expanded_count' not in data['receipt']
