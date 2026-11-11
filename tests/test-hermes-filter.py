"""Integration boundaries: bounded evidence, fail-closed filtering and scope isolation."""
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN = Path(__file__).parents[1] / 'integrations/hermes/library-query/__init__.py'
spec = importlib.util.spec_from_file_location('library_filter_tests', PLUGIN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def candidates(n=6, text_size=1600):
    return [({'source_id': f'doc{i}', 'title': f'Title {i}', 'authors': ['Author'], 'item_ids': ['item-'+ '1'*32]},
             {'evidence_id': f'e{i}', 'text': str(i)*text_size, 'text_truncated': True,
              'location': {'kind': 'page', 'value': i+1}}) for i in range(n)]


@pytest.fixture
def all_selected(monkeypatch):
    monkeypatch.setattr(m, 'judge', lambda scope, question, values: ('actual-model', [True]*len(values)))


def result(values, target=None):
    raw = m.bundle('scope', 'focused question', values, time.monotonic(), target)
    assert len(raw) <= 12000
    data = json.loads(raw)
    passages = [p for s in data['sources'] for p in s['passages']]
    assert len(passages) <= 3
    assert sum(len(p['text']) for p in passages) <= 4800
    return data


def research_call(monkeypatch, weight, target=None):
    monkeypatch.setitem(sys.modules, 'hermes_constants', SimpleNamespace(hermes_home_key=lambda: 'scope'))
    monkeypatch.setattr(m, 'preflight', lambda scope: None)
    requests = []
    def cli(command, args=None, request=None):
        requests.append(request)
        return {'sources': []}
    monkeypatch.setattr(m, 'call_cli', cli)
    args = {'question': 'focused question', 'weight': weight}
    if target is not None:
        args['target'] = target
    data = json.loads(m.handler('scope', 'research')(args))
    return requests[0], data


PDF_LOCATION = {'kind': 'page', 'value': 299, 'page_start': 299, 'page_end': 300,
                'printed_start': '277', 'printed_end': '278'}
PDF_SECTION = {'path': ['III Data Structures', '11 Hash Tables', '11.2 Hash tables'],
               'start_page': 297, 'start_printed': '275'}


def test_pdf_location_and_section_survive_bundle_and_candidates(all_selected):
    raw = {'sources': [{'source_id': 'doc0', 'passages': [
        {'evidence_id': 'e0', 'text': 'body', 'location': PDF_LOCATION, 'section': PDF_SECTION}]}]}
    (source, passage), = m.candidates_from(raw)
    assert passage['location'] == PDF_LOCATION and passage['section'] == PDF_SECTION
    data = json.loads(m.bundle('scope', 'q', [(source, passage)], time.monotonic()))
    emitted = data['sources'][0]['passages'][0]
    assert emitted['location'] == PDF_LOCATION and emitted['section'] == PDF_SECTION


def test_old_style_passage_without_new_fields_bundles_cleanly(all_selected):
    old = {'evidence_id': 'e0', 'text': 'body', 'location': {'kind': 'page', 'value': 7}}
    data = json.loads(m.bundle('scope', 'q', [({'source_id': 'doc0'}, old)], time.monotonic()))
    emitted = data['sources'][0]['passages'][0]
    assert emitted['location'] == {'kind': 'page', 'value': 7} and 'section' not in emitted


def test_source_metadata_allowlist_keeps_pdf_location_fields():
    chunk = {'evidence_id': 'e', 'chunk_index': 0, 'text': 'preview must not leak', 'location': PDF_LOCATION,
             'section': PDF_SECTION}
    kept = m.metadata(chunk)
    assert kept['location'] == PDF_LOCATION and kept['section'] == PDF_SECTION
    assert 'text' not in kept


def test_selection_order_caps_and_identity(all_selected):
    data = result(candidates())
    assert [s['passages'][0]['evidence_id'] for s in data['sources']] == ['e0', 'e1', 'e2']
    assert data['sources'][0]['passages'][0]['text_truncated'] is True
    assert data['receipt']['budget_omitted'] == 3
    assert data['receipt']['jev_model'] == 'actual-model'


def test_selected_passages_share_source_groups_and_coverage_order(all_selected):
    values = candidates(3, text_size=100)
    values[2] = (dict(values[0][0]), values[2][1])
    target = {'item_ids': ['item-' + '1' * 32]}
    data = result(values, target)
    assert len(data['sources']) == 2
    assert [p['evidence_id'] for p in data['sources'][0]['passages']] == ['e0', 'e2']
    assert [p['evidence_id'] for p in data['sources'][1]['passages']] == ['e1']
    assert data['coverage'][0]['returned_evidence_ids'] == ['e0', 'e1', 'e2']


def test_compaction_preserves_evidence_and_shrinks_ungrouped_response(all_selected):
    values = candidates(3, text_size=100)
    values[2] = (dict(values[0][0]), values[2][1])
    target = {'item_ids': ['item-' + '1' * 32]}
    data = result(values, target)
    expected = {p['evidence_id']: (source, p) for source, p in values}
    actual = {p['evidence_id']: ({k: v for k, v in source.items() if k != 'passages'}, p)
              for source in data['sources'] for p in source['passages']}
    assert actual == expected
    ungrouped = {**data, 'sources': [{**source, 'passages': [p]} for source, p in values]}
    assert len(m.encode(data)) < len(m.encode(ungrouped))


def test_grouping_uses_exact_source_metadata(all_selected):
    values = candidates(2, text_size=100)
    values[1] = (dict(values[0][0], title='Different'), values[1][1])
    data = result(values)
    assert len(data['sources']) == 2


def test_grouped_rollback_removes_only_offending_passage(all_selected):
    values = candidates(3, text_size=100)
    values[1] = (dict(values[0][0]), {**values[1][1], 'text': 'y' * 500})
    values[2] = (dict(values[0][0]), {**values[2][1], 'text': 'z' * 50})
    target = {'item_ids': ['item-' + '1' * 32]}
    baseline = m.bundle('scope', 'focused question', values[:1], time.monotonic(), target)
    budget = {**m.FULL_BUDGET, 'max_response': len(baseline) + 712}
    raw = m.bundle('scope', 'focused question', values, time.monotonic(), target, budget=budget)
    data = json.loads(raw)
    assert len(raw) <= budget['max_response']
    assert data['receipt']['selected_count'] == 2
    assert data['receipt']['budget_omitted'] == 1
    assert len(data['sources']) == 1
    assert [p['evidence_id'] for p in data['sources'][0]['passages']] == ['e0', 'e2']
    assert data['coverage'][0]['returned_evidence_ids'] == ['e0', 'e2']


def test_duplicate_removal(all_selected):
    values = candidates(2)
    values[1][1]['text'] = values[0][1]['text']
    data = result(values)
    assert data['receipt']['selected_count'] == 1
    assert data['receipt']['duplicate_count'] == 1


def test_same_source_duplicates_are_removed_before_one_judgment(monkeypatch):
    source, passage = candidates(1)[0]
    values = [(source, passage), (dict(source), {**passage, 'evidence_id': 'duplicate'})]
    calls = []
    def judge(scope, question, candidates):
        calls.append(candidates)
        return 'model', [True] * len(candidates)
    monkeypatch.setattr(m, 'judge', judge)
    data = result(values)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert data['receipt']['candidate_count'] == 2
    assert data['receipt']['judged_candidate_count'] == 1
    assert data['receipt']['prefilter_duplicate_count'] == 1
    assert data['receipt']['duplicate_count'] == 0
    assert data['sources'][0]['passages'][0]['evidence_id'] == 'e0'


def test_cross_source_duplicates_are_judged_independently(monkeypatch):
    values = candidates(2)
    values[1][1]['text'] = values[0][1]['text']
    calls = []
    def judge(scope, question, candidates):
        calls.append(candidates)
        return 'model', [True] * len(candidates)
    monkeypatch.setattr(m, 'judge', judge)
    data = result(values)
    assert len(calls) == 1 and len(calls[0]) == 2
    assert data['receipt']['judged_candidate_count'] == 2
    assert data['receipt']['prefilter_duplicate_count'] == 0
    assert data['receipt']['duplicate_count'] == 1


def test_no_matches_does_not_call_jev(monkeypatch):
    monkeypatch.setattr(m, 'judge', lambda *a: pytest.fail('must not call'))
    assert result([])['receipt']['filtering_status'] == 'no_retrieval_matches'


def test_no_selected(monkeypatch):
    monkeypatch.setattr(m, 'judge', lambda *a: ('model', [False]*2))
    assert result(candidates(2))['receipt']['filtering_status'] == 'no_selected_passages'


@pytest.mark.parametrize('error', [ValueError('raw secret candidate'), TimeoutError(), RuntimeError()])
def test_fail_closed_without_error_leaks(monkeypatch, error):
    def fail(*a):
        raise error
    monkeypatch.setattr(m, 'judge', fail)
    data = result(candidates())
    assert data['sources'] == []
    assert 'raw secret' not in json.dumps(data)
    assert data['receipt']['filtering_status'] == 'filter_unavailable'


def test_large_metadata_drops_whole_passage(all_selected):
    values = candidates(2)
    values[0][0]['title'] = 'X'*13000
    data = result(values)
    assert [s['source_id'] for s in data['sources']] == ['doc1']
    assert data['receipt']['budget_omitted'] == 1


def test_coverage_after_filter_and_serialization(all_selected):
    values = candidates(6)
    data = result(values, {'item_ids': ['item-'+'1'*32, 'item-'+'2'*32]})
    assert data['coverage'][0]['returned_evidence_ids'] == ['e0', 'e1', 'e2']
    assert data['coverage'][1]['returned_evidence_ids'] == []


def test_discovery_removes_preview_and_diagnostics():
    value = {'text': 'LEAK', 'warnings': ['LEAK'], 'item_id': 'id', 'attachments': [
        {'path': 'work.pdf', 'role': 'source', 'text': 'LEAK', 'description': 'LEAK'}]}
    assert 'LEAK' not in json.dumps(m.metadata(value))


def test_cli_direct_and_clean_environment(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', 'forbidden')
    monkeypatch.setenv('TYPESAFE_API_KEY', 'secret')
    def run(command, **kw):
        assert command[1:] == ['research', '--json', '--input', '-', '--cache-mode', 'use']
        assert 'TYPESAFE_API_KEY' not in kw['env'] and 'PYTHONPATH' not in kw['env']
        assert json.loads(kw['input'])['scope'] == 'catalogued'
        return SimpleNamespace(returncode=0, stdout='{}')
    monkeypatch.setattr(m.subprocess, 'run', run)
    m.call_cli('research', request={'scope': 'catalogued'})


def test_cli_uses_remaining_tool_deadline(monkeypatch):
    monkeypatch.setattr(m.time, 'monotonic', lambda: 10.0)
    token = m.TOOL_DEADLINE.set(12.5)
    def run(*args, **kw):
        assert kw['timeout'] == 2.5
        assert float(kw['env']['LIBRARY_QUERY_RESEARCH_TIMEOUT']) == 1.5
        return SimpleNamespace(returncode=0, stdout='{}')
    monkeypatch.setattr(m.subprocess, 'run', run)
    try:
        m.call_cli('research', request={})
        m.TOOL_DEADLINE.set(9.0)
        monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('expired work started'))
        with pytest.raises(m.RetrievalError, match='retrieval_process_timeout'):
            m.call_cli('research', request={})
    finally:
        m.TOOL_DEADLINE.reset(token)


def test_judge_does_not_start_after_tool_deadline(monkeypatch, tmp_path):
    calls = []
    hc, ss, home_token, scope_token = install_judge(monkeypatch, tmp_path, lambda args: calls.append(args))
    token = m.TOOL_DEADLINE.set(time.monotonic() - 1)
    try:
        with pytest.raises(ValueError, match='filter_timeout'):
            m.judge(hc.hermes_home_key(), 'question', candidates(1))
        assert calls == []
    finally:
        m.TOOL_DEADLINE.reset(token)
        ss.reset_secret_scope(scope_token)
        hc.reset_hermes_home_override(home_token)


@pytest.mark.parametrize('stderr,expected', [
    ('{"error":"research_timeout","message":"private diagnostic"}', 'research_timeout'),
    ('{"error":"embedding_api_error"}', 'embedding_api_error'),
    ('{"error":"private diagnostic"}', 'retrieval_failed'),
    ('{"error":[]}', 'retrieval_failed'),
    ('private diagnostic', 'retrieval_failed'),
    ('[]', 'retrieval_failed'),
])
def test_cli_errors_preserve_only_known_categories(monkeypatch, stderr, expected):
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=4, stdout='', stderr=stderr))
    with pytest.raises(m.RetrievalError, match=expected) as caught:
        m.call_cli('research', request={})
    assert 'private diagnostic' not in str(caught.value)


def test_process_timeout_receipt_is_specific_and_contains_no_output(monkeypatch):
    monkeypatch.setitem(sys.modules, 'hermes_constants', SimpleNamespace(hermes_home_key=lambda: 'scope'))
    monkeypatch.setattr(m, 'preflight', lambda scope: None)
    def timeout(*a, **kw):
        raise m.subprocess.TimeoutExpired('library-query', 30, output='private output')
    monkeypatch.setattr(m.subprocess, 'run', timeout)
    raw = m.handler('scope', 'research')({'question': 'focused question', 'weight': 'important'})
    data = json.loads(raw)
    assert data['sources'] == []
    assert data['receipt']['filtering_reason'] == 'retrieval_process_timeout'
    assert 'private output' not in raw


@pytest.mark.parametrize('target', [None, {}, {'item_ids': []}, {'document_ids': ['bad']},
    {'item_ids': ['item-'+'1'*32], 'document_ids': ['doc_'+'1'*64]},
    {'document_ids': ['doc_'+'1'*64], 'roles': ['source']}])
def test_invalid_targets(target):
    with pytest.raises(ValueError):
        m.validate_target(target)


def test_exact_target_and_fixed_retrieval_budget(monkeypatch, all_selected, tmp_path):
    hc = pytest.importorskip("hermes_constants")
    token = hc.set_hermes_home_override(str(tmp_path))
    monkeypatch.setattr(m, 'preflight', lambda scope: None)
    target = {'item_ids': ['item-'+'1'*32], 'roles': ['source', 'note']}
    try:
        def cli(command, request):
            assert command == 'research'
            assert request['target'] == target and request['schema_version'] == 2
            assert request['max_sources'] == 6 and request['max_passages_per_source'] == 2
            assert request['max_excerpt_chars'] == 1600 and request['scope'] == 'catalogued'
            return {'sources': []}
        monkeypatch.setattr(m, 'call_cli', cli)
        data = json.loads(m.handler(hc.hermes_home_key(), 'research')(
            {'question': 'question', 'weight': 'important', 'target': target}))
        assert data['receipt']['filtering_status'] == 'no_retrieval_matches'
    finally:
        hc.reset_hermes_home_override(token)


@pytest.mark.parametrize('weight', ['light', 'meaningful', 'important'])
def test_weight_resolves_explicit_full_fallback_budget(monkeypatch, weight):
    request, data = research_call(monkeypatch, weight)
    assert request['schema_version'] == 1 and request['scope'] == 'catalogued'
    for key in ('max_sources', 'max_passages_per_source', 'max_excerpt_chars'):
        assert request[key] == m.FULL_BUDGET[key]
    assert data['receipt']['weight'] == weight
    assert data['receipt']['resolved_budget'] == m.FULL_BUDGET


def test_light_exact_target_uses_full_budget(monkeypatch):
    target = {'item_ids': ['item-' + '1' * 32], 'roles': ['source']}
    request, data = research_call(monkeypatch, 'light', target)
    assert request['schema_version'] == 2 and request['target'] == target
    for key in ('max_sources', 'max_passages_per_source', 'max_excerpt_chars'):
        assert request[key] == m.FULL_BUDGET[key]
    assert data['receipt']['weight'] == 'light'
    assert data['receipt']['resolved_budget'] == m.FULL_BUDGET


@pytest.mark.parametrize('overrides,expected', [({'max_passages': 2}, 2), ({'max_text': 100}, 1)])
def test_bundle_honors_resolved_output_budgets(all_selected, overrides, expected):
    budget = {**m.FULL_BUDGET, **overrides}
    raw = m.bundle('scope', 'focused question', candidates(4, text_size=60), time.monotonic(), budget=budget)
    data = json.loads(raw)
    assert data['receipt']['selected_count'] == expected
    assert data['receipt']['budget_omitted'] == 4 - expected
    assert len(raw) <= budget['max_response']


def test_invalid_weight_is_rejected_without_retrieval(monkeypatch):
    monkeypatch.setitem(sys.modules, 'hermes_constants', SimpleNamespace(hermes_home_key=lambda: 'scope'))
    monkeypatch.setattr(m, 'call_cli', lambda *a, **k: pytest.fail('must not retrieve'))
    data = json.loads(m.handler('scope', 'research')({'question': 'q', 'weight': 'verbose'}))
    assert data['sources'] == []
    assert data['receipt']['filtering_status'] == 'consultation_unavailable'
    assert 'resolved_budget' not in data['receipt']


def test_filter_failure_preserves_research_budget_receipt(monkeypatch):
    monkeypatch.setitem(sys.modules, 'hermes_constants', SimpleNamespace(hermes_home_key=lambda: 'scope'))
    monkeypatch.setattr(m, 'preflight', lambda scope: (_ for _ in ()).throw(ValueError('filter_missing_credential')))
    monkeypatch.setattr(m, 'call_cli', lambda *a, **k: pytest.fail('must not retrieve'))
    data = json.loads(m.handler('scope', 'research')({'question': 'q', 'weight': 'meaningful'}))
    assert data['sources'] == []
    assert data['receipt']['filtering_status'] == 'filter_unavailable'
    assert data['receipt']['weight'] == 'meaningful'
    assert data['receipt']['resolved_budget'] == m.FULL_BUDGET


def install_judge(monkeypatch, tmp_path, output):
    hc = pytest.importorskip("hermes_constants")
    from tools.registry import registry
    ss = pytest.importorskip("agent.secret_scope")
    home = str(tmp_path)
    token = hc.set_hermes_home_override(home)
    st = ss.set_secret_scope({'TYPESAFE_API_KEY': 'own'}, profile_home=home)
    def lookup(name, *, scope):
        assert name == 'jev_evaluate' and scope == hc.hermes_home_key(home)
        return SimpleNamespace(handler=lambda args: output(args))
    monkeypatch.setattr(registry, 'snapshot_registration', lookup)
    return hc, ss, token, st


@pytest.mark.parametrize('response', [None, {}, {'ok': False}, {'ok': True, 'model': 'm', 'answers': {}},
    {'ok': True, 'model': 'm', 'answers': {'p0': {'type': 'choice', 'choice': 'other'}}}])
def test_malformed_jev(monkeypatch, tmp_path, response):
    hc, ss, token, st = install_judge(monkeypatch, tmp_path, lambda args: response)
    try:
        with pytest.raises(ValueError):
            m.judge(hc.hermes_home_key(), 'question', candidates(1))
    finally:
        ss.reset_secret_scope(st)
        hc.reset_hermes_home_override(token)


def test_timeout_no_retry(monkeypatch, tmp_path):
    calls = []
    def slow(args):
        calls.append(args)
        time.sleep(.1)
        return {}
    hc, ss, token, st = install_judge(monkeypatch, tmp_path, slow)
    monkeypatch.setattr(m, 'JEV_TIMEOUT', .01)
    try:
        with pytest.raises(ValueError, match='filter_timeout'):
            m.judge(hc.hermes_home_key(), 'question', candidates(1))
        assert len(calls) == 1
    finally:
        ss.reset_secret_scope(st)
        hc.reset_hermes_home_override(token)


def test_concurrent_handler_and_credential_isolation(monkeypatch, tmp_path):
    hc = pytest.importorskip("hermes_constants")
    from tools.registry import registry
    ss = pytest.importorskip("agent.secret_scope")
    homes = [str(tmp_path / p) for p in ('code', 'second', 'study')]
    def lookup(name, *, scope):
        assert scope in homes
        def evaluate(args):
            assert hc.hermes_home_key() == scope
            assert ss.get_secret('TYPESAFE_API_KEY') == scope
            assert len(args['questions']) == 2
            return json.dumps({'ok': True, 'model': scope.split('/')[-1], 'answers': {
                'p0': {'type': 'choice', 'choice': 'directly_useful'},
                'p1': {'type': 'choice', 'choice': 'background_only'}}})
        return SimpleNamespace(handler=evaluate)
    monkeypatch.setattr(registry, 'snapshot_registration', lookup)
    def work(home):
        ht = hc.set_hermes_home_override(home)
        st = ss.set_secret_scope({'TYPESAFE_API_KEY': home}, profile_home=home)
        try:
            return m.judge(home, 'question', candidates(2))
        finally:
            ss.reset_secret_scope(st)
            hc.reset_hermes_home_override(ht)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(work, homes))
    assert results == [(p, [True, False]) for p in ('code', 'second', 'study')]


def test_missing_scoped_handler_does_not_fall_back(monkeypatch, tmp_path):
    hc, ss, token, st = install_judge(monkeypatch, tmp_path, lambda args: {})
    from tools.registry import registry
    monkeypatch.setattr(registry, 'snapshot_registration', lambda *a, **kw: None)
    try:
        with pytest.raises(ValueError, match='filter_unavailable'):
            m.judge(hc.hermes_home_key(), 'question', candidates(1))
    finally:
        ss.reset_secret_scope(st)
        hc.reset_hermes_home_override(token)


def test_passage_context_filtered_and_no_unfiltered_metadata_route(monkeypatch):
    monkeypatch.setattr(m, "preflight", lambda scope: None)
    monkeypatch.setitem(sys.modules, 'hermes_constants', SimpleNamespace(hermes_home_key=lambda: 'scope'))
    seen = []
    def cli(command, args=None, request=None):
        if command == 'research':
            assert request['target'] == {'document_ids': ['doc_'+'1'*64]}
            return {'sources': [{'source_id': 'doc_'+'1'*64, 'title': 'Original', 'authors': ['A']}]}
        return {'document_id': 'doc_'+'1'*64, 'passages': [p for s, p in candidates(2)], 'omitted_chunk_indexes': []}
    def judge(scope, question, values):
        seen.extend(values)
        return 'model', [False, True]
    monkeypatch.setattr(m, 'call_cli', cli)
    monkeypatch.setattr(m, 'judge', judge)
    data = json.loads(m.handler('scope', 'passage')({'evidence_id': 'id', 'question': 'gap'}))
    assert len(seen) == 2 and len(data['sources']) == 1
    assert data['sources'][0]['title'] == 'Original'
    assert data['sources'][0]['passages'][0]['evidence_id'] == 'e1'
    assert 'weight' not in data['receipt'] and 'resolved_budget' not in data['receipt']


@pytest.mark.parametrize("command,args", [
    ("research", {"question": "focused question", "weight": "meaningful"}),
    ("passage", {"question": "specific gap", "evidence_id": "identity"}),
])
def test_missing_filter_stops_before_any_retrieval(monkeypatch, tmp_path, command, args):
    hc = pytest.importorskip("hermes_constants")
    token = hc.set_hermes_home_override(str(tmp_path))
    try:
        def fail(scope):
            raise ValueError("filter_missing_credential")
        monkeypatch.setattr(m, "preflight", fail)
        monkeypatch.setattr(m, "call_cli", lambda *a, **k: pytest.fail("must not retrieve"))
        data = json.loads(m.handler(hc.hermes_home_key(), command)(args))
        assert data["sources"] == []
        assert data["receipt"]["filtering_status"] == "filter_unavailable"
        assert data["receipt"]["filtering_reason"] == "filter_missing_credential"
        assert data["receipt"]["candidate_count"] == 0
    finally:
        hc.reset_hermes_home_override(token)


@pytest.mark.parametrize("backend,model,credential_name,credential,error", [
    ("typesafe", "jev-latest", "TYPESAFE_API_KEY", "", "filter_missing_credential"),
    ("openrouter", "typesafe/jev-1.14", "OPENROUTER_API_KEY", "", "filter_missing_credential"),
    ("cloudflare", "typesafe/jev", "CLOUDFLARE_JEV_API_TOKEN", "own", "filter_not_configured"),
    (None, None, "TYPESAFE_API_KEY", "own", "filter_not_configured"),
    ("openrouter", "typesafe/jev-latest", "OPENROUTER_API_KEY", "own", "filter_not_configured"),
    ("typesafe", "jev-latest", "TYPESAFE_API_KEY", "own", None),
    ("openrouter", "typesafe/jev-1.14", "OPENROUTER_API_KEY", "own", None),
])
def test_preflight_uses_only_supported_active_configuration(monkeypatch, tmp_path, backend, model,
                                                             credential_name, credential, error):
    hc, ss, token, st = install_judge(monkeypatch, tmp_path, lambda args: {})
    config = pytest.importorskip("hermes_cli.config")
    monkeypatch.setattr(config, "load_config_readonly", lambda: {
        "plugins": {"entries": {"jev": {"settings": {"connection": {
            "backend": backend, "model": model}}}}}})
    own = ss.set_secret_scope({credential_name: credential}, profile_home=str(tmp_path))
    try:
        if error:
            with pytest.raises(ValueError, match=error):
                m.preflight(hc.hermes_home_key())
        else:
            m.preflight(hc.hermes_home_key())
    finally:
        ss.reset_secret_scope(own)
        ss.reset_secret_scope(st)
        hc.reset_hermes_home_override(token)
