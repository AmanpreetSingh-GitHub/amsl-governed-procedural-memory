import copy
import hashlib
import json
import unittest
import io
import os
import re
import runpy
import sys
import contextlib
import subprocess
from unittest import mock
from types import SimpleNamespace

from test_criterion_rationale import ROOT, access, worker
import run_investigation_to_db as runner
import model_dispatch as dispatch
import procedural_retrieval as retrieval
from test_expected_memory import CASE, ORIGIN, completed_worker_result, episode_row, release_row


class WorkerBoundaryTests(unittest.TestCase):
    def test_paired_manifest_rejects_standalone(self):
        release = {'manifest': {'experiment_protocol': 'PAIRED_EXPERIMENT_V1'}}
        argv = ['runner', 'run', '--case', 'CA274982', '--stage', '0', '--model', 'stub']
        with mock.patch.object(runner.sys, 'argv', argv), \
             mock.patch.object(runner, 'authorize_model', return_value='configured_local'), \
             mock.patch.object(runner, 'approved_release', return_value=release), \
             mock.patch.object(runner, 'run_agent') as run, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(), 1)
        run.assert_not_called()

    def test_worker_rejects_retired_memory_keys(self):
        evidence = {'interpretation.json': {}, 'guideline_v2.json': {}}
        memory = {'title': 'method', 'content': 'guidance'}
        retired_inputs = {
            'procedural_memory': memory,
            'procedural_memory_support': {'supporting_case_count': 2},
        }
        for key, value in retired_inputs.items():
            with self.subTest(key=key):
                payload = {'evidence': evidence, 'procedural_memories': [memory], key: value}
                stdin = io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()))
                with mock.patch.object(worker.sys, 'stdin', stdin), \
                     mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'stub', 'OPENAI_MODEL': 'stub'}), \
                     mock.patch.object(worker, 'execute') as execute:
                    with self.assertRaisesRegex(ValueError, 'Legacy memory inputs are retired'):
                        worker.main()
                    execute.assert_not_called()


    def test_removed_review_mode_cannot_fall_through_to_investigation(self):
        stdin = io.TextIOWrapper(io.BytesIO(b'{"mode":"review_learning_draft","evidence":{}}'))
        with mock.patch.object(worker.sys, 'stdin', stdin), mock.patch.object(worker, 'execute') as execute:
            with self.assertRaisesRegex(ValueError, 'Unknown worker mode'):
                worker.main()
            execute.assert_not_called()

    def test_stage_requires_real_supplied_references_for_approval(self):
        chat = mock.Mock()
        model = chat.ChatOpenAI.return_value
        reviewer = model.with_structured_output.return_value
        with mock.patch.dict(sys.modules, {'langchain_openai': chat,
                                          'pydantic': SimpleNamespace(BaseModel=object)}):
            stages = runpy.run_path(str(ROOT / 'src' / 'first_investigation.py'))
        evidence = {
            'interpretation.json': {'criteria': [{'outcome': 'Met'}]},
            'clinvar/SCV000000001.1.json': {'field/with~escapes': 'Reported'},
        }
        state = {'evidence': evidence, 'report': 'Report', 'reflection': 'DECISION: PROPOSE'}
        valid = 'interpretation.json#/criteria/0/outcome'
        other = 'clinvar/SCV000000001.1.json#/field~1with~0escapes'
        unknown = 'interpretation.json#/criteria/1/outcome'
        cases = [
            ([valid, other], 'APPROVE', 'APPROVE'),
            ([unknown], 'APPROVE', 'DEFER'),
            (['interpretation.json'], 'APPROVE', 'DEFER'),
            (['Report heading: Criterion outcomes'], 'APPROVE', 'DEFER'),
            ([valid, unknown], 'APPROVE', 'DEFER'),
            (['Interpretation.json#/criteria/0/outcome'], 'APPROVE', 'DEFER'),
            (['evidence/' + valid], 'APPROVE', 'DEFER'),
            (['interpretation.json:criteria.0.outcome'], 'APPROVE', 'DEFER'),
            (['interpretation.json#/criteria/00/outcome'], 'APPROVE', 'DEFER'),
            (['clinvar/SCV000000001.1.json#/field~1with~2escapes'], 'APPROVE', 'DEFER'),
            ([''], 'APPROVE', 'DEFER'),
            ([unknown], 'DEFER', 'DEFER'),
            ([unknown], 'REJECT', 'REJECT'),
        ]
        with mock.patch.dict(os.environ, {'OPENAI_MODEL': 'stub'}):
            for references, decision, expected in cases:
                with self.subTest(references=references, decision=decision):
                    reviewer.invoke.return_value.model_dump.return_value = {
                        'decision': decision, 'grounded': True, 'reusable': True, 'contradictions': [],
                        'evidence_references': references, 'rationale': 'Review',
                    }
                    validation = stages['validate'](state)['validation']
                    self.assertEqual(validation['decision'], expected)
                    self.assertEqual(validation['evidence_references'], references)
        messages = dict(reviewer.invoke.call_args.args[0])
        self.assertIn('exact JSON Pointer paths', messages['system'])
        self.assertIn('interpretation.json#/criteria/0/outcome', messages['system'])
        self.assertIn('exactly matches a key in evidence', messages['system'])
        self.assertIn('zero-based array indices', messages['system'])
        self.assertIn('descriptive report headings, prose summaries, or invented paths', messages['system'])
        request = json.loads(messages['human'].split('\n\n', 1)[1])
        self.assertEqual(request['evidence_source_filenames'], sorted(evidence))
        self.assertEqual(request['evidence'], evidence)



class UnifiedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.releases = {CASE: release_row(CASE, 4), ORIGIN: release_row(ORIGIN, 1)}
        self.releases[CASE]['sources'][0]['document'] = {'caid': CASE}
        self.episodes = {3: episode_row(3, self.releases[ORIGIN])}
        self.origin_queries = []
        first = {
            'memory_id': 1, 'version_id': 11, 'version_number': 1, 'title': 'Procedure',
            'content': 'Check each approved source', 'original_episode_id': 3,
            'change_reason': 'Proposed in origin episode ' + self.episodes[3]['run_id'],
            'embedding_model': retrieval.EMBEDDING_MODEL,
            'similarity': 0.9,
        }
        second = dict(first, memory_id=2, version_id=22, content='Compare reported outcomes', similarity=0.8)
        self.memories = [second, first]
        self.statements = []
        self.embed = self.enterContext(mock.patch.object(retrieval, 'embed', return_value=[1.0] * 768))
        for module in (access, retrieval):
            self.enterContext(mock.patch.object(module, 'rows', side_effect=self.database_rows))
        self.enterContext(mock.patch.dict(os.environ, {
            'OPENAI_BASE_URL': 'http://127.0.0.1:8000/v1', 'OPENAI_API_KEY': 'stub',
        }))
        worker_result = completed_worker_result()
        response = subprocess.CompletedProcess([], 0, stdout='AGENT_RESULT_JSON:' + json.dumps(worker_result))
        self.dispatch = self.enterContext(mock.patch.object(dispatch.subprocess, 'run', return_value=response))
        self.release = runner.approved_release(CASE, 0)

    def database_rows(self, sql):
        if 'FROM scientific.evidence_releases er' in sql:
            case = next(case for case in self.releases if access.sql_text(case) in sql)
            return [copy.deepcopy(self.releases[case])]
        if 'FROM agent_memory.episodes e' in sql:
            self.origin_queries.append(sql)
            episode_id = int(re.search(r'e.episode_id=(\d+)', sql).group(1))
            episode = self.episodes.get(episode_id)
            return [copy.deepcopy(episode)] if episode else []
        if 'FROM agent_memory.memories m' in sql:
            self.statements.append(sql)
            if 'JOIN agent_memory.embeddings em' in sql:
                return copy.deepcopy(self.memories)
            return [item for item in self.memories
                    if f"m.memory_id={item['memory_id']} AND v.version_id={item['version_id']}" in sql]
        raise AssertionError('Unexpected database query')

    def test_retrieval_candidate_query_and_audit_provenance(self):
        ranked = runner.retrieve_procedural_memories(CASE, self.release)
        query, = self.statements
        self.assertIn('v.version_id=m.active_version_id', query)
        self.assertIn("m.status='VALIDATED'", query)
        self.assertIn('v.change_reason', query)
        for origin_query in self.origin_queries:
            self.assertIn('e.episode_id=3', origin_query)
            self.assertIn("'status',e.status", origin_query)
        self.assertEqual([item['memory_id'] for item in ranked], [1, 2])
        self.assertTrue(all(item['similarity'] == item['score'] for item in ranked))
        embedded_query = retrieval.INVESTIGATION_QUESTION + f'\ninterpretation.json#/caid: {CASE}'
        self.embed.assert_called_once_with(embedded_query)
        for rank, item in enumerate(ranked, 1):
            provenance = item['retrieval_provenance']
            self.assertEqual(provenance['policy'], 'CONTEXT_AWARE_PROCEDURAL_RETRIEVAL_V2')
            self.assertEqual(provenance['query'], embedded_query)
            self.assertEqual(provenance['query_sha256'], hashlib.sha256(embedded_query.encode()).hexdigest())
            self.assertEqual(provenance['rank'], rank)
            self.assertTrue(provenance['ranking_only'])
            self.assertFalse(provenance['selected_for_injection'])
            self.assertEqual(item['content_sha256'], hashlib.sha256(item['content'].encode()).hexdigest())

    def refine(self, source_case):
        release = self.releases[source_case]
        release['sources'][0]['canonical_key'] = 'refined-interpretation'
        release['sources'][0]['sha256'] = 'd' * 64
        release['manifest']['evidence'][0]['sha256'] = 'd' * 64
        release['receipt']['evidence'][0]['sha256'] = 'd' * 64
        release['sources'][0]['declared_sha256'] = 'd' * 64
        episode = self.episodes[8] = episode_row(8, release)
        metadata = {'episode_id': 8, 'run_id': episode['run_id'], 'record_sha256': episode['record_sha256'],
                    'decision': 'REFINE', 'target_memory_id': 1, 'active_version_id': 11}
        revised = dict(self.memories[1], version_id=12, version_number=2,
                       change_reason='DRAFT_LEARNING:8:REFINE:' + json.dumps(metadata))
        self.memories[:] = [revised]
        return revised, metadata

    def test_refine_uses_its_own_origin_for_retrieval_and_selection(self):
        refined_case = 'CA555'
        self.releases[refined_case] = release_row(refined_case, 8)
        revised, _ = self.refine(refined_case)
        self.episodes[3] = episode_row(3, self.releases[CASE])
        ranked = runner.retrieve_procedural_memories(CASE, self.release)
        item, = ranked
        self.assertEqual((item['source_episode_id'], item['source_case_id'], item['source_release_id']),
                         (8, refined_case, 8))
        self.assertEqual(item['origin_evidence'][0], {'canonical_key': 'refined-interpretation', 'sha256': 'd' * 64})
        review = {'decisions': [
            {'memory_id': 1, 'version_id': 12, 'decision': 'APPLY',
             'source_field_references': ['interpretation.json#/caid'], 'reason': 'Reviewed case context'}
        ], 'conflicts': []}
        selection = runner.select_applicable_memories(self.release, ranked, review)
        self.assertEqual(selection['selected_memories'], [item])
        self.assertEqual([re.search(r'e.episode_id=(\d+)', query).group(1)
                          for query in self.origin_queries], ['8', '8'])

        self.episodes[8] = episode_row(8, self.releases[CASE])
        with self.assertRaisesRegex(ValueError, 'source changed or is ineligible'):
            runner.select_applicable_memories(self.release, ranked, review)
        self.assertEqual(runner.retrieve_procedural_memories(CASE, self.release), [])
        self.dispatch.assert_not_called()

    def test_refine_rejects_unverifiable_provenance(self):
        self.releases['CA555'] = release_row('CA555', 8)
        revised, metadata = self.refine('CA555')
        for reason, rejection in (
            ('Manual revision without a source episode', 'Revised memory version lacks its own source episode'),
            ('DRAFT_LEARNING:3:REFINE:' + json.dumps(metadata), 'Conflicting or incomplete memory version provenance'),
            ('DRAFT_LEARNING:8:REFINE:' + json.dumps(dict(metadata, target_memory_id=99)),
             'REFINE provenance targets a different memory or invalid version'),
            ('DRAFT_LEARNING:8:REFINE:' + json.dumps(dict(metadata, record_sha256='0' * 64)),
             'Memory version source episode identity or recorded origin hash mismatch'),
            ('DRAFT_LEARNING:8:REFINE:{', 'Invalid memory version provenance JSON:'),
        ):
            with self.subTest(reason=reason):
                revised['change_reason'] = reason
                with contextlib.redirect_stderr(io.StringIO()) as diagnostics:
                    self.assertEqual(runner.retrieve_procedural_memories(CASE, self.release), [])
                self.assertIn(f'memory_id=1 | version_id=12 | {rejection}', diagnostics.getvalue())
        revised['change_reason'] = 'DRAFT_LEARNING:8:REFINE:' + json.dumps(metadata)
        del self.episodes[8]
        with contextlib.redirect_stderr(io.StringIO()) as diagnostics:
            self.assertEqual(runner.retrieve_procedural_memories(CASE, self.release), [])
        self.assertIn('memory_id=1 | version_id=12 | Memory version source episode not found',
                      diagnostics.getvalue())

    def test_retrieval_excludes_invalid_version_but_ranks_valid_versions(self):
        self.memories.append(dict(self.memories[-1], memory_id=4, version_id=44, version_number=2,
                                  change_reason='Untraceable revision', similarity=0.99))
        with contextlib.redirect_stderr(io.StringIO()) as diagnostics:
            ranked = runner.retrieve_procedural_memories(CASE, self.release)
        self.assertEqual([item['version_id'] for item in ranked], [11, 22])
        self.assertEqual([item['retrieval_provenance']['rank'] for item in ranked], [1, 2])
        self.assertIn('memory_id=4 | version_id=44 | Revised memory version lacks its own source episode',
                      diagnostics.getvalue())

    def test_invalid_origin_release_is_excluded_without_blocking_other_versions(self):
        revised_case = 'CA555'
        self.releases[revised_case] = release_row(revised_case, 8)
        self.episodes[8] = episode_row(8, self.releases[revised_case])
        self.memories[0]['original_episode_id'] = 8
        self.memories[0]['change_reason'] = 'Proposed in origin episode ' + self.episodes[8]['run_id']
        self.releases[revised_case]['receipt']['status'] = 'revoked'
        with contextlib.redirect_stderr(io.StringIO()) as diagnostics:
            ranked = runner.retrieve_procedural_memories(CASE, self.release)
        self.assertEqual([item['version_id'] for item in ranked], [11])
        self.assertIn('memory_id=2 | version_id=22 | Receipt has not approved this investigation',
                      diagnostics.getvalue())

    def test_retrieval_propagates_database_and_unexpected_errors(self):
        for error in (RuntimeError('PostgreSQL unavailable'), ValueError('Unexpected programming error')):
            with self.subTest(error=error):
                def query(sql):
                    if 'FROM agent_memory.episodes e' in sql:
                        raise error
                    return self.database_rows(sql)

                with mock.patch.object(access, 'rows', side_effect=query), \
                     contextlib.redirect_stderr(io.StringIO()) as diagnostics:
                    with self.assertRaises(type(error)) as failure:
                        runner.retrieve_procedural_memories(CASE, self.release)
                self.assertIs(failure.exception, error)
                self.assertEqual(diagnostics.getvalue(), '')

    def test_dispatch_rejects_memory_with_newly_invalid_provenance(self):
        ranked = runner.retrieve_procedural_memories(CASE, self.release)
        review = {'decisions': [
            {'memory_id': item['memory_id'], 'version_id': item['version_id'], 'decision': decision,
             'source_field_references': ['interpretation.json#/caid'], 'reason': 'Reviewed case context'}
            for item, decision in zip(ranked, ('APPLY', 'UNCERTAIN'))
        ], 'conflicts': []}
        selection = runner.select_applicable_memories(self.release, ranked, review)
        self.memories[-1]['change_reason'] = 'DRAFT_LEARNING:3:NEW:{'
        with self.assertRaisesRegex(ValueError, 'Invalid memory version provenance JSON'):
            runner.run_agent(self.release, 'stub', procedural_memories=selection['selected_memories'],
                             selection_report=selection['selection_report'])
        self.dispatch.assert_not_called()

    def test_top_three_ties_use_memory_and_version_ids(self):
        for item in self.memories:
            item['similarity'] = 0.8
        self.memories += [dict(self.memories[0], memory_id=3, version_id=33),
                          dict(self.memories[0], memory_id=4, version_id=44)]
        candidates = runner.retrieve_procedural_memories(CASE, self.release)
        self.assertEqual([item['memory_id'] for item in candidates], [1, 2, 3])

    def test_applicability_checks_exact_context_and_conflicts(self):
        ranked = runner.retrieve_procedural_memories(CASE, self.release)
        review = {'decisions': [
            {'memory_id': item['memory_id'], 'version_id': item['version_id'], 'decision': 'APPLY',
             'reason': 'Supported', 'source_field_references': ['interpretation.json#/caid']}
            for item in ranked
        ], 'conflicts': [{'memory_versions': [
            {k: item[k] for k in ('memory_id', 'version_id')} for item in ranked],
            'reason': 'Incompatible procedures'}]}
        with mock.patch.object(dispatch, 'call_worker', return_value=review) as call:
            self.assertEqual(runner.call_memory_applicability(self.release, ranked, 'stub'), review)
            self.assertEqual(call.call_args.args[0]['source_field_paths'], ['interpretation.json#/caid'])
            ranked[0]['retrieval_provenance']['query'] = 'unapproved context'
            with self.assertRaisesRegex(ValueError, 'provenance'):
                runner.call_memory_applicability(self.release, ranked, 'stub')
            call.assert_called_once()
        ranked = runner.retrieve_procedural_memories(CASE, self.release)
        with self.assertRaisesRegex(ValueError, 'reported conflict'):
            runner.select_applicable_memories(self.release, ranked, review)

    def test_empty_retrieval_produces_empty_governed_selection(self):
        self.memories.clear()
        ranked = runner.retrieve_procedural_memories(CASE, self.release)
        self.assertEqual(ranked, [])
        review = runner.call_memory_applicability(self.release, ranked, 'stub')
        self.assertEqual(review, {'decisions': [], 'conflicts': []})
        selection = runner.select_applicable_memories(self.release, ranked, review)
        self.assertEqual(selection['selected_memories'], [])
        report = selection['selection_report']
        self.assertEqual(report['release_id'], self.release['release_id'])
        self.assertEqual(report['ranked_candidates'], [])
        self.assertEqual(report['decisions'], [])
        self.assertEqual(report['conflicts'], [])
        self.dispatch.assert_not_called()

    def test_runner_payload_and_retrieval_provenance(self):
        ranked = runner.retrieve_procedural_memories(CASE, self.release)
        review = {'decisions': [
            {'memory_id': item['memory_id'], 'version_id': item['version_id'], 'decision': decision,
             'source_field_references': ['interpretation.json#/caid'], 'reason': 'Reviewed case context'}
            for item, decision in zip(ranked, ('SKIP', 'APPLY'))
        ], 'conflicts': []}
        selection = runner.select_applicable_memories(self.release, ranked, review)
        selected = ranked[1]
        self.assertEqual(selection['selected_memories'], [selected])
        record = runner.run_agent(self.release, 'stub', procedural_memories=selection['selected_memories'],
                                  selection_report=selection['selection_report'])

        self.dispatch.assert_called_once()
        payload = json.loads(self.dispatch.call_args.kwargs['input'])
        self.assertEqual(payload, {
            'evidence': self.release['evidence'],
            'procedural_memories': [{'title': selected['title'], 'content': selected['content']}],
        })
        audit = record['procedural_multi_memory_selection']
        self.assertEqual([row['memory_id'] for row in audit['ranked_candidates']], [1, 2])
        self.assertEqual(audit['ranked_candidates'][1]['retrieval_provenance'], selected['retrieval_provenance'])
        self.assertEqual(audit['decisions'], selection['selection_report']['decisions'])
        self.assertEqual(audit['application_status'], 'not_verified')
        self.assertTrue(record['procedural_memory_injection_checked'])
        marker = record['procedural_memory_supplied']
        self.assertTrue(marker['guidance_supplied_to_worker'])
        for key in ('memory_id', 'version_id', 'content_sha256'):
            self.assertEqual(marker[key], selected[key])


if __name__ == '__main__':
    unittest.main()
