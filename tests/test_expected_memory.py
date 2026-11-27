"""Offline tests for the runner's unified-memory CLI and dispatch guards."""

import contextlib
import copy
import hashlib
import io
import json
import os
import unittest
from unittest import mock

from test_criterion_rationale import access
import run_investigation_to_db as runner
import episode_persistence as episodes
import model_dispatch as dispatch
import procedural_retrieval as retrieval


CASE = 'CA274982'
ORIGIN = 'CA1585329339'


def release_row(case, release_id):
    experiment_id = f'EXP-{release_id}'
    sources = [
        {
            'source_id': index, 'source_type': kind, 'canonical_key': kind,
            'version': '1', 'sha256': str(index) * 64, 'snapshot_path': f'evidence/{name}',
            'declared_sha256': str(index) * 64,
            'document': {'caid': case} if kind == 'INTERPRETATION' else {'version': '2'},
        }
        for index, (name, kind) in enumerate(
            [('interpretation.json', 'INTERPRETATION'), ('guideline_v2.json', 'GUIDELINE')], 1)
    ]
    release = {
        'case_id': case, 'stage': 0, 'release_id': release_id, 'experiment_id': experiment_id,
        'sources': sources,
        'manifest': {
            'case_id': case, 'stage': 0, 'experiment_id': experiment_id,
            'evidence': [{'file': source['snapshot_path'], 'sha256': source['sha256']}
                         for source in sources],
        },
    }
    release['receipt'] = copy.deepcopy(dict(release['manifest'], status=access.APPROVED))
    return release


def completed_worker_result():
    return {
        'execution_status': 'COMPLETED',
        'trace': [{'stage': stage, 'status': 'COMPLETED'} for stage in ('investigate', 'reflect', 'validate')],
        'started_at_utc': '2026-01-01T00:00:00Z', 'completed_at_utc': '2026-01-01T00:00:01Z',
        'report': 'Test report', 'reflection': 'DECISION: ABSTAIN', 'validation': {'decision': 'DEFER'},
    }


def episode_row(episode_id, release):
    record = {
        'run_id': f'00000000-0000-4000-8000-{episode_id:012d}',
        'case_id': release['case_id'], 'stage': release['stage'], 'experiment_id': release['experiment_id'],
        'execution_status': 'COMPLETED', 'validation': {'decision': 'APPROVE'},
        'evidence': release['manifest']['evidence'],
        'sources_consulted': [{key: source[key] for key in ('source_id', 'canonical_key', 'version', 'sha256')}
                              for source in release['sources']],
    }
    return {
        'episode_id': episode_id, 'run_id': record['run_id'], 'case_id': record['case_id'],
        'stage': record['stage'], 'experiment_id': record['experiment_id'],
        'release_id': release['release_id'], 'status': 'COMPLETED', 'record': record,
        'record_sha256': hashlib.sha256(access.canonical_bytes(record)).hexdigest(),
    }


class ExpectedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.release = release_row(CASE, 4)
        self.origin = release_row(ORIGIN, 1)
        self.origin_episode = episode_row(1, self.origin)
        self.candidates = []
        self.decisions = []
        self.enterContext(mock.patch.dict(os.environ, {
            'OPENAI_BASE_URL': 'http://127.0.0.1:8000/v1', 'OPENAI_API_KEY': 'stub',
        }))
        self.rows = self.enterContext(mock.patch.object(access, 'rows', side_effect=self.database_rows))
        self.memory_rows = self.enterContext(mock.patch.object(retrieval, 'rows', side_effect=self.database_rows))
        self.retrieve = self.enterContext(mock.patch.object(
            runner, 'retrieve_procedural_memories', return_value=self.candidates))
        self.review = self.enterContext(mock.patch.object(
            runner, 'call_memory_applicability', return_value={'decisions': self.decisions, 'conflicts': []}))
        self.enterContext(mock.patch.object(
            retrieval, 'embed', side_effect=AssertionError('Unexpected embedding call')))
        worker_result = completed_worker_result()
        self.dispatch = self.enterContext(mock.patch.object(
            dispatch.subprocess, 'run', return_value=mock.Mock(
                returncode=0, stdout='AGENT_RESULT_JSON:' + json.dumps(worker_result))))
        self.save = self.enterContext(mock.patch.object(runner, 'save_episode', return_value=1))
        self.retry_embedding = self.enterContext(mock.patch.object(
            runner, 'retry_episode_embedding', return_value=(1, True)))
        self.persistence = mock.Mock()
        self.persistence.attach_mock(self.save, 'save_episode')
        self.persistence.attach_mock(self.retry_embedding, 'retry_episode_embedding')

    def database_rows(self, sql):
        if 'FROM scientific.evidence_releases er' in sql:
            release = self.origin if access.sql_text(ORIGIN) in sql else self.release
            return [copy.deepcopy(release)]
        if 'FROM agent_memory.episodes e' in sql:
            return [copy.deepcopy(self.origin_episode)]
        if 'FROM agent_memory.memories m' in sql:
            return [copy.deepcopy(item) for item in self.candidates
                    if f"m.memory_id={item['memory_id']} AND v.version_id={item['version_id']}" in sql]
        raise AssertionError('Unexpected database query')

    def add_candidate(self, decision='APPLY', memory_id=5, version_id=7):
        content = f'Frozen procedural guidance {memory_id}'
        source_path = 'interpretation.json#/caid'
        evidence = {source['snapshot_path'].removeprefix('evidence/'): source['document']
                    for source in self.release['sources']}
        context = retrieval.retrieval_context(dict(self.release, evidence=evidence))
        candidate = {
            'memory_id': memory_id, 'version_id': version_id, 'version_number': 1,
            'title': f'Procedure {memory_id}', 'content': content,
            'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'source_episode_id': 1, 'source_case_id': ORIGIN, 'source_stage': 0,
            'source_release_id': self.origin['release_id'],
            'source_record_sha256': self.origin_episode['record_sha256'],
            'original_episode_id': 1, 'change_reason': 'Original procedure',
            'origin_evidence': [{key: source[key] for key in ('canonical_key', 'sha256')}
                                for source in self.origin['sources']],
            'similarity': 0.9, 'score': 0.9,
            'retrieval_provenance': dict(context, rank=len(self.candidates) + 1),
        }
        self.candidates.append(candidate)
        self.decisions.append({
            'memory_id': memory_id, 'version_id': version_id, 'decision': decision,
            'reason': f'{decision} after review', 'source_field_references': [source_path],
        })
        return candidate

    def pin(self, candidate):
        return ['--expected-memory-id', str(candidate['memory_id']),
                '--expected-version-id', str(candidate['version_id']),
                '--expected-content-sha256', candidate['content_sha256']]

    def run_cli(self, *flags):
        argv = ['runner', 'run', '--case', CASE, '--stage', '0', '--model', 'stub', *flags]
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(runner.sys, 'argv', argv), \
             contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return runner.main()

    def assert_rejected(self, message, *flags):
        self.assertEqual(self.run_cli(*flags), 1)
        self.assertIn(message, self.stderr.getvalue())
        self.dispatch.assert_not_called()
        self.save.assert_not_called()
        self.retry_embedding.assert_not_called()

    def dispatched_payload(self):
        self.dispatch.assert_called_once()
        self.save.assert_called_once()
        record = self.save.call_args.args[0]
        self.assertEqual(self.persistence.mock_calls, [
            mock.call.save_episode(record),
            mock.call.retry_episode_embedding(record['run_id']),
        ])
        self.assertEqual(self.dispatch.call_args.args[0][:2], ['docker', 'run'])
        payload = json.loads(self.dispatch.call_args.kwargs['input'])
        self.assertEqual(payload['evidence'], {'interpretation.json': {'caid': CASE},
                                               'guideline_v2.json': {'version': '2'}})
        return payload

    def test_correct_pin_matches_the_single_selected_memory(self):
        self.add_candidate('SKIP')
        selected = self.add_candidate(memory_id=6, version_id=8)
        self.assertEqual(self.run_cli('--use-multi-memory', *self.pin(selected)), 0, self.stderr.getvalue())
        self.retrieve.assert_called_once()
        self.assertEqual(self.dispatched_payload()['procedural_memories'],
                         [{'title': selected['title'], 'content': selected['content']}])
        marker = self.save.call_args.args[0]['procedural_memory_supplied']
        for key in ('memory_id', 'version_id', 'content_sha256'):
            self.assertEqual(marker[key], selected[key])

    def test_mismatched_id_version_or_hash_stops_before_dispatch(self):
        candidate = self.add_candidate()
        for field, value in (('memory_id', 99), ('version_id', 99), ('content_sha256', '0' * 64)):
            with self.subTest(field=field):
                mismatched = dict(candidate, **{field: value})
                self.assert_rejected('does not match the expected ID, version or content hash',
                                     '--use-multi-memory', *self.pin(mismatched))

    def test_pin_rejects_zero_selected_memories(self):
        skipped = self.add_candidate('SKIP')
        self.assert_rejected('requires exactly one selected memory', '--use-multi-memory', *self.pin(skipped))

    def test_pin_rejects_multiple_selected_memories(self):
        candidate = self.add_candidate()
        self.add_candidate(memory_id=6, version_id=8)
        self.assert_rejected('requires exactly one selected memory', '--use-multi-memory', *self.pin(candidate))

    def test_partial_pins_and_pins_outside_execution_stop_before_retrieval(self):
        pin = self.pin(self.add_candidate())
        invalid = [pin, ['--preview-multi-memory', *pin]]
        invalid.extend(['--use-multi-memory', *pin[:i], *pin[i + 2:]] for i in (0, 2, 4))
        for flags in invalid:
            with self.subTest(flags=flags):
                self.assert_rejected('must be supplied together with --use-multi-memory', *flags)
        self.retrieve.assert_not_called()
        self.review.assert_not_called()

    def test_malformed_pin_stops_before_retrieval(self):
        candidate = self.add_candidate()
        for field, value in (('memory_id', 0), ('version_id', -1), ('content_sha256', 'not-a-hash')):
            with self.subTest(field=field):
                self.assert_rejected('Expected memory IDs must be positive', '--use-multi-memory',
                                     *self.pin(dict(candidate, **{field: value})))
        self.retrieve.assert_not_called()

    def test_retired_flag_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli('--use-procedural-memory')
        self.assertEqual(error.exception.code, 2)
        self.assertIn('unrecognized arguments', self.stderr.getvalue())
        self.retrieve.assert_not_called()
        self.dispatch.assert_not_called()
        self.save.assert_not_called()

    def test_off_skips_memory_retrieval_and_review(self):
        self.assertEqual(self.run_cli(), 0, self.stderr.getvalue())
        self.retrieve.assert_not_called()
        self.review.assert_not_called()
        self.assertNotIn('procedural_memories', self.dispatched_payload())
        self.assertNotIn('procedural_multi_memory_selection', self.save.call_args.args[0])

    def test_save_failure_skips_embedding(self):
        self.save.side_effect = RuntimeError('PostgreSQL unavailable')
        self.assertEqual(self.run_cli(), 1)
        self.dispatch.assert_called_once()
        self.save.assert_called_once()
        self.retry_embedding.assert_not_called()
        self.assertIn('Episode save unconfirmed', self.stderr.getvalue())
        self.assertNotIn('RESULT: RUN STORED', self.stdout.getvalue())

    def test_embedding_failure_keeps_saved_episode_successful(self):
        self.retry_embedding.side_effect = RuntimeError('Ollama unavailable')
        self.assertEqual(self.run_cli(), 0, self.stderr.getvalue())
        self.dispatched_payload()
        run_id = self.save.call_args.args[0]['run_id']
        self.assertIn(f'RESULT: RUN STORED | episode_id=1 | run_id={run_id}', self.stdout.getvalue())
        self.assertIn(f'RESULT: EMBEDDING PENDING | run_id={run_id}', self.stderr.getvalue())

    def test_unpinned_zero_selection_preserves_audit_without_guidance(self):
        self.add_candidate('SKIP')
        self.assertEqual(self.run_cli('--use-multi-memory'), 0, self.stderr.getvalue())
        self.assertNotIn('procedural_memories', self.dispatched_payload())
        record = self.save.call_args.args[0]
        self.assertNotIn('procedural_memory_supplied', record)
        self.assertEqual(record['procedural_multi_memory_selection']['selected_memories'], [])

    def test_unpinned_three_memories_preserve_order(self):
        for i in range(3):
            self.add_candidate(memory_id=5 + i, version_id=7 + i)
        self.assertEqual(self.run_cli('--use-multi-memory'), 0, self.stderr.getvalue())
        self.assertEqual(self.dispatched_payload()['procedural_memories'],
                         [{key: item[key] for key in ('title', 'content')} for item in self.candidates])

    def test_preview_reports_candidate_pins_without_dispatch_or_save(self):
        selected = self.add_candidate()
        skipped = self.add_candidate('SKIP', memory_id=6, version_id=8)
        self.assertEqual(self.run_cli('--preview-multi-memory'), 0, self.stderr.getvalue())
        self.assertEqual(json.loads(self.stdout.getvalue())['decisions'], [
            {'memory_id': 5, 'version_id': 7, 'decision': 'APPLY', 'selected': True,
             'content_sha256': selected['content_sha256'], 'reason': 'APPLY after review'},
            {'memory_id': 6, 'version_id': 8, 'decision': 'SKIP', 'selected': False,
             'content_sha256': skipped['content_sha256'], 'reason': 'SKIP after review'},
        ])
        self.dispatch.assert_not_called()
        self.save.assert_not_called()

    def test_stale_memory_at_dispatch_is_rejected_even_with_correct_pin(self):
        candidate = self.add_candidate()
        active_versions = iter(([candidate], []))
        self.memory_rows.side_effect = lambda sql: (
            next(active_versions) if 'FROM agent_memory.memories m' in sql else self.database_rows(sql))
        self.assert_rejected('Selected memories are stale or ineligible', '--use-multi-memory', *self.pin(candidate))

    def test_unapproved_release_stops_before_memory_review(self):
        self.release['receipt']['status'] = 'pending'
        self.assert_rejected('Receipt has not approved this investigation', '--use-multi-memory')
        self.retrieve.assert_not_called()
        self.review.assert_not_called()

    def test_incomplete_worker_success_does_not_save_episode(self):
        incomplete = dict(completed_worker_result(), trace=[], report=None)
        self.dispatch.return_value.stdout = 'AGENT_RESULT_JSON:' + json.dumps(incomplete)
        self.assertEqual(self.run_cli(), 1)
        self.assertIn('Incomplete successful investigation result', self.stderr.getvalue())
        self.save.assert_not_called()
        self.retry_embedding.assert_not_called()

    def test_altered_evidence_is_rejected_at_dispatch(self):
        supplied = runner.approved_release(CASE, 0)
        supplied['evidence']['interpretation.json']['caid'] = 'CA999'
        with self.assertRaisesRegex(ValueError, 'does not match the approved case release'):
            runner.run_agent(supplied, 'stub')
        self.dispatch.assert_not_called()

    def test_external_endpoint_requires_explicit_authorization(self):
        self.add_candidate()
        with mock.patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://example.invalid/v1'}):
            self.assert_rejected('External agent endpoint', '--use-multi-memory')
            self.rows.assert_not_called()
            self.review.assert_not_called()
            self.assertEqual(self.run_cli('--use-multi-memory', '--allow-external-model'),
                             0, self.stderr.getvalue())
        self.assertIs(self.review.call_args.kwargs['allow_external_model'], True)
        self.dispatched_payload()


class EpisodePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.release = release_row(CASE, 4)
        self.ep = episode_row(8, self.release)
        self.record = self.ep["record"]
        self.record.update(trace_level=episodes.TRACE_LEVEL, trace=[], known_uncertainties=[],
                           started_at_utc="2026-01-01T00:00:00Z", completed_at_utc="2026-01-01T00:00:01Z")
        self.ep.update(run_record=self.record, trace_level=episodes.TRACE_LEVEL, has_embedding=False,
                       embedding_model=None, record_sha256=hashlib.sha256(access.canonical_bytes(self.record)).hexdigest())
        self.enterContext(mock.patch.object(access, 'rows', return_value=[self.release]))
        self.db = self.enterContext(mock.patch.object(episodes, 'db'))
        self.rows = self.enterContext(mock.patch.object(episodes, 'rows', return_value=[self.ep]))
        self.embed = self.enterContext(mock.patch.object(episodes, 'embed', return_value=[1.0] * 768))

    def test_save_reconciles_lost_ack_by_run_id_before_retrying(self):
        self.db.side_effect = [RuntimeError('Lost acknowledgement'), ['8']]
        self.assertEqual(episodes.save_episode(self.record), 8)
        insert, readback = [call.args[0] for call in self.db.call_args_list]
        self.assertIn('NULL, NULL', insert)
        self.assertIn('ON CONFLICT (run_id) DO NOTHING', insert)
        self.assertIn(access.sql_text(self.ep['record_sha256']), readback)
        self.embed.assert_not_called()

    def test_retry_writes_only_embedding_and_skips_existing(self):
        self.db.return_value = ['8']
        self.assertEqual(episodes.retry_episode_embedding(self.ep['run_id']), (8, True))
        sql = self.db.call_args.args[0]
        self.assertIn('e.embedding_model IS NULL AND e.episode_embedding IS NULL', sql)
        self.assertNotIn('SET run_record', sql)
        self.ep.update(embedding_model=episodes.EMBEDDING_MODEL, has_embedding=True)
        self.assertEqual(episodes.retry_episode_embedding(self.ep['run_id']), (8, False))
        self.embed.assert_called_once()
        self.db.assert_called_once()

    def test_verify_and_replay_check_the_same_scoped_run_without_embedding(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            episodes.verify(CASE, 0, self.ep['run_id'])
            episodes.replay(CASE, 0, self.ep['run_id'])
        self.assertIn('VERIFY OK', output.getvalue())
        self.assertIn('REPLAY OK', output.getvalue())
        self.assertIn(self.ep['record_sha256'], output.getvalue())
        self.rows.return_value = []
        with self.assertRaisesRegex(ValueError, 'No approved episode'):
            episodes.replay(CASE, 0, self.ep['run_id'])
        self.embed.assert_not_called()


if __name__ == '__main__':
    unittest.main()
