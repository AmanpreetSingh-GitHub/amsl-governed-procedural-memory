"""Operator governance, provenance, and transaction tests with stubbed services."""

import base64
import contextlib
import hashlib
import io
import json
import re
import unittest
from unittest import mock

from test_expected_memory import CASE, episode_row, release_row
import evidence_access as access
import govern_procedural_lessons as govern


class PromotionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.release = release_row(CASE, 4)
        self.release['sources'].sort(key=lambda source: source['snapshot_path'])
        self.ep = episode_row(8, self.release)
        self.ep['record'].update(report='Approved evidence review', reflection=(
            'DECISION: PROPOSE\n### Candidate lesson\nCheck each claim.\n'
            '### Conditions for reuse\nUse an approved release.\n### Limitations\nSome facts remain uncertain.'
        ))
        self.ep['record_sha256'] = hashlib.sha256(access.canonical_bytes(self.ep['record'])).hexdigest()
        self.metadata = dict(episode_id=8, decision='REFINE', run_id=self.ep['run_id'],
                             record_sha256=self.ep['record_sha256'], target_memory_id=2, active_version_id=11)
        self.draft = {'version_id': 12, 'n': 2, 'title': 'Procedure', 'content': 'Record the exact evidence',
                      'change_reason': 'DRAFT_LEARNING:8:REFINE:' + json.dumps(self.metadata)}
        self.memory = {'memory_id': 2, 'episode_id': 8, 'status': 'VALIDATED', 'active_version_id': 11,
                       'versions': [{'version_id': 11, 'n': 1, 'title': 'Procedure', 'content': 'Record evidence',
                                     'change_reason': 'Operator candidate'}, self.draft]}
        self.enterContext(mock.patch.object(govern, 'record', return_value=self.memory))
        self.episodes = self.enterContext(mock.patch.object(govern, 'load_episode', return_value=self.ep))
        self.releases = self.enterContext(mock.patch.object(access, 'rows', return_value=[self.release]))
        self.write = self.enterContext(mock.patch.object(govern, 'db', return_value=['{"audit_id":42}']))
        self.embed = self.enterContext(mock.patch.object(govern, 'embed', return_value=[1.0] * 768))
        self.output = self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def promote(self, version=2):
        govern.approve(2, version, 'operator', 'Reviewed reproducibility', 'PROMOTE')

    def sql(self):
        return self.write.call_args.args[0]

    def audit_notes(self, field):
        literals = [base64.b64decode(value).decode('utf8')
                    for value in re.findall(r"decode\('([^']+)', 'base64'\)", self.sql())]
        return next(json.loads(value) for value in literals if value.startswith('{') and field in json.loads(value))

    def test_promotion_uses_episode_and_sources_without_second_model_review(self):
        self.promote()
        self.assertIn('RESULT: PROMOTE', self.output.getvalue())
        self.assertIn('WHERE memory_id=2 FOR UPDATE', self.sql())
        self.assertIn(f'e.record_sha256={access.sql_text(self.ep["record_sha256"])}', self.sql())
        self.assertIn('e.run_record::text=', self.sql())
        self.assertIn(f'r.release_status={access.sql_text(access.APPROVED)}', self.sql())
        self.assertIn('re.declared_sha256', self.sql())
        self.assertIn('FROM changed', self.sql())
        self.assertNotIn('CONSISTENCY_CHECK', self.sql())
        self.assertEqual(self.audit_notes('new_active_version_id')['previous_active_version_id'], 11)

    def test_promotion_rejects_revoked_release(self):
        self.releases.return_value = []
        with self.assertRaisesRegex(ValueError, 'approved case/stage release'):
            self.promote()
        self.write.assert_not_called()

    def test_promotion_rejects_changed_source_provenance(self):
        self.release['sources'][0]['canonical_key'] = 'CHANGED'
        with self.assertRaisesRegex(ValueError, 'source provenance mismatch'):
            self.promote()
        self.write.assert_not_called()

    def test_promotion_rejects_invalid_episode_hash(self):
        self.ep['record_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.promote()
        self.releases.assert_not_called()
        self.write.assert_not_called()

    def test_transaction_guard_no_match_cannot_report_promotion(self):
        self.write.return_value = []
        with self.assertRaisesRegex(ValueError, 'did not return a result'):
            self.promote()
        self.assertNotIn('RESULT: PROMOTE', self.output.getvalue())

    def test_stale_refine_cannot_activate(self):
        self.memory['active_version_id'] = 13
        with self.assertRaisesRegex(ValueError, 'base version is stale'):
            self.promote()
        self.write.assert_not_called()

    def test_operator_candidate_promotes_with_original_episode(self):
        self.memory.update(status='CANDIDATE', active_version_id=None)
        self.promote(1)
        self.assertIn('e.episode_id=8', self.sql())
        self.assertIsNone(self.audit_notes('new_active_version_id')['previous_active_version_id'])

    def test_rollback_checks_prior_operator_promotion_and_current_pointer(self):
        govern.approve(2, 2, 'operator', 'Use previously promoted version', 'ROLLBACK')
        sql = self.sql()
        self.assertIn('WHERE memory_id=2 FOR UPDATE', sql)
        self.assertIn('m.active_version_id IS NOT DISTINCT FROM 11', sql)
        self.assertIn("rationale LIKE 'PROMOTE:%'", sql)
        self.assertEqual(self.audit_notes('new_active_version_id')['new_active_version_id'], 12)
        self.episodes.assert_not_called()

    def test_rollback_no_match_cannot_report_success(self):
        self.write.return_value = []
        with self.assertRaisesRegex(ValueError, 'did not return a result'):
            govern.approve(2, 2, 'operator', 'Use previously promoted version', 'ROLLBACK')
        self.assertNotIn('RESULT: ROLLBACK', self.output.getvalue())

    def test_retire_chooses_active_or_latest_version_under_lock(self):
        govern.retire(2, 'operator', 'Retire the procedure')
        locks, sql = self.sql().split('WITH target AS (', 1)
        self.assertIn('WHERE memory_id=2 FOR UPDATE', locks)
        self.assertIn('COALESCE(m.active_version_id,', sql)
        self.assertIn('ORDER BY v.version_number DESC LIMIT 1', sql)
        self.assertIn("SELECT version_id,'OPERATOR_DECISION','FAIL'", sql)

    def test_new_and_refine_create_inactive_embedded_versions(self):
        for decision in ('NEW', 'REFINE'):
            with self.subTest(decision=decision):
                recommendation = dict(decision=decision, memory_id=2 if decision == 'REFINE' else None,
                                      rationale='Observed', supporting_references=['interpretation.json'],
                                      limitations=[], unresolved_contradictions=[], proposed_revised_lesson='Refined method')
                matches = [dict(memory_id=2, version_id=11, title='Procedure')]
                self.write.return_value = ['{"memory_id":2,"version_id":13,"version_number":3}']
                with mock.patch.object(govern, 'assess', return_value=(recommendation, self.ep, matches)):
                    govern.draft_learning(8)
                self.assertIn('INSERT INTO agent_memory.embeddings', self.sql())
                self.assertIn('INSERT INTO agent_memory.memory_versions', self.sql())
                self.assertNotIn('UPDATE agent_memory.memories', self.sql())
                if decision == 'REFINE':
                    self.assertIn('m.active_version_id=11', self.sql())
                    self.assertNotIn('INSERT INTO agent_memory.memories', self.sql())

    def test_reinforce_records_observation_with_exact_version_and_injection(self):
        marker = {'memory_id': 99, 'version_id': 4}
        self.ep['record']['procedural_memory_supplied'] = marker
        self.ep['record_sha256'] = hashlib.sha256(access.canonical_bytes(self.ep['record'])).hexdigest()
        govern.reinforce(8, 2, 'operator', 'Observed reuse')
        self.assertIn('m.active_version_id=11', self.sql())
        self.assertIn('NOT EXISTS', self.sql())
        self.assertNotIn('UPDATE agent_memory.memories', self.sql())
        notes = self.audit_notes('memory_injection')
        self.assertEqual(notes['episode_id'], 8)
        self.assertEqual(notes['memory_injection'], marker)
        self.assertEqual(notes['scope'], 'Operator observation only')

    def test_rejects_promotion_of_unattributed_manual_revision(self):
        self.draft['change_reason'] = 'Historical manual revision'
        with self.assertRaisesRegex(ValueError, 'no verifiable episode'):
            self.promote()
        self.write.assert_not_called()
