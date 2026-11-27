from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import verify_evidence_release as verifier
import ingest_approved_evidence as ingest
import evidence_access as access
# - Stub worker dependencies so these tests cannot execute a graph or model.
import types
first = types.ModuleType('first_investigation')
first.InvestigationState = dict
first.investigate = first.reflect = first.validate = lambda state: state
graph = types.ModuleType('langgraph.graph')
graph.END = 'END'; graph.START = 'START'; graph.StateGraph = object
chat = types.ModuleType('langchain_openai'); chat.ChatOpenAI = object
pydantic = types.ModuleType('pydantic')
pydantic.BaseModel = type('BaseModel', (), {})
pydantic.StrictBool = bool; pydantic.StrictInt = int
with mock.patch.dict(sys.modules, {'first_investigation': first, 'langgraph.graph': graph,
                                  'langchain_openai': chat, 'pydantic': pydantic}):
    import agent_langgraph_worker as worker


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.active = self.root / 'scientific_data/active/variants/CA274982'
        self.exp = self.root / 'experiments/test'
        self.active.mkdir(parents=True)
        (self.exp / 'evidence').mkdir(parents=True)
        self.page_hash = 'a' * 64
        self.url = 'https://example.org/exact?version=1.0.0'
        self.interp = {'caid': 'CA274982', 'interpretation_uuid': '8b322e84-d5bb-4038-bce4-875136c7e6cd',
                       'version': '1.0.0', 'classification': 'x', 'narrative_scope': 'partial',
                       'narrative_provenance': {'source_url': self.url, 'source_sha256': self.page_hash},
                       'guideline_assignment_provenance': {}, 'source_sepio_sha256': 'b' * 64,
                       'source_expert_page_sha256': self.page_hash, 'criteria': [], 'guideline_version': '2'}
        self.guideline = {'guideline_id': 'GN010', 'gene': 'GAA', 'version': '2.0'}
        self.rationale = {'case_id': 'CA274982', 'source_version': '1.0.0', 'source_url': self.url,
                          'status': 'REVIEWED_FOR_POC', 'source_provenance': {'source_sha256': self.page_hash},
                          'review': {'reviewer': 'operator', 'method': 'exact source review',
                                     'exact_excerpts_verified': True},
                          'independent_clinical_review': False, 'historical_availability_verified': False,
                          'excerpts': [{'criterion': 'PM5', 'source_text': 'Exact text (PM5 not met).'}],
                          'limitations': ['No independent clinical review or established availability.']}
        self.entries = []
        self.write_doc('interpretation.json', self.interp)
        self.write_doc('guideline_v2.json', self.guideline)
        self.write_doc('criterion_rationale.json', self.rationale)
        self.finish()
        patch = mock.patch.object(verifier, 'PROJECT_ROOT', self.root)
        patch.start(); self.addCleanup(patch.stop)

    def write_doc(self, name, document):
        data = json.dumps(document).encode()
        (self.active / name).write_bytes(data)
        (self.exp / 'evidence' / name).write_bytes(data)
        self.entries = [e for e in self.entries if e['file'] != 'evidence/' + name]
        self.entries.append({'file': 'evidence/' + name, 'source': 'active/variants/CA274982/' + name,
                             'sha256': digest(data)})

    def finish(self, entries=None, status='approved_for_first_investigation'):
        entries = copy.deepcopy(self.entries if entries is None else entries)
        inv = {'caid': 'CA274982', 'policy': 'case-scoped', 'files': [
            {'path': x['file'].removeprefix('evidence/'), 'sha256': x['sha256']} for x in entries]}
        (self.active / 'case_inventory.json').write_text(json.dumps(inv))
        manifest = {'experiment_id': 'test', 'case_id': 'CA274982', 'stage': 0, 'evidence': entries}
        receipt = dict(manifest, status=status, released_at_utc='fixture', agent_execution_verified=False)
        manifest_path = self.exp / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        receipt['owner_authorization'] = {
            'authority': 'project owner', 'statement': 'Fixture authorization for controlled POC.',
            'scope': 'Agent Memory & Self-Learning Lab controlled POC only',
            'independent_clinical_review': False, 'manifest_sha256': digest(manifest_path.read_bytes()),
        }
        (self.exp / 'release_receipt.json').write_text(json.dumps(receipt))

    def verify(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verifier.verify(str(self.exp))

    def test_valid_and_ingestion_sql(self):
        self.verify()
        with mock.patch.object(ingest, 'PROJECT_ROOT', self.root):
            sql = ingest.build_sql(self.exp)
        self.assertIn(ingest.sql_text('CRITERION_RATIONALE'), sql)
        self.assertIn(ingest.sql_text('evidence/criterion_rationale.json'), sql)

    def test_reject_bad_content(self):
        cases = [lambda x: x.update(status='DRAFT_UNAPPROVED'), lambda x: x.update(case_id='CA999'),
                 lambda x: x.update(source_version='2.0.0'), lambda x: x.update(source_url=self.url + 'x'),
                 lambda x: x.pop('source_provenance'), lambda x: x.pop('review'),
                 lambda x: x['review'].update(exact_excerpts_verified=False),
                 lambda x: x.update(limitations=[]), lambda x: x['excerpts'][0].update(source_text='  '),
                 lambda x: x['excerpts'][0].update(criterion=''),
                 lambda x: x['source_provenance'].update(source_sha256='f' * 64)]
        for mutate in cases:
            with self.subTest(mutate=mutate):
                doc = copy.deepcopy(self.rationale); mutate(doc)
                self.write_doc('criterion_rationale.json', doc); self.finish()
                with self.assertRaises(ValueError): self.verify()

    def test_owner_authorization_required(self):
        path = self.exp / 'release_receipt.json'
        original = json.loads(path.read_text())
        bad = [None, [], {},
               dict(original['owner_authorization'], authority='another person'),
               dict(original['owner_authorization'], authority='not project owner'),
               dict(original['owner_authorization'], statement='  '),
               dict(original['owner_authorization'], scope='unrelated POC'),
               dict(original['owner_authorization'], scope='not Agent Memory & Self-Learning Lab controlled POC'),
               dict(original['owner_authorization'], manifest_sha256='f' * 64),
               dict(original['owner_authorization'], manifest_sha256='invalid'),
               dict(original['owner_authorization'], independent_clinical_review=True)]
        for field in ('authority', 'statement', 'scope', 'manifest_sha256', 'independent_clinical_review'):
            missing = dict(original['owner_authorization']); missing.pop(field); bad.append(missing)
        for value in bad:
            with self.subTest(authorization=value):
                receipt = dict(original, owner_authorization=value)
                path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError): self.verify()
        receipt = dict(original); receipt.pop('owner_authorization')
        path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError): self.verify()

    def test_modified_hash(self):
        (self.exp / 'evidence/criterion_rationale.json').write_text('{}')
        with self.assertRaises(ValueError): self.verify()

    def test_unlisted_source(self):
        inventory_path = self.active / 'case_inventory.json'
        inventory = json.loads(inventory_path.read_text())
        inventory['files'] = inventory['files'][:-1]
        inventory_path.write_text(json.dumps(inventory))
        with self.assertRaises(ValueError): self.verify()

    def test_receipt_mismatch_and_stage(self):
        receipt_path = self.exp / 'release_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['evidence'] = receipt['evidence'][:-1]
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError): self.verify()
        self.finish()
        manifest = json.loads((self.exp / 'manifest.json').read_text())
        receipt = json.loads(receipt_path.read_text())
        manifest['stage'] = receipt['stage'] = 1
        (self.exp / 'manifest.json').write_text(json.dumps(manifest))
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError): self.verify()

    def test_duplicate(self):
        self.finish(entries=self.entries + [self.entries[-1]])
        with self.assertRaises(ValueError): self.verify()

    def test_authorization_and_unexpected_path(self):
        self.finish(status='DRAFT_UNAPPROVED')
        with self.assertRaises(ValueError): self.verify()
        self.finish(); (self.exp / 'evidence/unexpected.json').write_text('{}')
        with self.assertRaises(ValueError): self.verify()

    def test_runner_source_type_and_evidence_separation(self):
        manifest = json.loads((self.exp / 'manifest.json').read_text())
        sources = []
        types = ['INTERPRETATION', 'GUIDELINE', 'CRITERION_RATIONALE']
        for entry, typ, doc in zip(self.entries, types, (self.interp, self.guideline, self.rationale)):
            sources.append({'snapshot_path': entry['file'], 'sha256': entry['sha256'],
                            'source_type': typ, 'document': doc, 'declared_sha256': entry['sha256']})
        release = {'experiment_id': 'test', 'manifest': manifest, 'receipt': dict(manifest, status=access.APPROVED),
                   'sources': sources}
        with mock.patch.object(access, 'rows', return_value=[release]):
            result = access.approved_release('CA274982', 0)
        self.assertEqual(result['evidence']['criterion_rationale.json'], self.rationale)
        sources[-1]['source_type'] = 'INTERPRETATION'
        with mock.patch.object(access, 'rows', return_value=[release]), self.assertRaises(ValueError):
            access.approved_release('CA274982', 0)

        sources[-1]['source_type'] = 'CRITERION_RATIONALE'
        release['receipt']['evidence'] = []
        with mock.patch.object(access, 'rows', return_value=[release]), self.assertRaisesRegex(ValueError, 'receipt'):
            access.approved_release('CA274982', 0)
        release['receipt']['evidence'] = manifest['evidence']
        sources[-1]['declared_sha256'] = '0' * 64
        with mock.patch.object(access, 'rows', return_value=[release]), self.assertRaisesRegex(ValueError, 'link hash'):
            access.approved_release('CA274982', 0)

    def test_ingestion_uses_the_verified_bytes_without_reopening(self):
        snapshot = self.exp / 'evidence' / 'criterion_rationale.json'
        original_read = Path.read_bytes
        reads = []

        def read(path):
            if path == snapshot:
                reads.append(path)
                if len(reads) > 1:
                    return b'{"unapproved":true}'
            return original_read(path)

        with mock.patch.object(ingest, 'PROJECT_ROOT', self.root), mock.patch.object(Path, 'read_bytes', read):
            sql = ingest.build_sql(self.exp)
        self.assertEqual(reads, [snapshot])
        self.assertIn(access.sql_json(self.rationale), sql)

    def test_worker_exact_path(self):
        evidence = {'interpretation.json': {}, 'guideline_v2.json': {}, 'criterion_rationale.json': self.rationale}
        def invoke(e):
            with mock.patch.object(worker.sys, 'stdin', io.TextIOWrapper(io.BytesIO(json.dumps({'evidence': e}).encode()))), \
                 mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'stub', 'OPENAI_MODEL': 'stub'}), \
                 mock.patch.object(worker, 'execute', return_value={}) as execute, contextlib.redirect_stdout(io.StringIO()):
                worker.main()
                return execute.call_args
        self.assertEqual(invoke(evidence).args[0], evidence)
        for name in ('x/criterion_rationale.json', '../criterion_rationale.json', '/criterion_rationale.json'):
            with self.subTest(name=name), self.assertRaises(ValueError): invoke(dict(evidence, **{name: {}}))


class ExistingReleaseTests(unittest.TestCase):
    def test_existing_releases(self):
        project = Path(os.environ.get('TASK14_TEST_PROJECT_ROOT', ROOT))
        cases = ('CA1585329339', 'CA8815234', 'CA401366477')
        inventories = [
            project / 'scientific_data' / 'active' / 'variants' / case / 'case_inventory.json'
            for case in cases
        ]
        if not all(path.is_file() for path in inventories):
            self.skipTest('Historical verification requires private case inventories')
        for name in ('EXP-001_CA1585329339_stage0', 'EXP-002_CA8815234_stage0', 'EXP-003_CA401366477_stage0'):
            with self.subTest(name=name), mock.patch.object(verifier, 'PROJECT_ROOT', project), contextlib.redirect_stdout(io.StringIO()):
                verifier.verify(str(project / 'experiments' / name))


if __name__ == '__main__': unittest.main()
