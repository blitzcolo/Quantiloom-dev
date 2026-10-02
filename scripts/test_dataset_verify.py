"""Independent malformed-record regressions; run with Python unittest."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import dataset_verify


class DatasetVerifyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'frame.metadata.json'
        replay, data = b'# config\n', b'sample'
        (self.root / 'frame.replay.toml').write_bytes(replay)
        (self.root / 'frame.bin').write_bytes(data)
        self.record = {
            'schema': 'quantiloom.dataset.export', 'schema_version': 1,
            'record_id': '1' * 64, 'state': 'complete',
            'capture_status': 'complete', 'pairing_status': 'not_requested',
            'provenance': {},
            'replay': {'path': 'frame.replay.toml', 'sha256': hashlib.sha256(replay).hexdigest()},
            'products': [{'product_id': 'frame', 'path': 'frame.bin',
                'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data), 'description': {}}],
        }

    def verify(self, record):
        self.path.write_text(json.dumps(record), encoding='utf-8')
        return dataset_verify.verify(self.path)

    def test_valid_manifest(self):
        self.assertTrue(self.verify(self.record)['valid'])

    def test_required_root_fields(self):
        for key in self.record:
            with self.subTest(key=key):
                record = copy.deepcopy(self.record)
                del record[key]
                with self.assertRaises((ValueError, KeyError)):
                    self.verify(record)

    def test_required_product_fields(self):
        for key in self.record['products'][0]:
            with self.subTest(key=key):
                record = copy.deepcopy(self.record)
                del record['products'][0][key]
                with self.assertRaises((ValueError, KeyError)):
                    self.verify(record)

    def test_unknown_root_fields(self):
        self.record['unexpected'] = 1
        with self.assertRaises(ValueError): self.verify(self.record)

    def test_claim_is_not_verification(self):
        self.record['provenance']['reproducibility'] = {'verified': True}
        result = self.verify(self.record)
        self.assertTrue(result['valid'])
        self.assertFalse(result['reproducibility_verified'])

    def test_duplicate_json_keys(self):
        text = json.dumps(self.record)
        self.path.write_text('{"state":"failed",' + text[1:], encoding='utf-8')
        with self.assertRaises(ValueError): dataset_verify.verify(self.path)

    def test_numeric_overflow_is_not_finite_json(self):
        text = json.dumps(self.record).replace('"provenance": {}', '"provenance": {"value": 1e999}')
        self.path.write_text(text, encoding='utf-8')
        with self.assertRaises(ValueError): dataset_verify.verify(self.path)

    def test_reserved_internal_path(self):
        self.record['products'][0]['path'] = '.INTERNAL/frame.bin'
        with self.assertRaises(ValueError): self.verify(self.record)

    def test_reserved_per_artifact_claim_path(self):
        self.record['products'][0]['path'] = 'other.EXR.QUANTILOOM-EXPORT.LOCK/frame.bin'
        with self.assertRaises(ValueError): self.verify(self.record)


if __name__ == '__main__':
    unittest.main()
