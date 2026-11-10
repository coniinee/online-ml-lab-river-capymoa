"""Public metadata must hide local paths without blessing altered measurements."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab_support.identity import record_digest
from scripts.redact_public_paths import ROOT, redact_file, redact_text


class PublicPathTests(unittest.TestCase):
    def test_environment_paths_are_redacted_and_hardware_text_is_retained(self):
        java = ROOT / '.cache' / 'jdk'
        text = '\n'.join([
            str(ROOT / '.venv-capy/bin/python'),
            str(ROOT / 'config/autoclass.json'),
            str(Path.home() / 'Library/Caches/pip'),
            str(java),
            'Python 3.11; Java 17; memory 32 GiB',
        ])
        with patch.dict(os.environ, {'JAVA_HOME': str(java)}):
            result = redact_text(text)
        self.assertNotIn(str(ROOT), result)
        self.assertNotIn(str(Path.home()), result)
        self.assertIn('<CAPY_ENV>/bin/python', result)
        self.assertIn('<PROJECT_ROOT>/config/autoclass.json', result)
        self.assertIn('<USER_HOME>/Library/Caches/pip', result)
        self.assertIn('<JAVA_HOME>', result)
        self.assertIn('Python 3.11; Java 17; memory 32 GiB', result)
        historical_root = Path.home() / 'earlier-project-copy'
        historical = '\n'.join([
            str(historical_root / '.venv-capy/bin/python'),
            f'AutoClass -f {historical_root / "config/autoclass.json"}',
            str(historical_root / 'data/capymoa'),
            str(historical_root / 'river_lab/evaluation.py'),
        ])
        result = redact_text(historical)
        self.assertNotIn('earlier-project-copy', result)
        self.assertIn('<CAPY_ENV>/bin/python', result)
        self.assertIn('AutoClass -f <PROJECT_ROOT>/config/autoclass.json', result)
        self.assertIn('<PROJECT_ROOT>/data/capymoa', result)
        self.assertIn('<PROJECT_ROOT>/river_lab/evaluation.py', result)

    def test_record_redaction_preserves_measurements_and_rejects_bad_seals(self):
        record = {
            'parameters': f'AutoClass -f {ROOT / "config/autoclass.json"}',
            'summary': {'accuracy': 0.85, 'mean_latency_us': 100.0},
            'curves': [{'n_seen': 100, 'accuracy': 0.85}],
            'prediction_sha256': 'unchanged', 'fingerprint': 'unchanged',
        }
        record['record_sha256'] = record_digest(record)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            path.write_text(json.dumps(record))
            self.assertTrue(redact_file(path))
            after = json.loads(path.read_text())
            for key in record.keys() - {'parameters', 'record_sha256'}:
                self.assertEqual(record[key], after[key])
            self.assertEqual(after['record_sha256'], record_digest(after))
            self.assertNotEqual(record['record_sha256'], after['record_sha256'])
            self.assertFalse(redact_file(path))
            after['summary']['accuracy'] = 0.99
            path.write_text(json.dumps(after))
            with self.assertRaises(ValueError):
                redact_file(path)


if __name__ == '__main__':
    unittest.main()
