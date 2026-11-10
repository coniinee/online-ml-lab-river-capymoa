"""Remove machine-specific paths from public metadata, never from measurements.

Run from a working copy before publication. Raw model records must already have
valid content seals; only their parameter-description text may be redacted.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def redact_text(value, root=ROOT):
    root = Path(root).resolve()
    replacements = {
        str(root / '.venv-capy'): '<CAPY_ENV>',
        str(root / '.venv'): '<RIVER_ENV>',
        str(root): '<PROJECT_ROOT>',
        str(Path.home()): '<USER_HOME>',
    }
    if sys.prefix != sys.base_prefix:
        replacements.setdefault(str(Path(sys.prefix).resolve()), '<PYTHON_ENV>')
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        replacements[str(Path(java_home).resolve())] = '<JAVA_HOME>'
    for original, placeholder in sorted(replacements.items(), key=lambda item: -len(item[0])):
        value = value.replace(original, placeholder)
    # Frozen evidence can refer to the original project before it was packaged.
    # Normalize its known project resources even after the home was anonymized.
    user_root = r'(?:<USER_HOME>|/(?:Users|home)/[^/\s\"\'<>\\]+)'
    for folder, placeholder in (('.venv-capy', '<CAPY_ENV>'), ('.venv', '<RIVER_ENV>')):
        value = re.sub(user_root + r'/[^\"\'\n<>]*?/' + re.escape(folder) + r'(?=/|$)',
                       placeholder, value)
    resources = r'(?:config/autoclass\.json|data/capymoa\b|river_lab/|capy_lab/|lab_support/)'
    value = re.sub(user_root + r'/[^\"\'\n<>]*?/(?=' + resources + r')',
                   '<PROJECT_ROOT>/', value)
    # Also cover saved records produced on another computer or in another copy.
    value = re.sub(r'/Library/Java/JavaVirtualMachines/[^\s\"\'<>]+/Contents/Home',
                   '<JAVA_HOME>', value)
    value = re.sub(r'/(?:Users|home)/[^/\s\"\'<>\\]+', '<USER_HOME>', value)
    value = re.sub(r'[A-Za-z]:\\Users\\[^\\\s\"\'<>]+', '<USER_HOME>', value)
    return value


def redact_value(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    return value


def redact_file(path):
    path = Path(path)
    before = path.read_text()
    if path.suffix == '.json':
        value = json.loads(before)
        if 'record_sha256' in value:
            from lab_support.identity import record_digest
            if value['record_sha256'] != record_digest(value):
                raise ValueError('Refusing to redact a record with an invalid content seal')
            # Preserve predictions, timings, curves, settings and runtime identity.
            parameters = value.get('parameters')
            if not isinstance(parameters, str):
                return False
            redacted = redact_text(parameters)
            if redacted == parameters:
                return False
            value['parameters'] = redacted
            value['record_sha256'] = record_digest(value)
            after = json.dumps(value, allow_nan=False) + '\n'
        else:
            redacted = redact_value(value)
            if redacted == value:
                return False
            after = json.dumps(redacted, indent=2, allow_nan=False) + '\n'
    else:
        after = redact_text(before)
    if after == before:
        return False
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(after)
    temporary.replace(path)
    return True


def redact_distribution():
    paths = [ROOT / name for name in (
        'results/capymoa/environment.json',
        'results/capymoa/protocol_tests.log',
        'results/fixes/pip_capy.log',
        'results/fixes/pip_river.log',
    )]
    paths.extend(sorted((ROOT / 'results/capymoa/runs').glob('*.json')))
    return [str(path.relative_to(ROOT)) for path in paths
            if path.is_file() and redact_file(path)]


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    changed = redact_distribution()
    print(json.dumps({'redacted_files': changed, 'model_training_performed': False}, indent=2))
