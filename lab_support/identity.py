"""Runtime-aware identities and content seals for measured records."""
from functools import lru_cache
from pathlib import Path
import hashlib
import copy
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def environment(part):
    packages = ['river', 'numpy', 'pandas', 'psutil']
    if part == 2: packages += ['capymoa', 'scikit-learn', 'scipy', 'jpype1']
    result = {'schema': 3, 'python': sys.version, 'platform': platform.platform(),
              'versions': {name: metadata.version(name) for name in packages},
              'threads': {name: os.environ.get(name) for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}}
    if part == 2:
        java = Path(os.environ['JAVA_HOME'])/'bin/java'
        version = subprocess.run([str(java), '-version'], capture_output=True, text=True, check=True)
        jar = Path(metadata.distribution('capymoa').locate_file('capymoa/jar/moa.jar'))
        result.update({'java': version.stderr.strip(), 'jvm_args': os.environ['CAPYMOA_JVM_ARGS'],
                       'moa_jar_sha256': file_hash(jar), 'automl_overlay_sha256': file_hash(ROOT/'vendor/automl-seeded.jar')})
        patch = json.loads((ROOT/'vendor/automl-seeded.json').read_text())
        if result['moa_jar_sha256'] != patch['upstream_moa_sha256']:
            raise RuntimeError('MOA build differs from the reviewed patch target')
    return result


def calculation_signature(config, part, stage=None):
    files = ['lab_support/identity.py', 'river_lab/data.py']
    if part == 1:
        files += [str(p.relative_to(ROOT)) for p in (ROOT/'river_lab').glob('*.py') if p.name != 'analysis.py']
        files += ['requirements.txt']
    else:
        files += [str(p.relative_to(ROOT)) for p in (ROOT/'capy_lab').glob('*.py') if p.name != 'analysis.py']
        files += ['requirements-capy.txt','vendor/automl-seeded.json']
        if stage == 'automl': files += ['config/autoclass.json']
    payload = {'schema': 3, 'part': part, 'stage': stage, 'config': config,
               'environment': environment(part),
               'manifest': json.loads((ROOT/'data/manifest.json').read_text()),
               'sources': {name: file_hash(ROOT/name) for name in sorted(set(files))}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def record_digest(record):
    payload = {k:v for k,v in record.items() if k != 'record_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seal(record, signature, part):
    record['fingerprint'] = signature
    record['environment'] = copy.deepcopy(environment(part))
    record['versions'] = copy.deepcopy(environment(part)['versions'])
    record['record_sha256'] = record_digest(record)
    return record


def valid_record(record, signature, part):
    try:
        return (record.get('fingerprint') == signature and record.get('environment') == environment(part)
                and record.get('versions') == environment(part)['versions']
                and record.get('record_sha256') == record_digest(record))
    except (TypeError, ValueError):
        return False
