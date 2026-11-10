"""Configure project-local paths and the JVM before importing CapyMOA."""

import json
import hashlib
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure():
    config = json.loads((ROOT / "config/capymoa.json").read_text())
    from .configuration import validate_config
    validate_config(config)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    os.environ["CAPYMOA_DATASETS_DIR"] = str(ROOT / "data/capymoa")
    os.environ["CAPYMOA_JVM_ARGS"] = config["jvm_args"]
    (ROOT / "data/capymoa").mkdir(parents=True, exist_ok=True)
    if platform.system() == "Darwin":
        result = subprocess.run(["/usr/libexec/java_home", "-v", "17"], capture_output=True, text=True)
        if result.returncode == 0:
            os.environ["JAVA_HOME"] = result.stdout.strip()
    # Load narrowly patched AutoML classes before CapyMOA adds the upstream JAR.
    overlay = ROOT / "vendor/automl-seeded.jar"
    metadata = json.loads((ROOT / "vendor/automl-seeded.json").read_text())
    if hashlib.sha256(overlay.read_bytes()).hexdigest() != metadata["overlay_sha256"]:
        raise RuntimeError("AutoML overlay checksum mismatch; rebuild scripts/build_moa_patch.py")
    for name, expected in metadata["sources"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("AutoML patch sources changed; rebuild scripts/build_moa_patch.py")
    import jpype
    if not jpype.isJVMStarted():
        jpype.addClassPath(str(overlay))
    return config
