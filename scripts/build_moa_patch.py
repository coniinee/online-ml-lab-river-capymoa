"""Build a reproducible, narrowly scoped seeded-AutoML overlay with Java 17."""
from pathlib import Path
import hashlib, importlib.metadata, json, os, platform, subprocess, tempfile, zipfile

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor'
EXPECTED = 'e7d7c3d533a4800bf2581f820d2c1e9310d70ff40b52fc1a742793eceb89a660'


def main():
    jar = Path(importlib.metadata.distribution('capymoa').locate_file('capymoa/jar/moa.jar'))
    assert hashlib.sha256(jar.read_bytes()).hexdigest() == EXPECTED, 'Review patch against this MOA build first'
    home = os.environ.get('JAVA_HOME')
    if platform.system() == 'Darwin':
        home = subprocess.check_output(['/usr/libexec/java_home', '-v', '17'], text=True).strip()
    if not home: raise RuntimeError('Set JAVA_HOME to a Java 17 JDK')
    java = Path(home) / 'bin'
    exports = '--add-exports=java.base/jdk.internal.org.objectweb.asm=ALL-UNNAMED'
    with tempfile.TemporaryDirectory(prefix='lab4-moa-patch-') as tmp:
        build = Path(tmp)
        subprocess.run([str(java/'javac'), exports, '-d', str(build), *map(str, sorted((VENDOR/'java/lab4').glob('*.java')))], check=True)
        patched = build/'overlay'
        result = subprocess.run([str(java/'java'), exports, '-cp', str(build), 'lab4.PatchAutoML', str(jar), str(patched)], check=True, text=True, capture_output=True)
        changes = {line.split()[0]: int(line.split()[1]) for line in result.stdout.splitlines()}
        assert len(changes) >= 4 and sum(changes.values()) >= 4, changes
        classes = {str(p.relative_to(patched)): p.read_bytes() for p in patched.rglob('*.class')}
        classes['lab4/SeededRandom.class'] = (build/'lab4/SeededRandom.class').read_bytes()
        target = VENDOR/'automl-seeded.jar'
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as output:
            for name, data in sorted(classes.items()):
                info = zipfile.ZipInfo(name, date_time=(2020,1,1,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED
                output.writestr(info, data)
    manifest = {'patch_version': 1, 'upstream_moa_sha256': EXPECTED,
                'overlay_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'replacements': changes,
                'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((VENDOR/'java').rglob('*.java'))}}
    (VENDOR/'automl-seeded.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
