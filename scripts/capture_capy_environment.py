"""Capture the actual JVM and reviewed overlay used by corrected Part 2 runs."""
from pathlib import Path
import sys,json,io,contextlib,platform,importlib.metadata,hashlib,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from capy_lab.runtime import configure
cfg=configure()
import capymoa,jpype,psutil
from lab_support.identity import environment
from scripts.redact_public_paths import redact_value
buffer=io.StringIO()
with contextlib.redirect_stdout(buffer):capymoa.about()
jar=Path(importlib.metadata.distribution('capymoa').locate_file('capymoa/jar/moa.jar'))
value={'python':platform.python_version(),'python_executable':sys.executable,'platform':platform.platform(),
       'logical_cpus':os.cpu_count(),'physical_memory_gib':psutil.virtual_memory().total/2**30,
       'java_version':str(jpype.java.lang.System.getProperty('java.version')),'java_home':os.environ['JAVA_HOME'],
       'jvm_arguments':[str(v) for v in jpype.java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments()],
       'moa_jar_sha256':hashlib.sha256(jar.read_bytes()).hexdigest(),
       'automl_overlay':json.loads((ROOT/'vendor/automl-seeded.json').read_text()),
       'versions':environment(2)['versions'],'about':buffer.getvalue()}
(ROOT/'results/capymoa/environment.json').write_text(json.dumps(redact_value(value),indent=2)+'\n')
print('Captured upstream JAR, overlay, actual Java arguments and dependency versions.')
