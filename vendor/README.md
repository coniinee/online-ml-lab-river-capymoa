# Seeded AutoClass overlay

`automl-seeded.jar` is a small classpath overlay for the exact upstream MOA build identified in `automl-seeded.json`. It does not replace or edit the installed 63 MiB MOA JAR.

The build rewrites all seven uncontrolled random calls in `moa.classifiers.meta.AutoML`: six `Math.random()` calls and one unseeded `Random()` constructor. They use the explicit per-learner random source activated by `SeededAutoClass`. Java work is restricted to one core, and a lock serializes calls when learners share a Python process.

Build from project root using `.venv-capy` and a Java 17 JDK:

```bash
.venv-capy/bin/python scripts/build_moa_patch.py
```

The builder checks the upstream SHA, uses Java's bundled ASM for the reviewed bytecode substitutions, and writes a timestamp-stable overlay plus its source/replacement/checksum manifest. At runtime, configure paths **before importing CapyMOA**:

```python
from capy_lab.runtime import configure
config = configure()
```

`configure()` verifies overlay/source hashes and puts the overlay before upstream MOA on the classpath. `SeededAutoClass` verifies which JAR actually supplied its class. After changing JVM classpaths, restart the Python/Jupyter process; Java cannot safely unload already loaded classes.

Original model settings and search logic are unchanged. Fix details and empirical reproduction checks are in `FIXES_AND_VALIDATION.md` and `results/fixes/`.
