# Reproduction guide

Run commands from this directory. The actual test-then-train loops are in
[river_lab/evaluation.py](river_lab/evaluation.py) and
[capy_lab/experiments.py](capy_lab/experiments.py); the executed notebooks link to
these implementations and preserve their measured outputs.

## Separate environments

The measured setup used Python 3.11.16, River 0.26.1, CapyMOA 0.15.1 and Java
17.0.20.1. Do not merge the two dependency locks: their NumPy/Pandas versions differ.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m ipykernel install --sys-prefix --name lab4-river --display-name "Python 3.11 (lab4 River)"

python3.11 -m venv .venv-capy
.venv-capy/bin/python -m pip install -r requirements-capy.txt
.venv-capy/bin/python -m pip check
.venv-capy/bin/python -m ipykernel install --sys-prefix --name lab4-capy --display-name "Python 3.11 (lab4 CapyMOA)"
```

Use a Java 17 JDK and set `JAVA_HOME` if automatic detection is unavailable.
`capy_lab.runtime.configure()` must run before importing CapyMOA. It checks the
source/overlay metadata in `vendor/` and loads the project overlay before the
unchanged upstream MOA JAR. `scripts/build_moa_patch.py` rebuilds that overlay.
Restart Python/Jupyter after changing JVM classpaths.

Notebook environment-folder names are advisory. Expected/observed core versions
are printed, and runtime identities still guard reuse. For a nonstandard River
environment, strict result checking can select its interpreter explicitly:

```bash
python scripts/verify_submission_payload.py --strict-runtime --river-python /absolute/path/to/river-environment/bin/python
```

Run that command in the CapyMOA environment. The default River interpreter for
strict two-environment checks is `.venv/bin/python`.

## Check existing evidence without retraining

```bash
.venv-capy/bin/python scripts/verify_submission_payload.py
.venv-capy/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_notebook_delivery.py --part 1
.venv-capy/bin/python scripts/verify_notebook_delivery.py --part 2
```

The result checker compares 315 sealed records with CSVs, verifies eight ordered
caches, alarm matching/positions, 900 AutoML checkpoints, six existing AutoClass
reproduction cases and Java overlay/source hashes. Its default mode checks
stored evidence, not the current platform; `--strict-runtime` adds the measured
runtime-identity requirement. The 18 tests comprise 12 core protocol/regression
checks, four notebook-entry checks and two public-metadata redaction checks.

The notebook checks read saved `.ipynb` outputs directly. Optional
checks may generate a local `verification/` folder; these delivery-check
snapshots and standalone whole-project hash manifests are not shipped here.
Internal `results/*/validation.json` and record seals remain because the program
uses them. Targeted checks do not prove arbitrary-platform
or fresh-install equivalence.

## Execute the notebooks

```bash
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/river_steps_1_5.ipynb --ExecutePreprocessor.kernel_name=lab4-river
.venv-capy/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/capymoa_steps_7_10.ipynb --ExecutePreprocessor.kernel_name=lab4-capy
```

The default mode reads formal evidence and fits only the clearly labeled
100-example illustration in each notebook. Those scores never enter formal
CSVs or the report. Notebook rerun controls use `--resume`: compatible records
can be reused. River calls `run --resume`, then `analyze`, avoiding unnecessary
`prepare` and downloads. Kernel folder names do not have to match the suggested
names if the dependencies are correctly installed.

## Recompute formal experiments

Use a second working copy: these commands replace formal results and omit
`--resume` to request recomputation from the supplied ordered caches.

```bash
.venv/bin/python scripts/run_river.py run
.venv/bin/python scripts/run_river.py analyze
.venv-capy/bin/python scripts/run_capy.py all
.venv-capy/bin/python scripts/verify_fixed_reproducibility.py --against-formal
.venv-capy/bin/python scripts/verify_submission_payload.py --strict-runtime
.venv-capy/bin/python scripts/run_capy.py analyze
```

Adding `--resume` permits compatible reuse; it does not guarantee retraining.
River `all` additionally invokes `prepare`, so it is not the cached-data rerun
entry used here. Explicit preparation may need downloads if data scope changes.
Model timing and sampled RSS vary across machines. Fixed environments were used
for the existing checks; a fresh network installation was not tested.

The experiment interpretation retains the disclosed limits: different Adam/SGD optimizers, different
Part 1/2 River profiles, real-data prefixes, static file order, shared stream
realizations, common k=3 macro clustering and unequal AutoML spaces/budgets.

## Code delivery and analysis outputs

The code-delivery link is [coniinee/online-ml-lab-river-capymoa](https://github.com/coniinee/online-ml-lab-river-capymoa). The written lab report is submitted
separately and is not stored in this repository.

`run_river.py analyze` and `run_capy.py analyze` generate summaries, experimental
plots and English findings under `results/`. Neither needs report templates or
creates the written report. Preserve the stated limitations when interpreting
these results.

To package a working copy, run `python scripts/package_repository.py`. The data
caches, raw records, dependency locks and Java overlay are part of reproduction;
keep them alongside the notebooks and source modules.

## Public metadata

The public distribution replaces machine-specific paths in environment evidence,
installation/test logs and six AutoClass parameter descriptions with placeholders
such as `<PROJECT_ROOT>`, `<CAPY_ENV>`, `<USER_HOME>` and `<JAVA_HOME>`.
These descriptions are evidence, not commands to execute verbatim. Measurements,
predictions, selector trajectories, calculation fingerprints and dependency/JAR
hashes are unchanged. The six affected record content seals were recomputed for
the redacted parameter text; this is not a new training run.

The CapyMOA CLI redacts saved parameter descriptions after each worker completes,
outside the measured training loop. The environment-capture script also redacts
paths before writing. For an existing working copy, run:

```bash
python scripts/redact_public_paths.py
```

The redactor verifies existing model-record seals before changing parameter text;
it does not repair invalid records or alter measured results. Custom external
logs and direct calls to lower-level worker functions need a separate review
before publication. The upstream MOA class's embedded default developer path is
unchanged; the supplied configuration overrides that upstream default.
