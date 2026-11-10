# Online Machine Learning with River and CapyMOA

This repository contains the actual experiment code, executed English notebooks
and resources needed to reproduce their results. The written report and its
source templates are submitted separately.

## Start here

- [River notebook — Steps 1–6](notebooks/river_steps_1_5.ipynb): regression,
  classification, drift monitoring/reset comparisons and analysis.
- [CapyMOA notebook — Steps 7–10](notebooks/capymoa_steps_7_10.ipynb): controlled
  cross-library classification, stream clustering and AutoML.
- [Reproduction guide](REPRODUCIBILITY.md): separate environments, quick checks,
  notebook execution and complete training commands.

The notebooks retain executed outputs: River has nine code cells and ten figures;
CapyMOA has eight code cells and eight figures. They display saved formal results
by default, with one separately labeled 100-example illustration each. The
illustrations do not write formal result files. `RERUN_EXPERIMENTS=True` uses
`--resume`, which permits compatible reuse and does not force retraining.

## Code and required resources

| Path | Contents |
| --- | --- |
| [river_lab/](river_lab/) | River data loading, learners and test-then-train evaluation |
| [capy_lab/](capy_lab/) | CapyMOA setup, paired classification, clustering, AutoML and analysis |
| [lab_support/](lab_support/) | Runtime/content identities and raw-to-export consistency checks |
| [scripts/](scripts/) | Experiment runners, notebook builders, analysis, validation and packaging |
| [notebooks/](notebooks/) | Executed results, figures, source navigation and bounded examples |
| [config/](config/), [data/](data/) | Settings, data provenance and ordered input caches |
| [results/](results/) | Raw records, CSVs, curves, alarms, windows, parameters and logs |
| [tests/](tests/), [vendor/](vendor/) | Protocol tests and the seeded Java overlay/build source |

Data and result files are retained because the notebooks and runners read them.
The formal study contains 168 River model records and 147 Part 2 records (96
paired classification, 27 clustering, 24 AutoML), plus 72 detector-only replays.
The two 100-example demonstrations are not additional formal benchmark runs.

Use Python 3.11 and two separate dependency locks: `requirements.txt` for River
and `requirements-capy.txt` for CapyMOA. Java 17 is required for CapyMOA. Do not
merge the two environments. Optional quick checks are documented in the guide.
Eighteen tests cover the core protocol, notebook entry points and public-metadata
redaction; stored
measurements are not a claim of fresh-install or arbitrary-platform equivalence.

The analysis commands produce experimental summaries, plots and English
findings under `results/`; they do not require or generate the written report.
The disclosed optimizer, input-profile, ordering, prefix, clustering-stage and
AutoML-budget limitations remain part of the result interpretation.
