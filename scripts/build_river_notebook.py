"""Build the notebook-first English notebook; execute it to save fresh outputs."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
CELLS=[('markdown',
  '# Online Machine Learning with River — Steps 1–6\n'
  '\n'
  'This notebook presents executed results and provides an entry point for reproduction. The full '
  'test-then-train loops are implemented in [river_lab/evaluation.py](../river_lab/evaluation.py); formal '
  'runs are reproduced through [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).\n'
  '\n'
  'By default, the notebook reads saved evidence rather than retraining the formal models. One clearly '
  'labeled 100-observation example trains an illustrative learner only; its scores never enter the formal '
  'CSV files or report. Set `RERUN_EXPERIMENTS=True` only in a separate working copy when formal runner '
  'execution is intended.\n'
  '\n'
  '## Source navigation\n'
  '\n'
  '- [Full evaluation implementation](../river_lab/evaluation.py)\n'
  '- [Model constructors and settings](../river_lab/models.py)\n'
  '- [Formal run entry point](../scripts/run_river.py)\n'
  '- [Installation and two-level reproduction guide](../REPRODUCIBILITY.md)\n'
  '- [Formal measured results](../results/river/)'),
 ('markdown',
  '## Step 1 — Environment verification\n'
  '\n'
  'Use the **Python 3.11 (lab4 River)** kernel in `.venv`. `requirements.txt` pins the measured environment. '
  'The next cell records the interpreter, dependency versions, platform and dependency consistency.'),
 ('code',
  'from pathlib import Path\n'
  'import importlib.metadata\n'
  'import json\n'
  'import os\n'
  'import sys\n'
  'import subprocess\n'
  '\n'
  'PROJECT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config/river.json").exists())\n'
  'os.chdir(PROJECT)\n'
  'sys.path.insert(0, str(PROJECT))\n'
  'import pandas as pd\n'
  'from IPython.display import display, Markdown, Image\n'
  '\n'
  'pd.set_option("display.max_rows", 100)\n'
  'pd.set_option("display.max_columns", 16)\n'
  'pd.set_option("display.float_format", lambda value: f"{value:.4f}")\n'
  '\n'
  'print("Project:", PROJECT.name)\n'
  'print("Python:", sys.version.split()[0])\n'
  'print("Python implementation:", sys.implementation.name)\n'
  'for package in ["river", "numpy", "pandas", "matplotlib", "jupyterlab", "psutil"]:\n'
  '    print(f"{package}: {importlib.metadata.version(package)}")\n'
  'print(subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, '
  'check=True).stdout)\n'
  '\n'
  '# A virtual-environment folder name is advisory; the interpreter can live elsewhere.\n'
  "if Path(sys.prefix).resolve() != (PROJECT / '.venv').resolve():\n"
  "    print('NOTE: custom kernel prefix detected. Folder names are not required to match.')\n"
  "expected_core_versions = {'river': '0.26.1'}\n"
  'observed_core_versions = {p: importlib.metadata.version(p) for p in expected_core_versions}\n'
  "print('Expected core versions:', expected_core_versions)\n"
  "print('Observed core versions:', observed_core_versions)\n"
  'if observed_core_versions != expected_core_versions:\n'
  "    print('NOTE: versions differ from the measured environment. Use the separate dependency locks before "
  "formal training; runtime identities control cache reuse.')\n"),
 ('markdown',
  '## Run control and evaluation protocol\n'
  '\n'
  '- Preserve the supplied order. Predict, score, optionally detect/reset, then learn.\n'
  '- All methods read the same SHA-256-verified cache. Encoding is stateless; scaling updates online.\n'
  '- Model seeds: 42, 43, 44. Synthetic generation seed: 2026. The three seeds share a stream; SDs are not '
  'independent-stream confidence intervals.\n'
  '- Bikes and Elec2 use their first 10,000 examples; smaller real datasets are used in full. Formal runs '
  'are sequential.\n'
  '- Item latency includes prediction and training, including internal transforms. Total loop time '
  'additionally includes evaluation/checkpoints. Data preparation is separate.\n'
  '- Sampled process RSS includes the interpreter and loaded data; it is not model-exclusive or an exact '
  'continuous peak.'),
 ('code',
  '# Refresh or reuse compatible formal results; --resume does not force retraining.\n'
  'RERUN_EXPERIMENTS = False\n'
  '\n'
  'if RERUN_EXPERIMENTS:\n'
  '    subprocess.run([sys.executable, "scripts/run_river.py", "run", "--resume"], check=True)\n'
  '    subprocess.run([sys.executable, "scripts/run_river.py", "analyze"], check=True)\n'
  '\n'
  'OUT = PROJECT / "results/river"\n'
  'FIG = OUT / "figures"\n'
  'config = json.loads((PROJECT / "config/river.json").read_text())\n'
  'manifest = json.loads((PROJECT / "data/manifest.json").read_text())\n'
  'experiment = json.loads((OUT / "experiment.json").read_text())\n'
  'assert experiment["complete"]\n'
  'print(json.dumps(config, indent=2))\n'
  'print("Model runs:", experiment["completed_model_runs"])\n'
  'print("Detector replays:", experiment["completed_monitor_replays"])\n'
  'print("Platform:", experiment["platform"])\n'
  'print("RAM (GiB):", experiment["physical_memory_gb"])\n'),
 ('markdown',
  '## Step 2 — Regression methods and datasets\n'
  '\n'
  '| Family | Method | Principal settings |\n'
  '| --- | --- | --- |\n'
  '| Tree | HoeffdingTreeRegressor | Grace period 100, maximum depth 12, mean leaves |\n'
  '| Ensemble | ARFRegressor | Five trees, maximum depth 12, internal ADWIN, fixed seed |\n'
  '| Adaptive linear | LinearRegression + Adam | Learning rate 0.03; online input and target scaling |\n'
  '| Other | KNNRegressor | k=5; FIFO window 300; online input scaling |\n'
  '\n'
  'Real streams: TrumpApproval, ChickWeights and Bikes. Synthetic stream: recurring abrupt '
  '`FriedmanDrift(gra)`. Bikes timestamps become hour/weekday cycles and date features. Categorical inputs '
  'use stateless one-hot encoding; chick and diet identifiers are categorical. Settings are fixed rather '
  'than selected from full-stream test performance.'),
 ('code',
  'data_table = pd.DataFrame(manifest["datasets"])\n'
  'display(data_table.loc[data_table.task == "regression", ["dataset", "kind", "n_samples", '
  '"original_n_samples", "encoded_n_features", "known_drift_positions", "source"]])\n'
  '\n'
  'from river_lab.models import make_model, REGRESSION_MODELS, CLASSIFICATION_MODELS\n'
  'from river_lab.data import load_stream\n'
  '\n'
  '# Model construction; full settings are stored in the raw records.\n'
  'regression_models = {name: make_model("regression", name, 42, config) for name in REGRESSION_MODELS}\n'
  'for name, model in regression_models.items():\n'
  '    print(f"\\n{name}\\n{model!r}")\n'),
 ('markdown',
  '## Bounded prequential illustration — not part of the formal results\n'
  '\n'
  'This 100-example prefix uses the supplied data loader and formal model constructor. Each prediction is '
  'scored before the current observation is learned. It is a small executable illustration of the API and '
  'ordering, not an additional model comparison. No formal result files are written.'),
 ('code',
  '# Illustration only: this does not write any formal run, CSV or report.\n'
  'from river import metrics as demo_metrics\n'
  'DEMO_N = 100\n'
  "demo_info = next(d for d in manifest['datasets'] if d['dataset'] == 'TrumpApproval')\n"
  'demo_rows = load_stream(demo_info)[:DEMO_N]\n'
  "demo_model = make_model('regression', 'HoeffdingTree', 42, config)\n"
  'demo_score = demo_metrics.MAE()\n'
  'for demo_x, demo_y in demo_rows:\n'
  '    demo_prediction = demo_model.predict_one(demo_x)\n'
  '    demo_prediction = 0.0 if demo_prediction is None else float(demo_prediction)\n'
  '    demo_score.update(demo_y, demo_prediction)  # Score before learning this target.\n'
  '    demo_model.learn_one(demo_x, demo_y)\n'
  "print('ILLUSTRATION ONLY — 100-example prefix; not a formal benchmark result.')\n"
  "display({'demo_only': True, 'dataset': 'TrumpApproval', 'examples': len(demo_rows), 'MAE': "
  'demo_score.get()})\n'),
 ('markdown',
  '## Step 3 — Regression comparison\n'
  '\n'
  'Report cumulative MAE, RMSE and R² with mean/P95 item latency, total loop time and sampled process RSS. '
  'Lower MAE/RMSE and higher R² are preferable. MAE units differ by dataset, so raw MAE is not averaged '
  'across datasets. Cold-start predictions count. Each row below averages three model seeds.'),
 ('code',
  'regression = pd.read_csv(OUT / "regression_metrics.csv")\n'
  'regression_means = regression.groupby(["dataset", "model"], sort=False)[["mae", "rmse", "r2", '
  '"mean_latency_us", "p95_latency_us", "total_runtime_s", '
  '"sampled_peak_process_rss_mb"]].mean().reset_index()\n'
  'display(regression_means)\n'
  'for filename in ["regression_performance", "regression_latency", "regression_learning_curves"]:\n'
  '    display(Image(filename=str(FIG / f"{filename}.png")))\n'),
 ('markdown',
  '## Step 4 — Regression drift detection and handling\n'
  '\n'
  'ADWIN, Page-Hinkley and KSWIN first replay identical ordinary-HT test-then-train errors without changing '
  'that model. Separate alarm-triggered tree-reset experiments measure whether handling improves '
  'prediction.\n'
  '\n'
  'The regression signal is absolute error divided by three times past-target SD, with SD at least one and '
  'the result clipped to [0,1]. The first 200 samples are predicted/learned but withheld from external '
  'detectors. ADWIN delta=0.002; Page-Hinkley threshold=10; KSWIN alpha=0.005. Complete settings are in the '
  'model module.\n'
  '\n'
  'FriedmanDrift changes at zero-based indices **3333 and 6666**. Each true change matches the first unused '
  'alarm within 1500 index steps. Misses and unmatched alarms are explicit; missed changes do not receive '
  'zero delay. Real streams have no verified drift positions.'),
 ('code',
  'drifts = pd.read_csv(OUT / "drift_summary.csv")\n'
  'events = pd.read_csv(OUT / "drift_events.csv")\n'
  'regression_detection = drifts[(drifts.task == "regression") & (drifts["mode"] == "monitor")]\n'
  'display(regression_detection.groupby(["dataset", "detector"], sort=False)[["n_alarms", "matched_changes", '
  '"missed_changes", "unmatched_alarms", "mean_delay", "detector_only_runtime_s"]].mean().reset_index())\n'
  'print("Seed 42 alarms (complete positions are also in drift_events.csv):")\n'
  'display(regression_detection.loc[regression_detection.seed == 42, ["dataset", "detector", '
  '"alarm_positions", "matches"]])\n'
  'display(Image(filename=str(FIG / "regression_drift_alarms.png")))\n'
  '\n'
  'effects = pd.read_csv(OUT / "reset_effect_summary.csv")\n'
  'display(effects[effects.task == "regression"].dropna(axis=1, how="all"))\n'
  'display(Image(filename=str(FIG / "regression_reset_effect.png")))\n'
  '\n'
  'windows = pd.read_csv(OUT / "recovery_windows.csv")\n'
  'display(windows[(windows.task == "regression") & (windows.model == "HoeffdingTree")].groupby(["dataset", '
  '"model", "mode", "detector", "change_position", "window"], sort=False).score.mean().reset_index())\n'),
 ('markdown',
  '## Step 5 — Classification: repeat method, dataset and drift comparisons\n'
  '\n'
  '| Family | Method | Principal settings |\n'
  '| --- | --- | --- |\n'
  '| Tree | HoeffdingTreeClassifier | Grace period 100; depth 12; adaptive-NB leaves |\n'
  '| Ensemble | ARFClassifier | Five trees; depth 12; internal ADWIN |\n'
  '| Adaptive linear | Adam LogisticRegression + OneVsRestClassifier | Rate 0.01; online scaling; multiclass '
  'support |\n'
  '| Other | GaussianNB | Gaussian naive Bayes |\n'
  '\n'
  'Real data: **Elec2, Phishing and ImageSegments**. ImageSegments has seven classes. Report accuracy, '
  "Macro-F1, Cohen's kappa and costs. Phishing and ImageSegments use static file order; alarms are not "
  'verified real-time concept changes.\n'
  '\n'
  '**SEARecurring** uses SEA variant 0 with 5% label noise. Labels invert from index 3333 and return at '
  "6666; the feature distribution stays unchanged. ADWIN, DDM and EDDM receive the model's 0/1 prediction "
  'errors.'),
 ('code',
  'display(data_table.loc[data_table.task == "classification", ["dataset", "kind", "n_samples", '
  '"original_n_samples", "encoded_n_features", "class_labels", "known_drift_positions"]])\n'
  'classification_models = {name: make_model("classification", name, 42, config) for name in '
  'CLASSIFICATION_MODELS}\n'
  'for name, model in classification_models.items():\n'
  '    print(f"\\n{name}\\n{model!r}")\n'
  '\n'
  'classification = pd.read_csv(OUT / "classification_metrics.csv")\n'
  'classification_means = classification.groupby(["dataset", "model"], sort=False)[["accuracy", "macro_f1", '
  '"kappa", "mean_latency_us", "p95_latency_us", "total_runtime_s", '
  '"sampled_peak_process_rss_mb"]].mean().reset_index()\n'
  'display(classification_means)\n'
  'for filename in ["classification_performance", "classification_latency", '
  '"classification_learning_curves"]:\n'
  '    display(Image(filename=str(FIG / f"{filename}.png")))\n'),
 ('code',
  'classification_detection = drifts[(drifts.task == "classification") & (drifts["mode"] == "monitor")]\n'
  'display(classification_detection.groupby(["dataset", "detector"], sort=False)[["n_alarms", '
  '"matched_changes", "missed_changes", "unmatched_alarms", "mean_delay", '
  '"detector_only_runtime_s"]].mean().reset_index())\n'
  'print("Seed 42 alarms:")\n'
  'display(classification_detection.loc[classification_detection.seed == 42, ["dataset", "detector", '
  '"alarm_positions", "matches"]])\n'
  'display(Image(filename=str(FIG / "classification_drift_alarms.png")))\n'
  'display(effects[effects.task == "classification"].dropna(axis=1, how="all"))\n'
  'display(Image(filename=str(FIG / "classification_reset_effect.png")))\n'
  'display(windows[(windows.task == "classification") & (windows.model == '
  '"HoeffdingTree")].groupby(["dataset", "model", "mode", "detector", "change_position", "window"], '
  'sort=False).score.mean().reset_index())\n'),
 ('markdown',
  '## Result validation and analysis\n'
  '\n'
  'The next cell checks completed combinations, sample counts, valid metrics and consistent alarms. The '
  'English analysis covers quality/cost trade-offs, missed changes, reset effects and experimental limits.'),
 ('code',
  'validation = json.loads((OUT / "validation.json").read_text())\n'
  'assert all(validation.values()), validation\n'
  'display(pd.DataFrame(validation.items(), columns=["check", "passed"]))\n'
  'display(Markdown((OUT / "findings.md").read_text()))\n'),
 ('markdown',
  '## Step 6 — Analysis and reproducibility\n'
  '\n'
  'The lab report is submitted separately. This repository provides executed result tables and figures, the '
  'actual evaluation loops and the saved experimental evidence. The reproduction guide describes the '
  'separate environments and complete runner commands.\n'
  '\n'
  '```bash\n'
  'source .venv/bin/activate\n'
  'python -m unittest discover -s tests -p test_protocol.py -v\n'
  'python scripts/run_river.py run --resume\n'
  'python scripts/run_river.py analyze\n'
  'python scripts/build_river_notebook.py\n'
  'python -m jupyter nbconvert --execute --to notebook --inplace notebooks/river_steps_1_5.ipynb '
  '--ExecutePreprocessor.kernel_name=lab4-river\n'
  '```\n'
  '\n'
  'The training command may overwrite results; use a separate working copy. The notebook displays existing '
  'formal results and one separately labeled 100-example illustration.\n'
  '\n'
  '- `results/river/*_metrics.csv`: per-run metrics; `*_summary.csv`: means and seed SDs.\n'
  '- `drift_events.csv`: alarm positions; `drift_summary.csv`: matching, misses and delays.\n'
  '- `reset_effect_pairs.csv`: per-seed controlled comparisons.\n'
  '- `recovery_windows.csv`: 500-point windows before/after changes and at the end of the matching '
  'interval.\n'
  '- `results/river/runs/`: raw records, parameters and content/runtime identities.\n'
  '- `data/manifest.json`: provenance, sample order/count and uncompressed SHA-256.\n'
  '- `config/river.json`, `requirements.txt`, `results/river/experiment.json`: settings, dependencies, '
  'platform and timing definitions.\n'
  '\n'
  'References: [River documentation](https://riverml.xyz/latest/), [source and '
  'datasets](https://github.com/online-ml/river).')]
METADATA={'kernelspec': {'display_name': 'Python 3.11 (lab4 River)', 'language': 'python', 'name': 'lab4-river'},
 'language': 'en',
 'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3},
                   'file_extension': '.py',
                   'mimetype': 'text/x-python',
                   'name': 'python',
                   'nbconvert_exporter': 'python',
                   'pygments_lexer': 'ipython3',
                   'version': '3.11.16'}}

def main():
    cells=[nbf.v4.new_markdown_cell(s) if kind=='markdown' else nbf.v4.new_code_cell(s) for kind,s in CELLS]
    nb=nbf.v4.new_notebook(cells=cells,metadata=METADATA)
    nbf.validate(nb)
    path=ROOT/'notebooks/river_steps_1_5.ipynb'
    nbf.write(nb,path)
    print('Built',path)

if __name__=='__main__':main()
