"""Build the notebook-first English notebook; execute it to save fresh outputs."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
CELLS=[('markdown',
  '# Online Machine Learning with CapyMOA — Steps 7–10\n'
  '\n'
  'This notebook presents executed results and provides an entry point for reproduction. The full '
  'test-then-train loops are implemented in [capy_lab/experiments.py](../capy_lab/experiments.py); formal '
  'runs are reproduced through [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).\n'
  '\n'
  'By default, the notebook reads saved evidence rather than retraining the formal models. One clearly '
  'labeled 100-observation example trains an illustrative learner only; its scores never enter the formal '
  'CSV files or report. Set `RERUN_EXPERIMENTS=True` only in a separate working copy when formal runner '
  'execution is intended.\n'
  '\n'
  '## Source navigation\n'
  '\n'
  '- [Full evaluation implementation](../capy_lab/experiments.py)\n'
  '- [Model constructors and settings](../capy_lab/models.py)\n'
  '- [Formal run entry point](../scripts/run_capy.py)\n'
  '- [Installation and two-level reproduction guide](../REPRODUCIBILITY.md)\n'
  '- [Formal measured results](../results/capymoa/)'),
 ('markdown',
  '## Step 7 — Installation and Java configuration\n'
  '\n'
  'Select **Python 3.11 (lab4 CapyMOA)**. Before importing CapyMOA, configure Java 17, the 4 GiB maximum '
  'heap, one active processor and the verified seeded overlay. Part 1 retains its `.venv`; this part uses '
  '`.venv-capy` and `requirements-capy.txt`.'),
 ('code',
  'from pathlib import Path\n'
  'import os, sys, json, subprocess, importlib.metadata\n'
  "PROJECT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'config/capymoa.json').exists())\n"
  'os.chdir(PROJECT);sys.path.insert(0,str(PROJECT))\n'
  'from capy_lab.runtime import configure\n'
  'config=configure()\n'
  'import capymoa\n'
  'import pandas as pd\n'
  'from IPython.display import display, Markdown, Image\n'
  "print('Project:', PROJECT.name)\n"
  "print('Python:', sys.version.split()[0])\n"
  '# Retain runtime/version evidence without printing machine-specific paths.\n'
  'import io\n'
  'from contextlib import redirect_stdout\n'
  'runtime_description = io.StringIO()\n'
  'with redirect_stdout(runtime_description):\n'
  '    capymoa.about()\n'
  'for runtime_line in runtime_description.getvalue().splitlines():\n'
  '    runtime_line = runtime_line.strip()\n'
  "    if runtime_line.startswith(('CapyMOA ', 'CAPYMOA_JVM_ARGS:', 'MOA version:', 'JAVA version:')):\n"
  '        print(runtime_line)\n'
  "print('Seeded AutoML "
  "patch:',json.loads((PROJECT/'vendor/automl-seeded.json').read_text())['overlay_sha256'])\n"
  "for name in ['river','numpy','pandas','scikit-learn','jupyterlab']:\n"
  '    print(name,importlib.metadata.version(name))\n'
  "pd.set_option('display.max_rows',100)\n"
  "pd.set_option('display.max_columns',20)\n"
  "pd.set_option('display.float_format',lambda x:f'{x:.4f}')\n"
  "OUT=PROJECT/'results/capymoa';FIG=OUT/'figures'\n"
  '\n'
  '# A virtual-environment folder name is advisory; the interpreter can live elsewhere.\n'
  "if Path(sys.prefix).resolve() != (PROJECT / '.venv-capy').resolve():\n"
  "    print('NOTE: custom kernel prefix detected. Folder names are not required to match.')\n"
  "expected_core_versions = {'capymoa': '0.15.1', 'river': '0.26.1'}\n"
  'observed_core_versions = {p: importlib.metadata.version(p) for p in expected_core_versions}\n'
  "print('Expected core versions:', expected_core_versions)\n"
  "print('Observed core versions:', observed_core_versions)\n"
  'if observed_core_versions != expected_core_versions:\n'
  "    print('NOTE: versions differ from the measured environment. Use the separate dependency locks before "
  "formal training; runtime identities control cache reuse.')\n"),
 ('markdown',
  '## Run control and scope\n'
  '\n'
  'The formal study contains 96 classification runs, 27 clustering runs and 24 AutoML runs. Each job runs '
  'sequentially in a fresh process. The three model seeds share the same cached stream; they are not '
  'independent dataset realizations. Exploratory trials are excluded.'),
 ('code',
  '# Refresh or reuse compatible formal results; --resume does not force retraining.\n'
  'RERUN_EXPERIMENTS=False\n'
  'if RERUN_EXPERIMENTS:\n'
  "    subprocess.run([sys.executable,'scripts/run_capy.py','all','--resume'],check=True)\n"
  "    subprocess.run([sys.executable,'scripts/run_capy.py','analyze'],check=True)\n"
  'print(json.dumps(config,indent=2))\n'
  "data_manifest=json.loads((PROJECT/'data/manifest.json').read_text())\n"
  "display(pd.DataFrame(data_manifest['datasets'])[['dataset','task','n_samples','original_n_samples']])\n"),
 ('markdown',
  '## Step 8 — Classification on common inputs\n'
  '\n'
  'Use Elec2, Phishing, ImageSegments and SEARecurring. Both libraries receive the same fixed-dimensional '
  'vectors, with absent one-hot entries filled by zero. Shared settings include grace=100, memory '
  'parameter=64 MiB, adaptive-NB leaves, five forest trees, lambda=6, integer floor(sqrt(d)) features and '
  'ADWIN deltas 0.002/0.01. Both omit a depth cap because MOA has no equivalent option. Part 1 instead uses '
  'sparse inputs and depth 12.\n'
  '\n'
  '**LinearLogistic is an approximate correspondence**: River Adam and CapyMOA inverse-scaling sklearn SGD '
  'are different optimizers. Linear models share causal online scaling. Prediction objects contain a fixed '
  'class-slot zero; true labels are used after prediction for scoring and training.\n'
  '\n'
  'Item latency includes adapters, preprocessing and learning, but excludes cache reading, initial JVM '
  'startup and metric computation. Interface costs and first-call JIT effects remain relevant.'),
 ('code',
  "classification=pd.read_csv(OUT/'classification_metrics.csv')\n"
  "display(classification.groupby(['dataset','model','library'],sort=False)[['accuracy','macro_f1','kappa','mean_latency_us','p95_latency_us','total_runtime_s','sampled_peak_process_rss_mib']].mean().reset_index())\n"
  "pairs=pd.read_csv(OUT/'classification_pairs.csv')\n"
  "display(pairs.groupby(['dataset','model'],sort=False)[['accuracy_difference_pp','capy_to_river_latency_ratio']].mean().reset_index())\n"
  "for name in ['classification_accuracy','classification_latency','classification_curves']:\n"
  "    display(Image(filename=str(FIG/f'{name}.png')))\n"),
 ('markdown',
  '### Relationship to Part 1 River results\n'
  '\n'
  'Data order, sample count and cache hashes are unchanged. Part 2 reruns River with explicit zero-filled '
  'features and no tree-depth cap. The following table separates the original profile from this controlled '
  'profile, preventing representation/settings changes from being attributed to the library alone.'),
 ('code',
  "historical=pd.read_csv(OUT/'part1_historical_comparison.csv')\n"
  "display(historical.groupby(['dataset','model'],sort=False)[['part1_accuracy','accuracy_river','accuracy_capy','mean_latency_us_river','mean_latency_us_capy']].mean().reset_index())\n"),
 ('markdown',
  '## Bounded prequential illustration — not part of the formal results\n'
  '\n'
  'This 100-example prefix uses the supplied data loader and formal model constructor. Each prediction is '
  'scored before the current observation is learned. It is a small executable illustration of the API and '
  'ordering, not an additional model comparison. No formal result files are written.'),
 ('code',
  '# Illustration only: this does not write any formal run, CSV or report.\n'
  'from capy_lab.common import classification_data, schema_for\n'
  'from capy_lab.models import classifier as demo_classifier\n'
  'from capymoa.core import LabeledInstance\n'
  'from river import metrics as demo_metrics\n'
  'DEMO_N = 100\n'
  "demo_x, demo_y, demo_meta = classification_data('Elec2')\n"
  "demo_schema = schema_for(demo_x.shape[1], len(demo_meta['class_labels']))\n"
  "demo_model = demo_classifier('CapyMOA', 'HoeffdingTree', demo_schema, demo_x.shape[1], 42, config)\n"
  'demo_score = demo_metrics.Accuracy()\n'
  'for demo_row, demo_label in zip(demo_x[:DEMO_N], demo_y[:DEMO_N]):\n'
  '    demo_prediction = demo_model.predict(LabeledInstance.from_array(demo_schema, demo_row, 0))\n'
  '    demo_prediction = 0 if demo_prediction is None else int(demo_prediction)\n'
  '    demo_score.update(int(demo_label), demo_prediction)\n'
  '    demo_model.train(LabeledInstance.from_array(demo_schema, demo_row, int(demo_label)))\n'
  "print('ILLUSTRATION ONLY — 100-example prefix; not a formal benchmark result.')\n"
  "display({'demo_only': True, 'dataset': 'Elec2', 'examples': DEMO_N, 'accuracy': demo_score.get()})\n"),
 ('markdown',
  '## Step 9 — Cluster quality and evolution\n'
  '\n'
  'CluStream, ClusTree and DenStream receive features from ImageSegments, Phishing and a three-component '
  'MovingGaussians stream. The first 200 unlabeled points calibrate a subsequently frozen scale. A common '
  'weighted KMeans macro stage targets k=3, refreshes every 200 points and uses Hungarian matching to '
  'stabilize IDs.\n'
  '\n'
  'Reference labels are reserved for ARI evaluation; silhouette uses features and assignments. Java '
  'instances contain only features plus a constant class-slot zero, avoiding a ClusTree missing-value '
  'problem without exposing reference labels.\n'
  '\n'
  'Each later point is assigned using previous centers before training. Window quality is assessed every 500 '
  'points over at most 500 recent assignments. Coverage and undefined silhouette remain explicit. '
  'DenStream/Phishing degenerates to one effective cluster under these settings: ARI=0 and silhouette '
  'undefined. DenStream uses micro-clusters plus the common macro stage, not native DBSCAN macro output. '
  'Snapshots show post-training centers; quality curves preserve historical pre-training assignments.'),
 ('code',
  "clustering=pd.read_csv(OUT/'clustering_metrics.csv')\n"
  "display(clustering.groupby(['dataset','model'],sort=False)[['mean_window_ari','mean_window_silhouette','mean_window_coverage','mean_window_mse','mean_latency_us','total_runtime_s','final_micro_clusters']].mean().reset_index())\n"
  "for name in ['clustering_evolution','clustering_silhouette','clustering_snapshots']:\n"
  "    display(Image(filename=str(FIG/f'{name}.png')))\n"),
 ('markdown',
  '## Step 10 — Online AutoML bonus\n'
  '\n'
  'Compare fixed HT with Bandit, SuccessiveHalving and AutoClass. Bandit and SH share five candidates. '
  'Bandit epsilon=0.1, burn-in=100 arm pulls (20 observations with five candidates). SH has nominal budget '
  '2n, which is not a strict total-update cap; actual candidate updates are recorded. Its adapter restricts '
  'prediction to active candidates.\n'
  '\n'
  'AutoClass searches HT/HAT configurations with window 500, ensemble 8 and two new configurations, using '
  'the project-local seeded Java overlay. Complete fresh-process reproduction evidence covers predictions '
  'and configuration trajectories. Search spaces/budgets differ across selectors, limiting strategy-only '
  'comparisons. The outer loop predicts/scores before selector training; no full-stream hindsight oracle is '
  'reported.'),
 ('code',
  "automl=pd.read_csv(OUT/'automl_metrics.csv')\n"
  "display(automl.groupby(['dataset','model'],sort=False)[['accuracy','macro_f1','kappa','mean_latency_us','total_runtime_s']].mean().reset_index())\n"
  "for name in ['automl_curves','automl_accuracy_cost']:\n"
  "    display(Image(filename=str(FIG/f'{name}.png')))\n"
  "states=json.loads((OUT/'automl_final_states.json').read_text())\n"
  'display(pd.DataFrame(states))\n'),
 ('markdown',
  '## Validation, analysis and source\n'
  '\n'
  'Raw records, CLI parameters, predictive hashes, selector states, input hashes and timing evidence are in '
  '`results/capymoa/`. The entry point is `scripts/run_capy.py`; actual training/evaluation code is in '
  '`capy_lab/`. The lab report is submitted separately.'),
 ('code',
  "checks=json.loads((OUT/'validation.json').read_text())\n"
  'assert all(checks.values()),checks\n'
  "display(pd.DataFrame(checks.items(),columns=['check','passed']))\n"
  "display(Markdown((OUT/'findings.md').read_text()))\n"),
 ('markdown',
  '## Reproduction commands\n'
  '\n'
  'Use a separate working copy when training, because the runner replaces result files. Only the explicitly '
  'labeled 100-example illustration fits a learner in the default mode; formal results are read from saved '
  'evidence.\n'
  '\n'
  '```bash\n'
  'source .venv-capy/bin/activate\n'
  'python scripts/run_capy.py all --resume\n'
  'python scripts/run_capy.py analyze\n'
  'python -m unittest discover -s tests -v\n'
  'python scripts/build_capy_notebook.py\n'
  'python -m jupyter nbconvert --execute --to notebook --inplace notebooks/capymoa_steps_7_10.ipynb '
  '--ExecutePreprocessor.kernel_name=lab4-capy\n'
  '```\n'
  '\n'
  'References: [installation](https://capymoa.org/setup/), '
  '[classification](https://capymoa.org/notebooks/classifier/getting_started.html), '
  '[clustering](https://capymoa.org/notebooks/clusterer/clustering.html), '
  '[AutoML](https://capymoa.org/notebooks/automl/automl.html).')]
METADATA={'kernelspec': {'display_name': 'Python 3.11 (lab4 CapyMOA)', 'language': 'python', 'name': 'lab4-capy'},
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
    path=ROOT/'notebooks/capymoa_steps_7_10.ipynb'
    nbf.write(nb,path)
    print('Built',path)

if __name__=='__main__':main()
