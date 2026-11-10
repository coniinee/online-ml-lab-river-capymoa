"""English presentation of existing sealed experiments; no model fitting."""
from pathlib import Path
import pandas as pd

def table(frame):
    def fmt(value):
        if pd.isna(value):return '—'
        return f'{value:.4f}' if isinstance(value,float) else str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])

def write_river(root):
    folder=Path(root)/'results/river'
    read=lambda name:pd.read_csv(folder/name)
    lines=['# River: measured results and analysis','',
           '168 model runs and 72 detector-only replays use model seeds 42, 43 and 44, with synthetic generation seed 2026. Each observation is predicted and scored before detection/reset and learning. All methods use the same ordered caches.','',
           '## Setup and scope','',
           'Regression families are an ordinary Hoeffding tree, a five-tree adaptive random forest, scaled Adam linear regression and scaled KNN (k=5, FIFO window 300). Classification uses the tree, forest, scaled one-vs-rest Adam logistic regression and Gaussian naive Bayes. The Part 1 trees have depth limit 12. Input scaling and regression target scaling update online; categorical encoding is stateless.','',
           'Bikes and Elec2 use their first 10,000 observations. Smaller real datasets are used in full; both synthetic streams contain 10,000 examples. Phishing and ImageSegments are static benchmarks without verified chronological order. The three model seeds share one stream realization: their SDs are not independent-stream confidence intervals.','']
    for task,metrics in [('regression',['mae','rmse','r2']),('classification',['accuracy','macro_f1','kappa'])]:
        frame=read(task+'_metrics.csv')
        columns=metrics+['mean_latency_us','total_runtime_s']
        means=frame.groupby(['dataset','model'],sort=False)[columns].mean().reset_index()
        lines.extend(['## '+task.capitalize()+' performance','',table(means),''])
        for dataset,part in means.groupby('dataset',sort=False):
            best=part.loc[part.mae.idxmin()] if task=='regression' else part.loc[part.accuracy.idxmax()]
            fast=part.loc[part.mean_latency_us.idxmin()];metric=metrics[0]
            lines.append(f'- {dataset}: best mean {metric}: {best.model} ({best[metric]:.4f}); fastest: {fast.model} ({fast.mean_latency_us:.2f} µs/item). The best-quality model costs {best.mean_latency_us/fast.mean_latency_us:.2f} times the lowest mean latency.')
        lines.append('')
    drift=read('drift_summary.csv')
    known=drift[(drift['mode']=='monitor')&drift.known_changes.notna()]
    summary=known.groupby(['dataset','detector'],sort=False)[['n_alarms','matched_changes','missed_changes','unmatched_alarms','mean_delay']].mean().reset_index()
    lines+=['## Drift detection and handling','',
            'ADWIN, Page-Hinkley and KSWIN replay the same ordinary-tree regression error signal; ADWIN, DDM and EDDM replay its classification error. Regression errors are normalized using past-target SD and clipped to [0,1]. Classification uses 0/1 error. The first 200 samples are scored/learned but withheld from external detectors.','',
            'Both synthetic changes occur at zero-based indices 3333 and 6666. FriedmanDrift uses the global recurring abrupt concept; SEARecurring inverts then restores labels with 5% noise. Each change matches the first unused alarm within 1500 index steps. Missed changes do not receive zero delay; other alarms remain unmatched rather than verified false positives.','',table(summary),'',
            'ADWIN matches both changes on both streams. Page-Hinkley has a shorter successful Friedman delay but misses the other change. KSWIN and EDDM produce more unmatched alarms here. Static real datasets provide no verified drift truth.','',
            'Reset runs change only whether an external alarm replaces the same ordinary tree after the current prediction is scored. The replacement learns the current observation once. Reset errors can change later alarm positions. ARF already adapts internally and is not used as this controlled reset baseline.','']
    pairs=read('reset_effect_pairs.csv')
    effects=pairs.groupby(['task','dataset','detector_reset'],sort=False)[['mae_change','accuracy_change','latency_ratio']].mean().reset_index()
    lines+=['Reset minus no-reset differences: negative MAE changes and positive accuracy changes are improvements. Accuracy is a fraction in this table.','',table(effects),'']
    for dataset,metric in [('FriedmanDrift','mae'),('SEARecurring','accuracy')]:
        part=pairs[(pairs.dataset==dataset)&(pairs.detector_reset=='ADWIN')]
        lines.append(f'- {dataset}: ADWIN reset changes {metric} from {part[metric+"_baseline"].mean():.4f} to {part[metric+"_reset"].mean():.4f}.')
    lines+=['','## Interpretation and limits','',
            'Full replacement worsens cumulative Friedman error but substantially improves recurring SEA classification. An alarm is not sufficient evidence that reset improves prediction. Relearning cost and loss of useful recurring knowledge are possible explanations, not isolated causal findings.','',
            'MAE units differ across datasets, so raw MAE is not averaged across tasks. Negative R² means squared error exceeds the target-mean evaluation reference; that reference is not used for training. Macro-F1 and kappa complement accuracy.','',
            'Latency includes prediction, online model transforms and learning; reset runs also include detection and reconstruction. Total loop time additionally includes evaluation/checkpoints. Data preparation and initial construction are excluded. Detector-only replay costs are separate. RSS samples cover the process and data, not model-exclusive or continuously measured peak memory.','',
            'The supplied order, modest fixed settings, sample prefixes and shared stream realization limit generalization. Detailed alarms are in drift_events.csv, matching in drift_summary.csv, recovery windows in recovery_windows.csv, and per-run parameters/records in runs/.','',
            'Configuration: config/river.json. Data scope and input hashes: data/manifest.json. Runtime and timing definitions: results/river/experiment.json. Dependencies: requirements.txt. The executed notebooks and CSV exports provide the analysis outputs.']
    (folder/'findings.md').write_text('\n'.join(lines)+'\n')

def write_capy(root):
    folder=Path(root)/'results/capymoa';read=lambda name:pd.read_csv(folder/name)
    cm=read('classification_metrics.csv');km=read('clustering_metrics.csv');am=read('automl_metrics.csv')
    means=cm.groupby(['dataset','model','library'],sort=False)[['accuracy','macro_f1','kappa','mean_latency_us','total_runtime_s']].mean().reset_index()
    pairs=read('classification_pairs.csv').groupby(['dataset','model'],sort=False)[['accuracy_difference_pp','capy_to_river_latency_ratio']].mean().reset_index()
    lines=['# CapyMOA: measured results and analysis','',
           'Steps 7–10 include 96 paired classification runs, 27 stream-clustering runs and 24 AutoML runs. Formal runs are sequential and use a fresh process for each model run. Seeds 42, 43 and 44 share one cached stream per dataset. Dependencies are pinned separately from Part 1.','',
           '## Controlled cross-library classification','',
           'Both libraries receive identical dense vectors with absent one-hot entries filled by zero. Shared settings include grace=100, memory parameter=64 MiB, NBA leaves, five forest trees, lambda=6, floor(sqrt(d)) features and ADWIN drift/warning deltas 0.002/0.01. Both profiles omit an explicit tree-depth cap because MOA has no matching option.','',
           'Linear models share causal scaling, but River Adam(lr=0.01) and CapyMOA inverse-scaling SGD(eta0=0.01, log_loss, no penalty) are different optimizers. Their differences cannot be attributed solely to the library. Backend split statistics, variance handling and random-number implementations also differ. Prediction objects contain a fixed zero class placeholder; true labels enter scoring and subsequent training.','',
           'Part 1 uses sparse inputs and tree depth 12. Part 2 reruns River with the aligned profile; the two River result tables must not be treated as an identical configuration.','',table(means),'',
           'Accuracy differences below are CapyMOA minus River in percentage points. A latency ratio below one favors CapyMOA under these interfaces and settings, not necessarily its isolated kernel.','',table(pairs),'']
    for dataset,part in means.groupby('dataset',sort=False):
        best=part.loc[part.accuracy.idxmax()]
        lines.append(f'- {dataset}: highest controlled mean accuracy: {best.library} / {best.model}, {best.accuracy:.2%}.')
    clusters=km.groupby(['dataset','model'],sort=False)[['mean_window_ari','mean_window_silhouette','mean_window_coverage','mean_latency_us']].mean().reset_index()
    lines+=['','## Unsupervised stream clustering','',
            'CluStream, ClusTree and DenStream receive features from ImageSegments, Phishing and the 6,000-point MovingGaussians stream (changes at 2000/4000). Reference labels/components are reserved for ARI and never used to train or tune the learners. A fixed class-slot zero avoids a ClusTree missing-value problem.','',
            'CluStream uses horizon 1000 and 50 kernels; ClusTree uses horizon 1000, height 5 and depth-first insertion. DenStream uses horizon 1000, epsilon 0.5, beta 0.2, mu 10, init 100, lambda 0.25 and speed 100. A common weighted KMeans macro stage targets k=3 and refreshes every 200 observations; Hungarian matching stabilizes IDs. DenStream native DBSCAN macro/noise output is not evaluated.','',
            'The first 200 unlabeled points calibrate and initialize the learners. Scaling then freezes and distances are normalized by sqrt(d). Subsequent points are assigned using previous centers before training. Window scores are recorded every 500 points; silhouette samples at most 200 assigned points. Coverage is explicit and invalid silhouette remains missing.','',table(clusters),'',
            'Synthetic components are recovered well, with a temporary ARI reduction in relocation-spanning windows. Fixed k=3 cannot exactly represent the seven ImageSegments labels. DenStream/Phishing degenerates to one effective cluster: ARI=0 and silhouette undefined. Low latency in that setting is not evidence of useful clustering. Snapshots show centers after training to that position; quality curves use historical pre-training assignments.','',
            '## Online AutoML','',
            'FixedHT is compared with epsilon-greedy Bandit, SuccessiveHalving and AutoClass on identical Elec2/SEA streams. Outer prediction/scoring precedes selector training; no full-stream hindsight oracle is used. Bandit and SH share five HT/HAT candidates. Bandit epsilon=0.1 and burn-in=100 arm pulls (20 all-arm observations). SH eta=2 has nominal budget 2n; its completed-rung counter is not a strict total-update cap, and actual updates are recorded. SH predictions remain in the active set.','',
            'AutoClass searches HT/HAT configurations with window 500, ensemble 8 and two new configurations on one core. The seeded Java overlay uses per-learner randomness; six formal prediction/configuration trajectories have matching fresh-process reproduction evidence. AutoML spaces and budgets differ, limiting pure strategy comparisons.','']
    auto=am.groupby(['dataset','model'],sort=False)[['accuracy','macro_f1','mean_latency_us','total_runtime_s']].mean().reset_index()
    lines+=[table(auto),'']
    for dataset,part in auto.groupby('dataset',sort=False):
        best=part.loc[part.accuracy.idxmax()];fast=part.loc[part.mean_latency_us.idxmin()]
        ref=cm[(cm.dataset==dataset)&(cm.library=='CapyMOA')&(cm.model=='AdaptiveRandomForest')]
        lines.append(f'- {dataset}: highest AutoML-study accuracy: {best.model} ({best.accuracy:.2%}); lowest latency: {fast.model} ({fast.mean_latency_us:.1f} µs). The separate fixed ARF reference reaches {ref.accuracy.mean():.2%} at {ref.mean_latency_us.mean():.1f} µs. Beating ordinary HT does not imply beating every strong fixed learner.')
    lines+=['','## Cost, validation and limits','',
            'The JVM heap limit is 4 GiB, ActiveProcessorCount=1, forest jobs=1 and BLAS threads=1. Initial construction, imports and cache loading are excluded from item latency. First-call/JIT costs remain. Classification latency includes conversion and preprocessing; clustering includes macro refresh but excludes fixed-array calibration/scaling. Evaluation enters total time. RSS includes the process/JVM and is sampled rather than model-exclusive.','',
            'Data hashes, per-run parameters, predictive hashes, selector states, logs and raw records are supplied in results/capymoa/. Classifier comparisons retain optimizer/backend differences. Real static order is not validated drift truth. Common k/scale, unequal search budgets, shared stream realizations and Java kernels without exposed seeds constrain interpretation. The executed notebooks and CSV exports provide the analysis outputs.']
    (folder/'findings.md').write_text('\n'.join(lines)+'\n')
