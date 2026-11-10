# River: measured results and analysis

168 model runs and 72 detector-only replays use model seeds 42, 43 and 44, with synthetic generation seed 2026. Each observation is predicted and scored before detection/reset and learning. All methods use the same ordered caches.

## Setup and scope

Regression families are an ordinary Hoeffding tree, a five-tree adaptive random forest, scaled Adam linear regression and scaled KNN (k=5, FIFO window 300). Classification uses the tree, forest, scaled one-vs-rest Adam logistic regression and Gaussian naive Bayes. The Part 1 trees have depth limit 12. Input scaling and regression target scaling update online; categorical encoding is stateless.

Bikes and Elec2 use their first 10,000 observations. Smaller real datasets are used in full; both synthetic streams contain 10,000 examples. Phishing and ImageSegments are static benchmarks without verified chronological order. The three model seeds share one stream realization: their SDs are not independent-stream confidence intervals.

## Regression performance

| dataset | model | mae | rmse | r2 | mean_latency_us | total_runtime_s |
| --- | --- | --- | --- | --- | --- | --- |
| TrumpApproval | HoeffdingTree | 1.5676 | 2.2029 | -0.6585 | 231.5967 | 0.2396 |
| TrumpApproval | AdaptiveRandomForest | 0.7700 | 1.8216 | -0.1541 | 375.0721 | 0.3870 |
| TrumpApproval | LinearAdam | 0.4635 | 1.5000 | 0.2310 | 27.0338 | 0.0342 |
| TrumpApproval | KNN | 0.5732 | 1.5527 | 0.1760 | 109.2790 | 0.1180 |
| ChickWeights | HoeffdingTree | 43.3367 | 64.3950 | 0.1776 | 41.5457 | 0.0280 |
| ChickWeights | AdaptiveRandomForest | 34.1241 | 52.5184 | 0.4529 | 195.7920 | 0.1195 |
| ChickWeights | LinearAdam | 23.5474 | 38.3797 | 0.7079 | 39.6085 | 0.0271 |
| ChickWeights | KNN | 31.6657 | 53.0822 | 0.4412 | 55.3277 | 0.0358 |
| Bikes | HoeffdingTree | 5.7878 | 7.4937 | 0.0306 | 46.9756 | 0.5444 |
| Bikes | AdaptiveRandomForest | 4.3955 | 5.9012 | 0.3987 | 240.2791 | 2.5254 |
| Bikes | LinearAdam | 4.6185 | 6.4336 | 0.2855 | 42.0985 | 0.5002 |
| Bikes | KNN | 5.0774 | 7.5720 | 0.0103 | 222.3014 | 2.3141 |
| FriedmanDrift | HoeffdingTree | 3.0577 | 3.8409 | 0.4030 | 63.6656 | 0.7089 |
| FriedmanDrift | AdaptiveRandomForest | 2.9220 | 3.6490 | 0.4611 | 430.1993 | 4.4243 |
| FriedmanDrift | LinearAdam | 2.3880 | 3.0644 | 0.6200 | 27.7467 | 0.3459 |
| FriedmanDrift | KNN | 2.5032 | 3.1399 | 0.6011 | 115.4009 | 1.2399 |

- TrumpApproval: best mean mae: LinearAdam (0.4635); fastest: LinearAdam (27.03 µs/item). The best-quality model costs 1.00 times the lowest mean latency.
- ChickWeights: best mean mae: LinearAdam (23.5474); fastest: LinearAdam (39.61 µs/item). The best-quality model costs 1.00 times the lowest mean latency.
- Bikes: best mean mae: AdaptiveRandomForest (4.3955); fastest: LinearAdam (42.10 µs/item). The best-quality model costs 5.71 times the lowest mean latency.
- FriedmanDrift: best mean mae: LinearAdam (2.3880); fastest: LinearAdam (27.75 µs/item). The best-quality model costs 1.00 times the lowest mean latency.

## Classification performance

| dataset | model | accuracy | macro_f1 | kappa | mean_latency_us | total_runtime_s |
| --- | --- | --- | --- | --- | --- | --- |
| Elec2 | HoeffdingTree | 0.8545 | 0.8511 | 0.7023 | 53.0026 | 0.5989 |
| Elec2 | AdaptiveRandomForest | 0.8758 | 0.8731 | 0.7462 | 358.4207 | 3.6661 |
| Elec2 | LogisticAdamOvR | 0.8508 | 0.8471 | 0.6943 | 45.1398 | 0.5148 |
| Elec2 | GaussianNB | 0.7940 | 0.7802 | 0.5667 | 132.9459 | 1.4128 |
| Phishing | HoeffdingTree | 0.8752 | 0.8734 | 0.7468 | 52.5289 | 0.0733 |
| Phishing | AdaptiveRandomForest | 0.8899 | 0.8880 | 0.7759 | 337.0464 | 0.4309 |
| Phishing | LogisticAdamOvR | 0.8984 | 0.8973 | 0.7948 | 50.1982 | 0.0709 |
| Phishing | GaussianNB | 0.8840 | 0.8828 | 0.7656 | 141.6537 | 0.1874 |
| ImageSegments | HoeffdingTree | 0.7779 | 0.7664 | 0.7409 | 212.3812 | 0.5079 |
| ImageSegments | AdaptiveRandomForest | 0.7491 | 0.7465 | 0.7072 | 604.6413 | 1.4167 |
| ImageSegments | LogisticAdamOvR | 0.8203 | 0.8039 | 0.7904 | 163.2426 | 0.3939 |
| ImageSegments | GaussianNB | 0.7333 | 0.7318 | 0.6889 | 234.9087 | 0.5639 |
| SEARecurring | HoeffdingTree | 0.6146 | 0.6119 | 0.2240 | 32.9978 | 0.3934 |
| SEARecurring | AdaptiveRandomForest | 0.9015 | 0.9004 | 0.8008 | 308.7545 | 3.1665 |
| SEARecurring | LogisticAdamOvR | 0.8722 | 0.8711 | 0.7422 | 25.6507 | 0.3139 |
| SEARecurring | GaussianNB | 0.6343 | 0.6216 | 0.2488 | 122.5585 | 1.3070 |

- Elec2: best mean accuracy: AdaptiveRandomForest (0.8758); fastest: LogisticAdamOvR (45.14 µs/item). The best-quality model costs 7.94 times the lowest mean latency.
- Phishing: best mean accuracy: LogisticAdamOvR (0.8984); fastest: LogisticAdamOvR (50.20 µs/item). The best-quality model costs 1.00 times the lowest mean latency.
- ImageSegments: best mean accuracy: LogisticAdamOvR (0.8203); fastest: LogisticAdamOvR (163.24 µs/item). The best-quality model costs 1.00 times the lowest mean latency.
- SEARecurring: best mean accuracy: AdaptiveRandomForest (0.9015); fastest: LogisticAdamOvR (25.65 µs/item). The best-quality model costs 12.04 times the lowest mean latency.

## Drift detection and handling

ADWIN, Page-Hinkley and KSWIN replay the same ordinary-tree regression error signal; ADWIN, DDM and EDDM replay its classification error. Regression errors are normalized using past-target SD and clipped to [0,1]. Classification uses 0/1 error. The first 200 samples are scored/learned but withheld from external detectors.

Both synthetic changes occur at zero-based indices 3333 and 6666. FriedmanDrift uses the global recurring abrupt concept; SEARecurring inverts then restores labels with 5% noise. Each change matches the first unused alarm within 1500 index steps. Missed changes do not receive zero delay; other alarms remain unmatched rather than verified false positives.

| dataset | detector | n_alarms | matched_changes | missed_changes | unmatched_alarms | mean_delay |
| --- | --- | --- | --- | --- | --- | --- |
| FriedmanDrift | ADWIN | 3.0000 | 2.0000 | 0.0000 | 1.0000 | 335.5000 |
| FriedmanDrift | PageHinkley | 4.0000 | 1.0000 | 1.0000 | 3.0000 | 137.0000 |
| FriedmanDrift | KSWIN | 12.0000 | 2.0000 | 0.0000 | 10.0000 | 730.5000 |
| SEARecurring | ADWIN | 6.0000 | 2.0000 | 0.0000 | 4.0000 | 31.5000 |
| SEARecurring | DDM | 5.0000 | 2.0000 | 0.0000 | 3.0000 | 51.0000 |
| SEARecurring | EDDM | 14.0000 | 2.0000 | 0.0000 | 12.0000 | 115.0000 |

ADWIN matches both changes on both streams. Page-Hinkley has a shorter successful Friedman delay but misses the other change. KSWIN and EDDM produce more unmatched alarms here. Static real datasets provide no verified drift truth.

Reset runs change only whether an external alarm replaces the same ordinary tree after the current prediction is scored. The replacement learns the current observation once. Reset errors can change later alarm positions. ARF already adapts internally and is not used as this controlled reset baseline.

Reset minus no-reset differences: negative MAE changes and positive accuracy changes are improvements. Accuracy is a fraction in this table.

| task | dataset | detector_reset | mae_change | accuracy_change | latency_ratio |
| --- | --- | --- | --- | --- | --- |
| regression | TrumpApproval | ADWIN | -0.4191 | — | 0.4877 |
| regression | TrumpApproval | PageHinkley | -0.4188 | — | 0.4394 |
| regression | TrumpApproval | KSWIN | -0.5816 | — | 0.4279 |
| regression | ChickWeights | ADWIN | 0.0000 | — | 1.1074 |
| regression | ChickWeights | PageHinkley | 0.0000 | — | 1.0515 |
| regression | ChickWeights | KSWIN | -12.8811 | — | 3.8685 |
| regression | Bikes | ADWIN | -1.6244 | — | 0.7777 |
| regression | Bikes | PageHinkley | -1.7457 | — | 0.7628 |
| regression | Bikes | KSWIN | -1.4974 | — | 2.4996 |
| regression | FriedmanDrift | ADWIN | 0.2594 | — | 0.9691 |
| regression | FriedmanDrift | PageHinkley | 0.5248 | — | 0.9784 |
| regression | FriedmanDrift | KSWIN | 0.4734 | — | 5.8687 |
| classification | Elec2 | ADWIN | — | 0.0249 | 0.9684 |
| classification | Elec2 | DDM | — | 0.0380 | 0.9920 |
| classification | Elec2 | EDDM | — | 0.0308 | 0.9972 |
| classification | Phishing | ADWIN | — | 0.0000 | 1.0351 |
| classification | Phishing | DDM | — | 0.0000 | 1.0563 |
| classification | Phishing | EDDM | — | -0.0048 | 1.1103 |
| classification | ImageSegments | ADWIN | — | 0.0000 | 0.9982 |
| classification | ImageSegments | DDM | — | 0.0000 | 1.0073 |
| classification | ImageSegments | EDDM | — | -0.0152 | 1.0144 |
| classification | SEARecurring | ADWIN | — | 0.2887 | 0.9645 |
| classification | SEARecurring | DDM | — | 0.2872 | 0.9922 |
| classification | SEARecurring | EDDM | — | 0.2718 | 0.9660 |

- FriedmanDrift: ADWIN reset changes mae from 3.0577 to 3.3171.
- SEARecurring: ADWIN reset changes accuracy from 0.6146 to 0.9033.

## Interpretation and limits

Full replacement worsens cumulative Friedman error but substantially improves recurring SEA classification. An alarm is not sufficient evidence that reset improves prediction. Relearning cost and loss of useful recurring knowledge are possible explanations, not isolated causal findings.

MAE units differ across datasets, so raw MAE is not averaged across tasks. Negative R² means squared error exceeds the target-mean evaluation reference; that reference is not used for training. Macro-F1 and kappa complement accuracy.

Latency includes prediction, online model transforms and learning; reset runs also include detection and reconstruction. Total loop time additionally includes evaluation/checkpoints. Data preparation and initial construction are excluded. Detector-only replay costs are separate. RSS samples cover the process and data, not model-exclusive or continuously measured peak memory.

The supplied order, modest fixed settings, sample prefixes and shared stream realization limit generalization. Detailed alarms are in drift_events.csv, matching in drift_summary.csv, recovery windows in recovery_windows.csv, and per-run parameters/records in runs/.

Configuration: config/river.json. Data scope and input hashes: data/manifest.json. Runtime and timing definitions: results/river/experiment.json. Dependencies: requirements.txt. The executed notebooks and CSV exports provide the analysis outputs.
