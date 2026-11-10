# CapyMOA: measured results and analysis

Steps 7–10 include 96 paired classification runs, 27 stream-clustering runs and 24 AutoML runs. Formal runs are sequential and use a fresh process for each model run. Seeds 42, 43 and 44 share one cached stream per dataset. Dependencies are pinned separately from Part 1.

## Controlled cross-library classification

Both libraries receive identical dense vectors with absent one-hot entries filled by zero. Shared settings include grace=100, memory parameter=64 MiB, NBA leaves, five forest trees, lambda=6, floor(sqrt(d)) features and ADWIN drift/warning deltas 0.002/0.01. Both profiles omit an explicit tree-depth cap because MOA has no matching option.

Linear models share causal scaling, but River Adam(lr=0.01) and CapyMOA inverse-scaling SGD(eta0=0.01, log_loss, no penalty) are different optimizers. Their differences cannot be attributed solely to the library. Backend split statistics, variance handling and random-number implementations also differ. Prediction objects contain a fixed zero class placeholder; true labels enter scoring and subsequent training.

Part 1 uses sparse inputs and tree depth 12. Part 2 reruns River with the aligned profile; the two River result tables must not be treated as an identical configuration.

| dataset | model | library | accuracy | macro_f1 | kappa | mean_latency_us | total_runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Elec2 | HoeffdingTree | River | 0.8394 | 0.8344 | 0.6694 | 80.2897 | 0.8623 |
| Elec2 | HoeffdingTree | CapyMOA | 0.8450 | 0.8402 | 0.6810 | 82.1077 | 0.8896 |
| Elec2 | AdaptiveRandomForest | CapyMOA | 0.8640 | 0.8610 | 0.7221 | 198.3995 | 2.0658 |
| Elec2 | AdaptiveRandomForest | River | 0.8465 | 0.8421 | 0.6847 | 418.6277 | 4.2634 |
| Elec2 | LinearLogistic | River | 0.8575 | 0.8542 | 0.7085 | 62.2236 | 0.6815 |
| Elec2 | LinearLogistic | CapyMOA | 0.7371 | 0.7009 | 0.4315 | 933.2891 | 9.4238 |
| Elec2 | GaussianNB | CapyMOA | 0.7720 | 0.7588 | 0.5227 | 57.5128 | 0.6366 |
| Elec2 | GaussianNB | River | 0.7719 | 0.7588 | 0.5226 | 157.4374 | 1.6449 |
| Phishing | HoeffdingTree | River | 0.8752 | 0.8734 | 0.7468 | 67.4644 | 0.0915 |
| Phishing | HoeffdingTree | CapyMOA | 0.8816 | 0.8803 | 0.7608 | 105.8447 | 0.1409 |
| Phishing | AdaptiveRandomForest | CapyMOA | 0.8987 | 0.8970 | 0.7940 | 228.1436 | 0.2953 |
| Phishing | AdaptiveRandomForest | River | 0.8899 | 0.8880 | 0.7759 | 371.2727 | 0.4735 |
| Phishing | LinearLogistic | River | 0.8984 | 0.8973 | 0.7948 | 53.9166 | 0.0748 |
| Phishing | LinearLogistic | CapyMOA | 0.8672 | 0.8662 | 0.7327 | 980.5134 | 1.2374 |
| Phishing | GaussianNB | CapyMOA | 0.8840 | 0.8826 | 0.7652 | 66.1978 | 0.0912 |
| Phishing | GaussianNB | River | 0.8840 | 0.8828 | 0.7656 | 147.1532 | 0.1926 |
| ImageSegments | HoeffdingTree | River | 0.7779 | 0.7664 | 0.7409 | 325.1881 | 0.7696 |
| ImageSegments | HoeffdingTree | CapyMOA | 0.7784 | 0.7680 | 0.7414 | 97.5584 | 0.2417 |
| ImageSegments | AdaptiveRandomForest | CapyMOA | 0.6834 | 0.6734 | 0.6306 | 230.3001 | 0.5514 |
| ImageSegments | AdaptiveRandomForest | River | 0.7491 | 0.7465 | 0.7072 | 730.6435 | 1.7070 |
| ImageSegments | LinearLogistic | River | 0.8203 | 0.8039 | 0.7904 | 168.9017 | 0.4068 |
| ImageSegments | LinearLogistic | CapyMOA | 0.6840 | 0.6453 | 0.6313 | 2855.2403 | 6.6189 |
| ImageSegments | GaussianNB | CapyMOA | 0.7788 | 0.7686 | 0.7419 | 70.1726 | 0.1782 |
| ImageSegments | GaussianNB | River | 0.7333 | 0.7318 | 0.6889 | 295.0696 | 0.7015 |
| SEARecurring | HoeffdingTree | River | 0.6146 | 0.6119 | 0.2240 | 44.9213 | 0.5086 |
| SEARecurring | HoeffdingTree | CapyMOA | 0.6153 | 0.6125 | 0.2252 | 76.7925 | 0.8560 |
| SEARecurring | AdaptiveRandomForest | CapyMOA | 0.8957 | 0.8946 | 0.7892 | 173.0693 | 1.8203 |
| SEARecurring | AdaptiveRandomForest | River | 0.8836 | 0.8823 | 0.7646 | 380.1403 | 3.8852 |
| SEARecurring | LinearLogistic | River | 0.8722 | 0.8711 | 0.7422 | 44.8906 | 0.5088 |
| SEARecurring | LinearLogistic | CapyMOA | 0.6393 | 0.6253 | 0.2575 | 988.3997 | 9.9796 |
| SEARecurring | GaussianNB | CapyMOA | 0.6344 | 0.6217 | 0.2489 | 59.4713 | 0.6611 |
| SEARecurring | GaussianNB | River | 0.6343 | 0.6216 | 0.2488 | 137.1619 | 1.4457 |

Accuracy differences below are CapyMOA minus River in percentage points. A latency ratio below one favors CapyMOA under these interfaces and settings, not necessarily its isolated kernel.

| dataset | model | accuracy_difference_pp | capy_to_river_latency_ratio |
| --- | --- | --- | --- |
| Elec2 | HoeffdingTree | 0.5600 | 1.0227 |
| Elec2 | AdaptiveRandomForest | 1.7467 | 0.4742 |
| Elec2 | LinearLogistic | -12.0400 | 15.0026 |
| Elec2 | GaussianNB | 0.0100 | 0.3657 |
| Phishing | HoeffdingTree | 0.6400 | 1.5690 |
| Phishing | AdaptiveRandomForest | 0.8800 | 0.6138 |
| Phishing | LinearLogistic | -3.1200 | 18.1929 |
| Phishing | GaussianNB | 0.0000 | 0.4493 |
| ImageSegments | HoeffdingTree | 0.0433 | 0.3030 |
| ImageSegments | AdaptiveRandomForest | -6.5657 | 0.3169 |
| ImageSegments | LinearLogistic | -13.6364 | 17.0017 |
| ImageSegments | GaussianNB | 4.5455 | 0.2406 |
| SEARecurring | HoeffdingTree | 0.0700 | 1.7127 |
| SEARecurring | AdaptiveRandomForest | 1.2067 | 0.4553 |
| SEARecurring | LinearLogistic | -23.2900 | 22.0231 |
| SEARecurring | GaussianNB | 0.0100 | 0.4338 |

- Elec2: highest controlled mean accuracy: CapyMOA / AdaptiveRandomForest, 86.40%.
- Phishing: highest controlled mean accuracy: CapyMOA / AdaptiveRandomForest, 89.87%.
- ImageSegments: highest controlled mean accuracy: River / LinearLogistic, 82.03%.
- SEARecurring: highest controlled mean accuracy: CapyMOA / AdaptiveRandomForest, 89.57%.

## Unsupervised stream clustering

CluStream, ClusTree and DenStream receive features from ImageSegments, Phishing and the 6,000-point MovingGaussians stream (changes at 2000/4000). Reference labels/components are reserved for ARI and never used to train or tune the learners. A fixed class-slot zero avoids a ClusTree missing-value problem.

CluStream uses horizon 1000 and 50 kernels; ClusTree uses horizon 1000, height 5 and depth-first insertion. DenStream uses horizon 1000, epsilon 0.5, beta 0.2, mu 10, init 100, lambda 0.25 and speed 100. A common weighted KMeans macro stage targets k=3 and refreshes every 200 observations; Hungarian matching stabilizes IDs. DenStream native DBSCAN macro/noise output is not evaluated.

The first 200 unlabeled points calibrate and initialize the learners. Scaling then freezes and distances are normalized by sqrt(d). Subsequent points are assigned using previous centers before training. Window scores are recorded every 500 points; silhouette samples at most 200 assigned points. Coverage is explicit and invalid silhouette remains missing.

| dataset | model | mean_window_ari | mean_window_silhouette | mean_window_coverage | mean_latency_us |
| --- | --- | --- | --- | --- | --- |
| ImageSegments | CluStream | 0.1233 | 0.2690 | 1.0000 | 217.0633 |
| ImageSegments | ClusTree | 0.1663 | 0.3234 | 1.0000 | 108.4035 |
| ImageSegments | DenStream | 0.1987 | 0.2879 | 1.0000 | 61.7026 |
| Phishing | CluStream | 0.2516 | 0.1694 | 1.0000 | 143.3476 |
| Phishing | ClusTree | 0.2822 | 0.1653 | 1.0000 | 121.3712 |
| Phishing | DenStream | 0.0000 | — | 1.0000 | 58.5228 |
| MovingGaussians | CluStream | 0.9602 | 0.8229 | 1.0000 | 124.4789 |
| MovingGaussians | ClusTree | 0.9619 | 0.8243 | 1.0000 | 91.7862 |
| MovingGaussians | DenStream | 0.9397 | 0.8064 | 1.0000 | 82.1779 |

Synthetic components are recovered well, with a temporary ARI reduction in relocation-spanning windows. Fixed k=3 cannot exactly represent the seven ImageSegments labels. DenStream/Phishing degenerates to one effective cluster: ARI=0 and silhouette undefined. Low latency in that setting is not evidence of useful clustering. Snapshots show centers after training to that position; quality curves use historical pre-training assignments.

## Online AutoML

FixedHT is compared with epsilon-greedy Bandit, SuccessiveHalving and AutoClass on identical Elec2/SEA streams. Outer prediction/scoring precedes selector training; no full-stream hindsight oracle is used. Bandit and SH share five HT/HAT candidates. Bandit epsilon=0.1 and burn-in=100 arm pulls (20 all-arm observations). SH eta=2 has nominal budget 2n; its completed-rung counter is not a strict total-update cap, and actual updates are recorded. SH predictions remain in the active set.

AutoClass searches HT/HAT configurations with window 500, ensemble 8 and two new configurations on one core. The seeded Java overlay uses per-learner randomness; six formal prediction/configuration trajectories have matching fresh-process reproduction evidence. AutoML spaces and budgets differ, limiting pure strategy comparisons.

| dataset | model | accuracy | macro_f1 | mean_latency_us | total_runtime_s |
| --- | --- | --- | --- | --- | --- |
| Elec2 | FixedHT | 0.8450 | 0.8402 | 89.5147 | 0.9723 |
| Elec2 | Bandit | 0.8793 | 0.8772 | 4852.4594 | 49.0649 |
| Elec2 | SuccessiveHalving | 0.8846 | 0.8826 | 7661.7100 | 77.2020 |
| Elec2 | AutoClass | 0.8833 | 0.8810 | 285.2872 | 2.9814 |
| SEARecurring | FixedHT | 0.6153 | 0.6125 | 71.1778 | 0.8025 |
| SEARecurring | Bandit | 0.6534 | 0.6506 | 4981.0691 | 50.3102 |
| SEARecurring | SuccessiveHalving | 0.6463 | 0.6398 | 7211.5912 | 72.6395 |
| SEARecurring | AutoClass | 0.8656 | 0.8643 | 179.4180 | 1.9098 |

- Elec2: highest AutoML-study accuracy: SuccessiveHalving (88.46%); lowest latency: FixedHT (89.5 µs). The separate fixed ARF reference reaches 86.40% at 198.4 µs. Beating ordinary HT does not imply beating every strong fixed learner.
- SEARecurring: highest AutoML-study accuracy: AutoClass (86.56%); lowest latency: FixedHT (71.2 µs). The separate fixed ARF reference reaches 89.57% at 173.1 µs. Beating ordinary HT does not imply beating every strong fixed learner.

## Cost, validation and limits

The JVM heap limit is 4 GiB, ActiveProcessorCount=1, forest jobs=1 and BLAS threads=1. Initial construction, imports and cache loading are excluded from item latency. First-call/JIT costs remain. Classification latency includes conversion and preprocessing; clustering includes macro refresh but excludes fixed-array calibration/scaling. Evaluation enters total time. RSS includes the process/JVM and is sampled rather than model-exclusive.

Data hashes, per-run parameters, predictive hashes, selector states, logs and raw records are supplied in results/capymoa/. Classifier comparisons retain optimizer/backend differences. Real static order is not validated drift truth. Common k/scale, unequal search budgets, shared stream realizations and Java kernels without exposed seeds constrain interpretation. The executed notebooks and CSV exports provide the analysis outputs.
