# Implementation fixes and validation

The submitted results use the corrected implementation. The earlier technical audit led to a full recomputation of 315 model runs and 72 detector replays. Twelve protocol/regression tests passed; 900 AutoML state checkpoints and six complete formal AutoClass reproduction cases have matching evidence.

| Issue | Implemented correction |
| --- | --- |
| Unseeded AutoClass randomness | A narrow Java overlay redirects six Math.random calls and one unseeded Random constructor to explicit per-learner sources; one core and serialized calls are used |
| Candidate scores overwritten by model names | Save unique candidate IDs, indices, parameters, scores and update counts |
| Missing AutoClass configuration state | Save ensemble/search settings, performance, selected slot and seed |
| SH could predict an eliminated candidate | Check the active set after elimination and before prediction |
| Unknown names or unsupported depth silently accepted | Explicit allowlists and fail-fast configuration validation |
| Stale or altered cache reuse | Runtime/source/input identities, task checks and record content seals |
| CSV/JSON disagreement undetected | Compare exported metrics, curves and windows with sealed raw evidence |
| Deprecated River rolling API | Use the supported class-construction interface without changing scoring |

Relative to earlier results, 309 predictive records were unchanged; the six AutoClass records changed. Costs were measured again and are not interpreted as a causal speed effect of the fixes. Earlier archives are excluded from this current-only distribution.

Bandit burn-in=100 counts arm pulls, equivalent to 20 all-candidate observations with five arms. SH's nominal budget is not a strict total-update cap; actual updates are recorded. These semantics are disclosed rather than hidden by retuning.

The Java source, overlay and source/upstream checksums are in `vendor/`; build with `scripts/build_moa_patch.py`. Restart Python/Jupyter after classpath changes. `capy_lab/automl_adapters.py`, `capy_lab/configuration.py` and `lab_support/` implement the state/configuration/evidence checks.

Evidence: `results/fixes/tests.log`, `preflight_reproduction.json`, `formal_reproduction.json` and `validation.json`. Use `scripts/verify_submission_payload.py` to check the distributed snapshot, and the 12 core tests in `tests/` to exercise protocol edges (four additional tests check notebook entry points). These are targeted checks, not an exhaustive proof. Fresh installation on arbitrary platforms has not been tested.

The Adam/SGD mismatch, ordered data prefixes, static benchmark order, common clustering macro stage, unequal AutoML spaces/budgets, sampled RSS and first-call/JIT timing limitations remain explicitly disclosed.

The English edition changes presentation only: computational modules, data/configuration, dependency locks and all sealed model records retain their original bytes. Analysis prose and notebook descriptions are localized separately.

The subsequent public-path cleanup redacts environment/log paths and the
parameter-description field of six AutoClass records. Only those six content
seals are recomputed; predictions, metrics, costs, selector states, runtime
identities and training sources are unchanged. Two additional tests verify
redaction and refusal to modify a record with an invalid seal, bringing the
current test total to 18.
