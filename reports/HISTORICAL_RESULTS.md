# Historical results and provenance

Source: `README.md` at original commit `d7ccd95a86f91c80ecb6fddb91b8e870ff959c9c`. Evidence level: **reported in documentation**. No trained checkpoint, original checkpoint hash, full evaluation output or Optuna study log is present in this source snapshot. A separately supplied checkpoint has not been linked to these historical metrics.

## Recorded model values

| Version | Reported split | Intent weighted-F1 | Slot entity F1 | Slot precision | Slot recall |
|---|---|---|---|---|---|
| v1 | Test | 87.54% | 73.94% | 72.25% | 75.70% |
| v3 | Test | 87.99% | 74.16% | 72.34% | 76.06% |

Original v3 evaluation example additionally reports intent accuracy **88.03%** and slot token accuracy **90.54%**. Expected official test size is **2,974**. Reported changes: intent +0.45 percentage points; slot +0.22 points. No significance or confidence interval is recorded.

v1 is a prior neural reference. It is not automatically a classical baseline. The old README called the test fully unseen; it is now a historically inspected test set, and the original checkpoint/source lineage is incomplete.

## Recorded configuration changes

| Parameter | v1 → v3 | Original motivation, not an isolated causal finding |
|---|---|---|
| `lambda_slot` | 1.0 → 1.5 | Give slot loss more weight |
| `warmup_ratio` | 0.1 → 0.15 | Training stability |
| `dropout_rate` | 0.1 → 0.12 | Regularization |

Historical runtime estimates: normal training approximately **10–12 minutes**, Optuna approximately **40–50 minutes**, RTX 3050 laptop. They are historical estimates, not runtime guarantees.

## Recorded Optuna observations

| Trial | Validation slot F1 | Details recorded by the original README |
|---|---|---|
| 0 | 0.6875 (68.75%) | lr 1.8e-5, dropout 0.29, class weights none |
| 1 | 0.1793 | both class weights; described as collapse |
| 2 | 0.6404 | lr 1.4e-5, dropout 0.10 |
| 3–6 | Not reported | Pruned |

The study is described as 10 trials; details for the remaining trial IDs are not present. Missing values remain missing.

The current objective is maximum **validation entity-level slot F1**. The original README compared best trial 0.6875 with v1 test score 0.7394. Those are different partitions and cannot establish that search underperformed a comparable baseline.

The earlier explanations concerning high dropout, class weights and label smoothing are useful experiment notes, but cannot establish general causal conclusions. Label smoothing is applied only to intent loss by the current implementation. The earlier claim that Optuna optimized validation loss does not match this committed source. These terminology corrections leave every historical number unchanged.

## Available checkpoint

A separately supplied checkpoint came from source commit `4b72b6bf6ea585e1845084e53d36ee54a5af15a4`,
which differs from the published source `d7ccd95a86f91c80ecb6fddb91b8e870ff959c9c`.
Its identity does not establish that it produced the historical scores above.
The matching configuration and label mappings have these SHA-256 identities:

| File | SHA-256 |
|---|---|
| `best_joint_nlu_model.pt` | `daf393caece62ca84756ed09059c04a12f16573318e0faad42006010597a8ff6` |
| `configs/joint_bert.yaml` | `48eea0ff7a4b736aa3c3e2205237fbcfa6239e5a3d98a90e8359cbe3a4ebe886` |
| `intent_to_id.json` | `ca0842aacaf75c4f426559ee88c563560f905f0ccbb03cb14d08b2f4a33a19c6` |
| `slot_to_id.json` | `b2c260b2addb43b2a18c7a62b1b31727e7a7e19a153ca7b690d9a3f1ff7e0dca` |

Original run lineage, environment, complete metric output and full Optuna logs
are still needed to link the historical claims to a run. Retraining does not
restore the original artifact automatically.
