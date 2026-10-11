# Recorded experiment results

These values were reported in the original project documentation. Complete
evaluation output and Optuna logs are unavailable. A separately supplied
checkpoint has not been linked to these results; retraining does not recreate
the original experiment automatically.

## Training and metric definitions

The encoder is `dbmdz/bert-base-turkish-cased`. Defaults in
`configs/joint_bert.yaml` are maximum length 64, batch size 16, up to ten epochs,
AdamW learning rate `3e-5`, weight decay 0.01, warmup 15%, gradient clipping 1.0,
dropout 0.12 and seed 42. Normal training selects the lowest average validation
batch loss with patience three. AMP, class weights and label smoothing are off.
Intent loss plus 1.5 times slot loss trains the shared encoder and classifiers;
slot padding uses ignore index `-100`.

Intent evaluation reports accuracy and weighted precision/recall/F1, rather
than macro-F1. Slot evaluation uses entity-level `seqeval` precision/recall/F1
and secondary token accuracy excluding ignored labels. It uses token argmax,
not the API's entity decoder.

Optional Optuna search uses ten trials, four epochs per trial, batch size 16,
AMP off, warmup 0.1, TPE sampling and MedianPruner. Selection maximizes
validation entity-level slot F1. It saves `configs/joint_bert_best.yaml`.

## Recorded model values

| Version | Reported split | Intent weighted-F1 | Slot entity F1 | Slot precision | Slot recall |
|---|---|---|---|---|---|
| v1 | Test | 87.54% | 73.94% | 72.25% | 75.70% |
| v3 | Test | 87.99% | 74.16% | 72.34% | 76.06% |

Original v3 evaluation example additionally reports intent accuracy **88.03%** and slot token accuracy **90.54%**. Expected official test size is **2,974**. Reported changes: intent +0.45 percentage points; slot +0.22 points. No significance or confidence interval is recorded.

v1 is an earlier neural reference, not a classical baseline. The test set has
already been inspected; another run on it is a re-evaluation, not a new unseen
evaluation.

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

Individual trials do not establish that class weights always fail or that
dropout alone caused a score difference. The current objective is validation
slot F1, and optional label smoothing applies only to intent loss.
