# Experiments

Answers two questions:

1. Is the approach wrong, or is my implementation wrong?
2. Is "does not work" a finding, or just "not working yet"?

Based on Niessner's I2DL Section 6 and "How to Start a Deep Learning Project" (2021). Items marked **(added)** are not from Niessner.

## Sample unit **(added)**

Overfit rungs use fixed, named data. Scale rungs use the train and val sets.

- If the algorithm takes a whole scene as one sample: use scenes.
- If it takes a single view as one sample: pick one fixed view per scene. Record which one.

## Splits **(added)**

Splits are those of `configs/hypersim.yaml`. Wherever this document says val, it means Hypersim's `val` split: fixed, never trained on, never augmented.

- Rungs 3 and 4 train on `ai_006_007` and on `dev`. They are diagnostics, not results, and nothing from them is reported as a score.
- Rung 5 trains on the train scenes and validates on `val`. Choices (early stopping, model selection, hyperparameters) are made on `val` and `dev`.
- `test` is only reported, once, after the choices are frozen. Never choose anything on it.

## Ladder

Climb in order. Do not skip a rung. Diagnose a failure at its own rung, not by changing things at a later one.

| # | Rung | Do | Pass | Failure means |
|---|------|----|------|---------------|
| 0 | Setup | Simplest model, under 1M params, about 1/5 of the layers you would normally start with. No dropout, BatchNorm or LR decay. Constant LR, no weight decay. | Runs end to end. | Fix the code. |
| 1 | Data | Visualize inputs, targets and augmentations. Check shapes, ranges, label alignment. | Samples look right to a human. | Data pipeline bug. |
| 2 | Speed | Time data loading and backprop separately. Estimate time to first pattern and to convergence. | Under 1 s per batch. Over 500 ms is a warning. | Fix the bottleneck first. Slow loops hide bugs. |
| 3 | Overfit 1 | Train on the bedroom scene `ai_006_007` only. | Loss reaches about 0 within minutes. Output matches the target. | Targets, loss or optimizer bug. Not the approach. |
| 4 | Overfit dev | Train on the `dev` scenes (3). | Loss reaches about 0. Outputs differ per input. | Input conditioning bug. The model ignores its inputs. |
| 5 | Generalize | Scale up on the train set: 5, 10, 100, 1000 samples. Validate on `val` (see Splits). | Any val decrease, even slight, then val loss below the trivial baseline. | See verdicts below. |
| 6 | Push | Bigger model (needs more data), better architecture, LR schedule, class re-weighting, hyperparameter search. | Only after rung 5 passes and overfitting is handled. | |

Notes:

- Rungs 3 and 4 have no meaningful val result. Val loss rises there because nothing general is learned yet. Expected.
- While scaling, check whether overfitting is plausible for the parameter count.
- After rung 5, re-enable removed features (BatchNorm, dropout, LR decay) one at a time.
- Larger architectures (ResNet, Inception, Xception variants) come last.

## Verdicts

Read the train and val curves, then pick a row.

| Symptom | Diagnosis | Next step |
|---------|-----------|-----------|
| Train loss does not fall (rung 3 or 4) | Implementation bug | Run the bug checklist. |
| Train loss falls, val loss plateaus then rises | Severe overfitting | Smaller model → weight decay and other regularization → more data or augmentation → dropout. In that order. |
| Val keeps falling, small persistent gap | Moderate overfitting | Acceptable. Push (rung 6). |
| Curves overlap closely | No overfitting | If both are high: underfitting. Bigger model or longer training, not regularization. |
| Val better than train | Suspect a bug | Check leakage, train/eval mode, last-batch handling. |
| Train error and eval-mode error differ | Mode bug | Dropout or BatchNorm handled differently between modes. |

Only address overfitting once train loss decreases.

## Claiming a negative result **(added)**

An experiment cannot prove an approach impossible. It can show that the approach failed under stated conditions. Claim "approach fails" only when all of these hold:

- Rungs 0–4 passed on this code. The pipeline can learn.
- A simple baseline (L1 regression, a truncated ResNet-18, or the trivial predictor) ran with the same data, split and budget. Papers sell a point. A simple baseline is often as good.
- The approach was tuned at least as much as the baseline.
- The result held across 3 or more seeds.
- The budget is written down: data size, steps, hyperparameters tried.

If any item is missing, write "not yet working" and name the missing item.

## Bug checklist

Run when a rung fails. Common causes first.

- Overfitting one batch skipped.
- `model.train()` and `model.eval()` mixed up or forgotten.
- `optimizer.zero_grad()` missing before `backward()`.
- Softmax applied before a loss that expects logits.
- Wrong `bias`: missing on the output layer, or misused with BatchNorm.
- Val or test data in the training set.
- Algorithm debugged on test data.
- Last, smaller batch of an epoch mishandled.
- Val set augmented.
- More than one thing changed between runs.

## Running experiments

- Plot train and val loss (TensorBoard or equivalent). Log loss every batch.
- After every iteration, run a forward pass on a random val batch.
- Plot the val metrics too (mIoU, accuracy, F1), not only the loss.
- Keep evaluation consistent so runs stay comparable. Change one thing at a time.
- Run ablations in parallel. After a few minutes, kill runs without promise.
- Keep debug cycles to minutes.
- Never augment the val set. Class balancing and augmentation matter most once data is the limit, after rung 5.
- Generative models: start with L1 regression and no discriminator. Add one later (Wasserstein is a good choice).
