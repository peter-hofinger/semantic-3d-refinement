# Training a network

Niessner's ladder (I2DL Section 6; "How to Start a Deep Learning Project", 2021) separates a wrong approach from a wrong implementation. When a result counts, and when a failure does, is set in `CLAUDE.md`. Items marked **(added)** are not from Niessner.

## Ladder

Climb in order. Do not skip a rung. Diagnose a failure at its own rung, not by changing things at a later one.

| # | Rung | Do | Pass | Failure means |
|---|------|----|------|---------------|
| 0 | Setup | Simplest model, under 1M params, about 1/5 of the layers you would normally start with. No dropout, BatchNorm or LR decay. Constant LR, no weight decay. | Runs end to end. | Fix the code. |
| 1 | Data | Visualize inputs, targets and augmentations. Check shapes, ranges, label alignment. | Samples look right to a human. | Data pipeline bug. |
| 2 | Speed | Time data loading and backprop separately. Estimate time to first pattern and to convergence. | Under 1 s per batch. Over 500 ms is a warning. | Fix the bottleneck first. Slow loops hide bugs. |
| 3 | Overfit 1 | Train on the bedroom scene `ai_006_007` only; one fixed, recorded view if a sample is a view. **(added)** | Loss reaches about 0 within minutes. Output matches the target. | Targets, loss or optimizer bug. Not the approach. |
| 4 | Overfit dev | Train on the `dev` scenes (3). | Loss reaches about 0. Outputs differ per input. | Input conditioning bug. The model ignores its inputs. |
| 5 | Generalize | Scale up on the train set: 5, 10, 100, 1000 samples. Validate on `val`. | Any val decrease, even slight, then val loss below the trivial baseline. | See verdicts below. |
| 6 | Push | Bigger model (needs more data), better architecture (e.g. adapters or LoRA inside the backbone **(added)**), LR schedule, class re-weighting, hyperparameter search. | Only after rung 5 passes and overfitting is handled. | |

Notes:

- Rungs 3 and 4 are diagnostics, never reported as a score. Val loss rises there because nothing general is learned yet.
- Loss about 0 is the bar for the simplest model on rungs 3 and 4, not for a constrained architecture on rung 6. **(added)**
- While scaling, check whether overfitting is plausible for the parameter count.
- After rung 5, re-enable removed features (BatchNorm, dropout, LR decay) one at a time.

## Verdicts

Read the train and val curves, then pick a row. Only address overfitting once train loss decreases.

| Symptom | Diagnosis | Next step |
|---------|-----------|-----------|
| Train loss does not fall (rung 3 or 4) | Implementation bug | Check targets, loss, and `zero_grad()` before `backward()`. |
| Train loss falls, val loss plateaus then rises | Severe overfitting | Smaller model → weight decay and other regularization → more data or augmentation → dropout. In that order. |
| Val keeps falling, small persistent gap | Moderate overfitting | Acceptable. Push (rung 6). |
| Curves overlap closely | No overfitting | If both are high: underfitting. Bigger model or longer training, not regularization. |
| Val better than train | Suspect a bug | Check leakage, train/eval mode, last-batch handling. |
| Train error and eval-mode error differ | Mode bug | Dropout or BatchNorm handled differently between modes. |

## Logging

- TensorBoard (`torch.utils.tensorboard`) only, event files under `runs/`. Log train loss every batch; after every iteration, run a forward pass on a random val batch and log it the same way.
- Log the val metrics too, not only the loss. Keep evaluation fixed so runs stay comparable.
- Run variants in parallel; after a few minutes, kill those without promise.
- Augmentation and class balancing come after rung 5, once data is the limit.
