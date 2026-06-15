# Week 3 — CNNs, the Training Loop & Evaluation

> The headline week. We build a **Convolutional Neural Network (CNN)** that looks at a galaxy *as an image* (not a flattened vector), write the **training loop** that actually teaches it, then learn to **evaluate and interpret** the result — including a confusion matrix that tells us, astrophysically, where the model gets confused. By the end you have a trained, saved galaxy classifier.

This week is split into **two parts**:

- **Part 1 — The training loop & CNNs:** `nn.Conv2d` and `nn.MaxPool2d`, assembling a `GalaxyCNN`, and the five-step training loop run over several epochs — plus the astronomy of spiral density waves, dust lanes, and H II regions.
- **Part 2 — Validation & interpretation:** `model.eval()` + `torch.no_grad()`, train-vs-validation curves, the confusion matrix, and saving weights — plus the astronomy of lenticulars, mergers, and galaxy evolution.

---

## At a Glance

- **Time estimate:** 5–7 hours (the biggest week).
- **Prerequisites:** Finish [Week 1](../Week-1/) and [Week 2](../Week-2/) first. You need a working `DataLoader` of galaxy images, comfort with `nn.Module`, and Week 2's loss + optimiser.
- **Deliverable:** Two completed Colab notebooks — `week3_cnn_starter.ipynb` (a CNN trained with a clearly decreasing loss curve) and `week3_eval_starter.ipynb` (train/val curves, a test accuracy beating the Week-2 baseline, a labelled confusion matrix, and a saved `galaxy_model.pth`).

---

## Learning Objectives

| ML / Computer Science | Astronomy / Physics |
|-----------------------|---------------------|
| Explain why flattening destroys spatial structure and convolution preserves it. | Explain the **winding problem** and why solid spiral arms can't exist. |
| Use **`nn.Conv2d`** (channels, kernel size, stride, padding) and read its output shape. | Describe **density-wave theory** with the traffic-jam analogy. |
| Use **`nn.MaxPool2d`** to downsample and grow the receptive field. | Connect compression in the wave to **star formation** along arms. |
| Assemble a **`GalaxyCNN`** (`Conv→ReLU→Pool` blocks + a `Linear` head) and track shapes. | Identify **dust lanes** and **H II regions** and what they signal. |
| Write the **five-step training loop** and run it for several **epochs**. | Explain what makes **lenticular (S0)** galaxies ambiguous to classify. |
| Evaluate with **`model.eval()`** and **`torch.no_grad()`**; compute accuracy. | Describe how **mergers** produce irregular morphology and starbursts. |
| Spot **overfitting** from train-vs-validation loss curves and name fixes. | Argue why galaxy morphology **evolves** (and why "early/late-type" misleads). |
| Build and read a **confusion matrix**; compute per-class precision/recall. | Read off-diagonal confusions as **real physical continua**, not just bugs. |
| **Save and load** model weights (`state_dict`) and verify the round-trip. | Tie the model's failure modes back to astrophysics. |

---

## Suggested Reading Order

Concept markdowns are numbered in reading order. Part 1 is files 01–04; Part 2 is files 05–08; the project task (09) ties everything together.

### Part 1 — The Training Loop & CNNs

1. [`01-convolutions-and-pooling.md`](01-convolutions-and-pooling.md) — `nn.Conv2d`, `nn.MaxPool2d`, feature maps, and why convolutions beat flattening.
2. [`02-building-a-cnn.md`](02-building-a-cnn.md) — assembling `GalaxyCNN` and tracking shapes to the `Linear` head.
3. [`03-the-training-loop.md`](03-the-training-loop.md) — the five steps, epochs vs batches, and backpropagation.
4. [`04-spiral-structure-and-star-formation.md`](04-spiral-structure-and-star-formation.md) — density waves, dust lanes, and H II regions: what the CNN's filters are really seeing.

### Part 2 — Validation & Interpretation

5. [`05-evaluation-and-overfitting.md`](05-evaluation-and-overfitting.md) — `eval()`, `no_grad()`, train/val curves, and the overfitting signature.
6. [`06-confusion-matrix-and-metrics.md`](06-confusion-matrix-and-metrics.md) — building/reading a confusion matrix and per-class metrics.
7. [`07-saving-and-loading-models.md`](07-saving-and-loading-models.md) — `state_dict`, `torch.save`/`load`, checkpoints, and the reload sanity check.
8. [`08-lenticulars-mergers-and-evolution.md`](08-lenticulars-mergers-and-evolution.md) — S0 galaxies, mergers, and morphology as a time-evolving movie.

### Project

9. [`09-project-task.md`](09-project-task.md) — the two-part deliverable for this week.

For deeper dives, see [`resources.md`](resources.md).

---

## Notebooks

Four notebooks live in [`notebooks/`](notebooks/), in two starter/solution pairs. All open directly in Colab.

| Part | Starter | Solution |
|------|---------|----------|
| 1 — CNN & training | [`week3_cnn_starter.ipynb`](notebooks/week3_cnn_starter.ipynb) | [`week3_cnn_solution.ipynb`](notebooks/week3_cnn_solution.ipynb) |
| 2 — Eval & confusion matrix | [`week3_eval_starter.ipynb`](notebooks/week3_eval_starter.ipynb) | [`week3_eval_solution.ipynb`](notebooks/week3_eval_solution.ipynb) |

Both pairs reuse the Week-1 data pipeline — re-create your `train_loader` / `val_loader` / `test_loader` first (the solutions include the setup cells). **Switch the runtime to GPU** before you start. Always attempt the **starter** before opening the **solution**. To open in Colab: push your fork to GitHub and use **File → Open notebook → GitHub**, or download the `.ipynb` and use **File → Upload notebook**.

---

## Definition of Done

You've completed Week 3 — and the modelling core of the track — when you can, without referring to the solutions:

- [ ] Explain why a `Conv2d` preserves spatial structure (and has far fewer parameters) than the Week-2 `Linear` layer.
- [ ] Define a `GalaxyCNN` and forward-pass a real batch to `(batch_size, num_classes)` logits, deriving the flattened feature size yourself.
- [ ] Write the five-step training loop and run it for several epochs with a **decreasing** training loss.
- [ ] Write an evaluation loop using `model.eval()` and `torch.no_grad()` and compute accuracy.
- [ ] Plot train-vs-validation loss and say whether the model is overfitting.
- [ ] Report a **test** accuracy that beats your Week-2 baseline.
- [ ] Display a labelled confusion matrix and explain at least one confusion in astrophysical terms.
- [ ] Save `galaxy_model.pth`, reload it into a fresh model, and verify the test accuracy matches.

---

## Where to Ask for Help

- Stuck on conv/pool shapes? Re-read [`01-convolutions-and-pooling.md`](01-convolutions-and-pooling.md) and [`02-building-a-cnn.md`](02-building-a-cnn.md), and print `model.features(dummy).shape`.
- Loss won't fall / is `nan`? The pitfalls table in [`03-the-training-loop.md`](03-the-training-loop.md) covers `zero_grad`, device mismatch, LR, and label dtype.
- Accuracy looks wrong or random at eval? See the `eval()` vs `no_grad()` section in [`05-evaluation-and-overfitting.md`](05-evaluation-and-overfitting.md).
- Confusion matrix errors (`cuda` → numpy, unlabelled axes)? See the pitfalls in [`06-confusion-matrix-and-metrics.md`](06-confusion-matrix-and-metrics.md).
- `state_dict` size-mismatch on load? See [`07-saving-and-loading-models.md`](07-saving-and-loading-models.md) — instantiate the same architecture first.
- Still stuck? Drop a message in the CAIC mentor channel — that's exactly what mentors are for.

---

⬅️ Back to [track overview](../README.md) · Previous week: [Week 2](../Week-2/) · Next week: [Week 4](../Week-4/)
