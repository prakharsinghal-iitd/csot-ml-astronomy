# Week 2 — Baselines & Fully-Connected Networks

> This week we go from "data is loaded" to "a model makes predictions". First we set a **traditional machine-learning baseline** with scikit-learn — a number every fancier model must beat — then we write our **first neural network** in PyTorch: a fully-connected (dense) network built from `nn.Module`, complete with a loss function and an optimiser.

This week is split into **two parts**:

- **Part 1 — A baseline with scikit-learn:** flatten galaxy images into NumPy rows and classify them with KNN / Logistic Regression, plus the astronomy of surface brightness and isophotes.
- **Part 2 — Fully-connected networks:** build an MLP with `nn.Module`, wire up `CrossEntropyLoss` + `Adam`, plus the astronomy of stellar populations and the Sérsic profile.

---

## At a Glance

- **Time estimate:** 4–6 hours.
- **Prerequisites:** Finish [Week 1](../Week-1/) first — you need a working `DataLoader` of galaxy images and comfort with tensors, shapes, and devices.
- **Deliverable:** Two completed Colab notebooks — `week2_baseline_starter.ipynb` (a scikit-learn baseline accuracy) and `week2_mlp_starter.ipynb` (a 2-layer MLP that forward-passes a batch, prints its architecture, and drops its loss on one optimisation step).

---

## Learning Objectives

| ML / Computer Science | Astronomy / Physics |
|-----------------------|---------------------|
| Convert PyTorch tensors to NumPy arrays (`.detach().cpu().numpy()`). | Define **surface brightness** and explain why it's distance-independent. |
| **Flatten** a `(3, 64, 64)` image into a 1D vector and explain what spatial info is lost. | Read a galaxy's **surface brightness profile** `I(r)`. |
| Train and evaluate **KNN** and **Logistic Regression** baselines with scikit-learn. | Recognise **isophotes** and how their shape differs across morphologies. |
| Reason about random-guess vs **majority-class** floors and the "bar to beat". | Connect **stellar demographics** (old red vs young blue stars) to galaxy colour. |
| Define a model by subclassing **`nn.Module`** (`__init__` + `forward`). | Explain why ellipticals are "red and dead" and spirals are blue. |
| Build an **MLP** with `nn.Linear` + `nn.ReLU`; inspect parameters. | Interpret the **Sérsic profile** and what its index `n` says about a galaxy. |
| Set up **`CrossEntropyLoss`** (logits!) and the **`Adam`** optimiser. | Distinguish disk-dominated (`n ≈ 1`) from bulge-dominated (`n ≈ 4`) systems. |
| Run the `zero_grad → backward → step` update and watch a loss decrease. | Explain why colour helps a pixel baseline but structure needs a CNN. |

---

## Suggested Reading Order

Concept markdowns are numbered in reading order. Part 1 is files 01–03; Part 2 is files 04–06; the project task (07) ties everything together.

### Part 1 — A Baseline with Scikit-Learn

1. [`01-tensors-to-numpy-and-flattening.md`](01-tensors-to-numpy-and-flattening.md) — tensor → NumPy, and flattening images (and what it destroys).
2. [`02-baseline-with-scikit-learn.md`](02-baseline-with-scikit-learn.md) — KNN, Logistic Regression, and what "baseline" means.
3. [`03-surface-brightness-and-isophotes.md`](03-surface-brightness-and-isophotes.md) — how light is distributed across a galaxy, and the shapes its contours make.

### Part 2 — Fully-Connected Networks

4. [`04-building-models-with-nn-module.md`](04-building-models-with-nn-module.md) — the `nn.Module` pattern, `nn.Linear`, and ReLU.
5. [`05-loss-functions-and-optimisers.md`](05-loss-functions-and-optimisers.md) — `CrossEntropyLoss`, `Adam`, and the three lines that make learning happen.
6. [`06-stellar-demographics-and-sersic.md`](06-stellar-demographics-and-sersic.md) — why galaxy colours and concentrations differ, and the Sérsic index.

### Project

7. [`07-project-task.md`](07-project-task.md) — the two-part deliverable for this week.

For deeper dives, see [`resources.md`](resources.md).

---

## Notebooks

Four notebooks live in [`notebooks/`](notebooks/), in two starter/solution pairs. All open directly in Colab.

| Part | Starter | Solution |
|------|---------|----------|
| 1 — Baseline | [`week2_baseline_starter.ipynb`](notebooks/week2_baseline_starter.ipynb) | [`week2_baseline_solution.ipynb`](notebooks/week2_baseline_solution.ipynb) |
| 2 — MLP | [`week2_mlp_starter.ipynb`](notebooks/week2_mlp_starter.ipynb) | [`week2_mlp_solution.ipynb`](notebooks/week2_mlp_solution.ipynb) |

Both notebooks reuse the Week-1 data pipeline — re-create your `train_loader` / `val_loader` / `test_loader` first (the solutions include the setup cells). Always attempt the **starter** before opening the **solution**. To open in Colab: push your fork to GitHub and use **File → Open notebook → GitHub**, or download the `.ipynb` and use **File → Upload notebook**.

---

## Definition of Done

You're ready for Week 3 when you can, without referring to the solutions:

- [ ] Flatten a `DataLoader` of galaxies into a NumPy `X` of shape `(N, 12288)` and a matching `y`.
- [ ] Train KNN and Logistic Regression baselines and report **test** accuracy against the majority-class floor.
- [ ] State, in one sentence, why flattening blinds these models to galaxy morphology.
- [ ] Define a 2-layer MLP as an `nn.Module` and forward-pass a real batch to get `(B, num_classes)` logits.
- [ ] Explain why the output layer returns raw logits and why the untrained loss is ≈ `ln(num_classes)`.
- [ ] Run one `zero_grad → backward → step` and confirm the loss on that batch decreases.
- [ ] Describe a galaxy's surface brightness profile, what an isophote is, and what the Sérsic index `n` measures.

---

## Where to Ask for Help

- Stuck on tensor → NumPy or flattening? Re-read [`01-tensors-to-numpy-and-flattening.md`](01-tensors-to-numpy-and-flattening.md) and its self-check.
- Stuck on scikit-learn? See [`02-baseline-with-scikit-learn.md`](02-baseline-with-scikit-learn.md) and the [scikit-learn Getting Started guide](https://scikit-learn.org/stable/getting_started.html).
- Stuck on `nn.Module`? Re-read [`04-building-models-with-nn-module.md`](04-building-models-with-nn-module.md) and the PyTorch ["Build the Neural Network" tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).
- Loss is `nan` or won't drop? The pitfalls table in [`05-loss-functions-and-optimisers.md`](05-loss-functions-and-optimisers.md) covers the usual suspects (label dtype, double-softmax, `zero_grad`).
- Still stuck? Drop a message in the CAIC mentor channel — that's exactly what mentors are for.

---

⬅️ Back to [track overview](../README.md) · Previous week: [Week 1](../Week-1/) · Next week: [Week 3](../Week-3/)
