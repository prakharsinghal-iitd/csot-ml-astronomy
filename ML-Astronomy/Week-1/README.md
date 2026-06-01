# Week 1 — Foundations: Tensors, GPUs & the Galaxy Data Pipeline

> The longest, most foundational week of the track. We get comfortable with **Google Colab**, learn PyTorch's core data structure (the **Tensor**), put it on a **GPU**, meet **Edwin Hubble's tuning fork** and the morphology of galaxies, understand **how telescopes turn photons into pixels**, and finally build the **data pipeline** that feeds every model in the rest of the track.

This week is split into **two parts**:

- **Part 1 — Foundations:** Colab + tensors + GPU, plus the astronomy of galaxy morphology.
- **Part 2 — Data pipeline:** how telescopes/CCDs/filters produce our images, then joining flat GZ2 files to CSV labels and loading them with `transforms`, `ImageFolder`, and `DataLoader`.

---

## At a Glance

- **Time estimate:** 7–10 hours total (this is a double-length week — pace it over a week).
- **Prerequisites:** Basic Python (variables, lists, functions). A Google account for Colab. A Kaggle account by Part 2.
- **Deliverable:** Two completed Colab notebooks — `week1_starter.ipynb` (tensors on GPU) and `week1_data_starter.ipynb` (a DataLoader of galaxies, plotted).

---

## Learning Objectives

| ML / Computer Science | Astronomy / Physics |
|-----------------------|---------------------|
| Set up Google Colab and switch to a GPU runtime. | Recount the historical context of Hubble's 1926 classification. |
| Explain what a **tensor** is and how it generalises matrices. | Sketch the **Hubble tuning fork** and name its main branches. |
| Create tensors; inspect `.shape`, `.dtype`, `.device`. | Distinguish **elliptical**, **spiral** (barred/unbarred), **lenticular**, and **irregular** galaxies. |
| Check GPU availability and move tensors with `.to(device)`. | Connect morphology to astrophysics (gas, star formation, dynamics). |
| Benchmark CPU vs. GPU and reason about the speedup. | Explain how a **CCD** turns photons into a 2D array of counts. |
| Build a `transforms` pipeline (resize, to-tensor, normalise). | Explain why telescopes use **monochrome detectors + filters** (`ugriz`). |
| Join flat GZ2 images to CSV labels; load with `ImageFolder`. | Describe how three filter bands become a false-colour RGB image. |
| Batch and shuffle data with a `DataLoader`; plot a batch. | Identify well-known galaxies (M31, M87, NGC 1300) by morphology. |

---

## Suggested Reading Order

Concept markdowns are numbered in reading order. Part 1 is files 01–05; Part 2 is files 06–08; the project task (09) ties everything together.

### Part 1 — Foundations

1. [`01-getting-started-with-colab.md`](01-getting-started-with-colab.md) — set up Colab and switch to GPU runtime.
2. [`02-pytorch-tensors.md`](02-pytorch-tensors.md) — meet the tensor and the operations you'll use all summer.
3. [`03-gpu-acceleration.md`](03-gpu-acceleration.md) — move computation onto the GPU and measure the speedup.
4. [`04-hubble-tuning-fork.md`](04-hubble-tuning-fork.md) — the classification scheme behind every label in our dataset.
5. [`05-galaxy-morphologies.md`](05-galaxy-morphologies.md) — what ellipticals, spirals, and irregulars actually look like (and why).

### Part 2 — Data Pipeline

6. [`06-how-telescopes-see.md`](06-how-telescopes-see.md) — CCDs, calibration, and SDSS: where our pixels come from.
7. [`07-photometry-and-filters.md`](07-photometry-and-filters.md) — monochrome detectors, the `ugriz` filters, and false-colour images.
8. [`08-data-pipelines.md`](08-data-pipelines.md) — `transforms`, `ImageFolder`, and `DataLoader` in PyTorch.

### Project

9. [`09-project-task.md`](09-project-task.md) — the two-part deliverable for this week.

For deeper dives, see [`resources.md`](resources.md).

---

## Notebooks

Four notebooks live in [`notebooks/`](notebooks/), in two starter/solution pairs. All open directly in Colab.

| Part | Starter | Solution |
|------|---------|----------|
| 1 — Foundations | [`week1_starter.ipynb`](notebooks/week1_starter.ipynb) | [`week1_solution.ipynb`](notebooks/week1_solution.ipynb) |
| 2 — Data pipeline | [`week1_data_starter.ipynb`](notebooks/week1_data_starter.ipynb) | [`week1_data_solution.ipynb`](notebooks/week1_data_solution.ipynb) |

Always attempt the **starter** before opening the **solution**. To open in Colab: push your fork to GitHub and use **File → Open notebook → GitHub**, or download the `.ipynb` and use **File → Upload notebook**.

---

## Definition of Done

You're ready for Week 2 when you can, without referring to the solutions:

- [ ] Open a Colab notebook with a **T4 GPU runtime** active and confirm it via `!nvidia-smi`.
- [ ] Print `torch.cuda.is_available()` → `True` and move a `(3, 64, 64)` tensor to the GPU.
- [ ] Explain in your own words what `.shape` and `.device` mean.
- [ ] Draw or describe the Hubble tuning fork and place an example galaxy on each prong.
- [ ] Explain why a CCD image is monochrome and how three filters make a colour image.
- [ ] Join the GZ2 mapping + label CSVs, build an `ImageFolder` layout, create a `DataLoader`, and plot a labelled batch.

---

## Where to Ask for Help

- Stuck on Colab? Re-read [`01-getting-started-with-colab.md`](01-getting-started-with-colab.md) and its "Common Pitfalls" section.
- Stuck on tensors? Read the PyTorch ["Tensors" tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) (linked in [`resources.md`](resources.md)).
- Stuck on the data pipeline? Re-read [`08-data-pipelines.md`](08-data-pipelines.md) — the pitfalls table covers the usual suspects.
- Stuck on morphology? Try the [Galaxy Zoo classification tutorial](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/).
- Still stuck? Drop a message in the CAIC mentor channel — that's exactly what mentors are for.
