# 09 — Project Task: Setup, Tensors & Your First Data Pipeline

> The Week 1 deliverable, in **two parts**. Part 1 is the bare-minimum smoke test (Colab + GPU + a tensor). Part 2 is the first piece of the real project (loading galaxy images through a `DataLoader` and looking at them). Everything we build in later weeks assumes both work.

---

## The Task

> **Part 1 — Foundations.** Set up a Google Colab notebook with the GPU runtime enabled, verify the GPU is active in PyTorch, create a random tensor shaped like a galaxy image, and move it to the GPU.
>
> **Part 2 — Data pipeline.** Download the Galaxy Zoo 2 dataset, join the official morphology labels to the flat JPG files, build an `ImageFolder`-ready folder layout, wrap it in `ImageFolder` with a `transforms` pipeline, build a `DataLoader`, and plot a batch of real galaxies with `matplotlib`.

Part 1 confirms your environment works. Part 2 produces the data pipeline that feeds every model in the rest of the track.

---

## Starter and Solution Notebooks

There are **two notebook pairs** this week — one per part. Attempt the starter before opening the solution.

| Part | Starter | Solution |
|------|---------|----------|
| 1 — Foundations | 🟢 [`notebooks/week1_starter.ipynb`](notebooks/week1_starter.ipynb) | 🟡 [`notebooks/week1_solution.ipynb`](notebooks/week1_solution.ipynb) |
| 2 — Data pipeline | 🟢 [`notebooks/week1_data_starter.ipynb`](notebooks/week1_data_starter.ipynb) | 🟡 [`notebooks/week1_data_solution.ipynb`](notebooks/week1_data_solution.ipynb) |

All four are designed to open in Colab directly. See "Opening in Colab" below.

---

## Part 1 — Step-by-Step Checklist

Each step maps to a cell (or small group of cells) in `week1_starter.ipynb`.

- [ ] **Step 0.** Open the starter notebook in Colab and switch to a **GPU runtime** (`Runtime → Change runtime type → GPU`).
- [ ] **Step 1.** Run `!nvidia-smi` and confirm an NVIDIA GPU appears in the output.
- [ ] **Step 2.** Import `torch`, `torchvision`, and `matplotlib.pyplot` and print their versions.
- [ ] **Step 3.** Print `torch.cuda.is_available()`. Expect `True`. Also print `torch.cuda.get_device_name(0)`.
- [ ] **Step 4.** Define `device = "cuda" if torch.cuda.is_available() else "cpu"` and print it.
- [ ] **Step 5.** Create a random tensor of shape `(3, 64, 64)` (one toy galaxy image). Print its `.shape`, `.dtype`, and `.device`.
- [ ] **Step 6.** Move the tensor to the GPU with `.to(device)`. Print its `.device` again and confirm it now reads `cuda:0`.
- [ ] **Step 7.** Perform a sanity matmul: take a `(64, 64)` slice of one channel and matrix-multiply it with itself. Print the result's shape.

---

## Part 2 — Step-by-Step Checklist

Each step maps to a cell in `week1_data_starter.ipynb`.

- [ ] **Step 1.** Get the dataset into Colab via the Kaggle API (`kaggle datasets download -d jaimetrickz/galaxy-zoo-2-images`) with your `kaggle.json` token. Unzip `images_gz2.zip` and download the Hart et al. (2016) label CSV from [data.galaxyzoo.org](https://data.galaxyzoo.org/) (`gz2_hart16.csv.gz`).
- [ ] **Step 2.** Inspect the **raw** layout: flat `{asset_id}.jpg` files plus `gz2_filename_mapping.csv` — *not* class subfolders.
- [ ] **Step 3.** Merge `gz2_filename_mapping.csv` with `gz2_hart16.csv` on `objid` (rename `dr7objid` first). Collapse `gz2_class` to high-level labels. Print label counts.
- [ ] **Step 4.** Symlink (or copy) images into `galaxy_data/{train,val,test}/<class_name>/`. Split each class ~70% / 15% / 15%.
- [ ] **Step 5.** Build a `transforms.Compose([...])` pipeline: `Resize((64, 64))`, `ToTensor()`, `Normalize([0.5]*3, [0.5]*3)`.
- [ ] **Step 6.** Create three `ImageFolder`s for `train`, `val`, and `test`. Print `len`, `classes`, and `class_to_idx`.
- [ ] **Step 7.** Fetch one training sample; print `image.shape` and `label`.
- [ ] **Step 8.** Create `DataLoader`s for each split (`shuffle=True` for train only). Print one batch shape.
- [ ] **Step 9.** Plot a grid of ~16 training images. Undo normalisation and `.permute(1, 2, 0)` before `imshow`. Print class names.

---

## Acceptance Criteria

Your submission counts as complete when, on a fresh Colab session:

1. **Both** notebooks run top-to-bottom (`Runtime → Restart and run all`) **without errors**.
2. **Part 1:** `torch.cuda.is_available()` prints `True`, and the tensor's `.device` prints `cuda:0` after the move.
3. **Part 2:** Three `ImageFolder` splits (`train`/`val`/`test`) each report a non-zero image count and matching `class_to_idx`; a training `DataLoader` batch has shape `(32, 3, 64, 64)`; the plotted grid shows recognisable galaxies with correct labels.
4. Both notebooks are saved to your Drive *or* pushed to your fork so a mentor can review them.

---

## Stretch Goals (Optional but Recommended)

### 1. Benchmark CPU vs GPU

Adapt the benchmark snippet from [`03-gpu-acceleration.md`](03-gpu-acceleration.md) and report the speedup for matmul of `(4096, 4096)` matrices.

> Bonus: plot CPU time vs GPU time for sizes `[256, 512, 1024, 2048, 4096]`. Does the speedup grow with size? Why?

### 2. Play with `dtype`

Create the same `(3, 64, 64)` tensor with `dtype=torch.float16` and `dtype=torch.float64`. Compare `.element_size()` and total memory. When would you use each?

- `float16` (half) — fast on modern GPUs, less memory, less range. Used in mixed-precision training.
- `float64` (double) — overkill for ML, standard in scientific computing.
- `float32` (single) — the ML default, the right choice almost always.

### 3. Compute the *real* normalisation statistics

Instead of assuming `mean = std = 0.5`, iterate over the (un-normalised) training set and compute the **actual per-channel mean and standard deviation**. Plug those into `Normalize`. How different are they from `0.5`? (This is what a careful pipeline does.)

### 4. Visualise per-class examples

Plot a 3×N grid showing several examples of each galaxy class side by side. Eyeball the differences you read about in [`05-galaxy-morphologies.md`](05-galaxy-morphologies.md). Which two classes look hardest to tell apart? Write down your guess — we'll check it against the model's confusion matrix later.

### 5. Try data augmentation

Build a second `train_transform` that adds `RandomHorizontalFlip()` and `RandomRotation(180)`. Plot the same image a few times through it and watch it flip/rotate. Why is rotation a *physically reasonable* augmentation for galaxies but not for, say, photos of street signs?

---

## Opening the Notebooks in Colab

### Option A — From a GitHub fork (recommended)

1. Fork the [CSoT repo](../../../README.md) on GitHub.
2. In Colab: **File → Open notebook → GitHub** tab.
3. Search your fork and open the relevant notebook under `ML-Astronomy/Week-1/notebooks/`.
4. Edits autosave to Drive; push back with **File → Save a copy in GitHub**.

### Option B — Direct GitHub link

Paste a URL like this into Colab's "GitHub" tab (swap in your username):

```
https://github.com/YOUR_USER/csot-ml-astronomy/blob/main/ML-Astronomy/Week-1/notebooks/week1_data_starter.ipynb
```

### Option C — Upload

Download the `.ipynb` from this folder, then in Colab use **File → Upload notebook**.

---

## Getting the Dataset (Part 2)

The full dataset lives on Kaggle: [Galaxy Zoo 2 Images](https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images). Important: the download is **not** pre-sorted by morphology class. You get:

- `images_gz2.zip` — flat JPGs named `{asset_id}.jpg`
- `gz2_filename_mapping.csv` — links each JPG to an SDSS `objid`

Morphology labels come from the official GZ2 catalogue ([Hart et al. 2016 Table 1](https://data.galaxyzoo.org/), `gz2_hart16.csv.gz`). The data notebook merges these files and builds an `ImageFolder`-ready layout.

- **Kaggle API (recommended).** Create a Kaggle account → Account → "Create New API Token" → download `kaggle.json`. Upload it to Colab, then follow the download cells in the data notebook.
- **Balanced subset.** For a fast first run you don't need all ~243k images. Symlink ~200 galaxies per class (as the solution notebook does) so loading and plotting stay quick on Colab.

> Reminder from [`07-photometry-and-filters.md`](07-photometry-and-filters.md): each image is a false-colour `(3, H, W)` composite of the SDSS `i/r/g` bands. The "colours" you plot are real measurements in different wavelength bands.

---

## Submission

Submission logistics are confirmed via the official CAIC channels. Default expectation:

- Push both completed notebooks into a `submissions/<your-name>/week1/` folder in your fork, **or**
- Share Colab "share links" (with comment access) with your mentor.

Ask in the mentor channel **before** the week ends if anything is unclear.

---

## Reflection Questions

Write 2–3 sentences each in a Markdown cell at the bottom of your data notebook:

1. What was the most confusing part of setup or the data pipeline? How did you resolve it?
2. Pick one galaxy class (E / S / S0 / Irr) and describe what features a CNN might need to detect to recognise it.
3. After plotting real batches: which two classes do you think will be hardest for a model to tell apart, and why?
4. Why do we spend so much effort on the data pipeline *before* building any model?

These won't be graded — they exist so you can look back at the end of the track and see how far you've come.

---

⬅️ Previous: [`08-data-pipelines.md`](08-data-pipelines.md) | 📚 See also: [`resources.md`](resources.md)
