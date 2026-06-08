# 07 — Project Task: A Baseline, Then Your First Neural Network

> The Week 2 deliverable, in **two parts**. Part 1 establishes the **scikit-learn baseline** — the accuracy number every later model must beat. Part 2 builds your **first PyTorch neural network** (a 2-layer MLP) and proves the plumbing works by forward-passing a batch and printing the architecture. Together they are the bridge from "data is loaded" (Week 1) to "a model learns" (Week 3).

---

## The Task

> **Part 1 — Baseline with scikit-learn.** Take your Week-1 `DataLoader`s, flatten the galaxy images into NumPy feature matrices, train a simple classifier (KNN and/or Logistic Regression), and record the **test accuracy**. Compare it to the random-guess and majority-class floors, and write down why a flatten-and-classify model struggles with morphology.
>
> **Part 2 — Fully-connected network.** Define a 2-layer MLP as an `nn.Module`, move it to the GPU, forward-pass one real batch to confirm it outputs `(B, num_classes)` logits, print the architecture and parameter count, set up `CrossEntropyLoss` + `Adam`, and run a single optimisation step to watch the loss drop.

Part 1 gives you a yardstick. Part 2 gives you the model that Week 3 will actually train.

---

## Starter and Solution Notebooks

There are **two notebook pairs** this week — one per part. Attempt the starter before opening the solution.

| Part | Starter | Solution |
|------|---------|----------|
| 1 — Baseline | 🟢 [`notebooks/week2_baseline_starter.ipynb`](notebooks/week2_baseline_starter.ipynb) | 🟡 [`notebooks/week2_baseline_solution.ipynb`](notebooks/week2_baseline_solution.ipynb) |
| 2 — MLP | 🟢 [`notebooks/week2_mlp_starter.ipynb`](notebooks/week2_mlp_starter.ipynb) | 🟡 [`notebooks/week2_mlp_solution.ipynb`](notebooks/week2_mlp_solution.ipynb) |

Both are designed to open in Colab directly. They reuse the Week-1 data pipeline — see "Reusing your Week-1 data" below.

---

## Part 1 — Step-by-Step Checklist

Each step maps to a cell (or small group) in `week2_baseline_starter.ipynb`.

- [ ] **Step 0.** Open the starter in Colab with a **GPU runtime**, and re-create your Week-1 `DataLoader`s (train/val/test) — copy the data-prep cells from `week1_data_solution.ipynb`.
- [ ] **Step 1.** Write a helper that turns a `DataLoader` into NumPy arrays: for each batch, `images.flatten(start_dim=1)` then `.numpy()`; concatenate into `X` `(N, 12288)` and `y` `(N,)`. (See [`01-tensors-to-numpy-and-flattening.md`](01-tensors-to-numpy-and-flattening.md).)
- [ ] **Step 2.** Build `X_train, y_train` and `X_test, y_test`. Print their shapes and confirm `X_train.shape[1] == 3*64*64`.
- [ ] **Step 3.** Fit a `DummyClassifier(strategy="most_frequent")`; print the **majority-class** accuracy (your lazy floor).
- [ ] **Step 4.** Fit a `KNeighborsClassifier(n_neighbors=5)` on the train set; print its **test** accuracy.
- [ ] **Step 5.** Fit a `LogisticRegression(max_iter=1000)`; print its **test** accuracy.
- [ ] **Step 6.** Print a small comparison table: random `1/num_classes`, majority, KNN, LogReg. Identify the number to beat in Week 3.
- [ ] **Step 7.** In a Markdown cell, write 2–3 sentences on *why* these models struggle with galaxy shape (hint: flattening; [`03-surface-brightness-and-isophotes.md`](03-surface-brightness-and-isophotes.md)).

---

## Part 2 — Step-by-Step Checklist

Each step maps to a cell in `week2_mlp_starter.ipynb`.

- [ ] **Step 0.** Re-create the Week-1 `DataLoader`s and define `device = "cuda" if torch.cuda.is_available() else "cpu"`.
- [ ] **Step 1.** Define a `GalaxyMLP(nn.Module)` with `nn.Flatten`, `nn.Linear(3*64*64, 128)`, `nn.ReLU`, `nn.Linear(128, num_classes)`. (See [`04-building-models-with-nn-module.md`](04-building-models-with-nn-module.md).) Remember `super().__init__()`.
- [ ] **Step 2.** Instantiate the model with the correct `num_classes` (`len(train_ds.classes)`) and `.to(device)` it.
- [ ] **Step 3.** `print(model)` and print the total/trainable parameter count.
- [ ] **Step 4.** Grab one batch from `train_loader`, move it to `device`, forward-pass it, and confirm `logits.shape == (batch_size, num_classes)`.
- [ ] **Step 5.** Define `criterion = nn.CrossEntropyLoss()` and `optimizer = optim.Adam(model.parameters(), lr=1e-3)`.
- [ ] **Step 6.** Compute the loss on that batch and confirm it's near `ln(num_classes)` for the untrained model. (See [`05-loss-functions-and-optimisers.md`](05-loss-functions-and-optimisers.md).)
- [ ] **Step 7.** Run **one** optimisation step (`zero_grad` → `backward` → `step`), re-compute the loss on the same batch, and confirm it **decreased**.

---

## Acceptance Criteria

Your submission counts as complete when, on a fresh Colab session:

1. **Both** notebooks run top-to-bottom (`Runtime → Restart and run all`) **without errors**.
2. **Part 1:** `X_train` has shape `(N, 12288)`; you print majority-class, KNN, and LogReg **test** accuracies; and you state the single baseline number Week 3 must beat.
3. **Part 2:** `print(model)` shows the 2-layer MLP; a forward pass on a real batch returns logits of shape `(batch_size, num_classes)`; the untrained loss is sensible (≈ `ln(num_classes)`); and after one optimisation step the loss on the same batch **goes down**.
4. Both notebooks are saved to your Drive *or* pushed to your fork so a mentor can review them.

---

## Stretch Goals (Optional but Recommended)

### 1. A confusion matrix for the baseline

Use `sklearn.metrics.confusion_matrix` and `ConfusionMatrixDisplay` on your KNN/LogReg predictions. Which two classes get confused most? Write down your guess — we'll compare it to the CNN's confusion matrix in Week 3.

### 2. Feature engineering beats raw pixels

Instead of 12 288 raw pixels, compute a handful of *physically-motivated* features per image — mean colour per channel, overall brightness, a crude central-concentration ratio (central-region brightness ÷ total). Train a classifier on just these few features. Does a *smaller, smarter* feature set rival raw pixels? (This is the spirit of the Sérsic index from [`06-stellar-demographics-and-sersic.md`](06-stellar-demographics-and-sersic.md).)

### 3. Vary the MLP

Add a second hidden layer, or change the hidden width (e.g. 64, 256, 512). Re-print the parameter count each time. How fast does the count grow? Which layer dominates, and why? (Recall: the first `Linear` carries `12288 × hidden` weights.)

### 4. The pixel-shuffle demonstration

Apply one fixed random permutation to the pixel order of *every* image, rebuild `X_train`/`X_test`, and re-train the baseline. Confirm the accuracy is essentially **unchanged**. Explain in 2 sentences why this proves the baseline ignores spatial structure (see [`01-tensors-to-numpy-and-flattening.md`](01-tensors-to-numpy-and-flattening.md)).

### 5. Learning-rate sensitivity

In Part 2, run the single optimisation step with `lr=1e-1`, `1e-3`, and `1e-6`. Which lowers the loss most on one step? Which diverges? This previews why the learning rate matters so much in Week 3.

---

## Reusing Your Week-1 Data

Both notebooks assume you can produce the Week-1 `train_loader` / `val_loader` / `test_loader`. The simplest path:

1. Copy the dataset-download and `galaxy_data/{train,val,test}/<class>/` build cells from [`week1_data_solution.ipynb`](../Week-1/notebooks/week1_data_solution.ipynb).
2. Re-create the `transforms.Compose([...])`, the three `ImageFolder`s, and the three `DataLoader`s exactly as in Week 1.
3. Keep `class_to_idx` handy — you'll need `num_classes = len(train_ds.classes)` in Part 2.

> If you mounted Google Drive and saved `galaxy_data/` there in Week 1, you can skip re-downloading — just re-mount Drive and point `ImageFolder` at the saved folders.

---

## Opening the Notebooks in Colab

### Option A — From a GitHub fork (recommended)

1. Fork the [CSoT repo](../../../README.md) on GitHub.
2. In Colab: **File → Open notebook → GitHub** tab.
3. Open the relevant notebook under `ML-Astronomy/Week-2/notebooks/`.
4. Edits autosave to Drive; push back with **File → Save a copy in GitHub**.

### Option B — Direct GitHub link

Paste a URL like this into Colab's "GitHub" tab (swap in your username):

```
https://github.com/YOUR_USER/csot-ml-astronomy/blob/main/ML-Astronomy/Week-2/notebooks/week2_baseline_starter.ipynb
```

### Option C — Upload

Download the `.ipynb` from this folder, then in Colab use **File → Upload notebook**.

---

## Submission

Submission logistics are confirmed via the official CAIC channels. Default expectation:

- Push both completed notebooks into a `submissions/<your-name>/week2/` folder in your fork, **or**
- Share Colab "share links" (with comment access) with your mentor.

Ask in the mentor channel **before** the week ends if anything is unclear.

---

## Reflection Questions

Write 2–3 sentences each in a Markdown cell at the bottom of the relevant notebook:

1. What baseline accuracy did you get, and how far above the majority-class floor is it? What does that tell you about how hard the problem is?
2. Why does flattening hurt a galaxy classifier specifically (as opposed to, say, classifying simple coloured shapes)?
3. In Part 2, your untrained loss should have been near `ln(num_classes)`. Why is that the expected starting value?
4. After one optimisation step the loss dropped on that batch. Why is that *not yet* the same as "the model is trained"?

These won't be graded — they exist so you can look back at the end of the track and see how far you've come.

---

⬅️ Previous: [`06-stellar-demographics-and-sersic.md`](06-stellar-demographics-and-sersic.md) | 📚 See also: [`resources.md`](resources.md)
