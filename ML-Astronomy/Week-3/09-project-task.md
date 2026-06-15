# 09 — Project Task: Train a CNN, Then Evaluate It

> The headline deliverable of the whole track, in **two parts**. Part 1 replaces Week 2's fully-connected model with a **CNN** and writes the **training loop** that teaches it — you watch the loss fall over several epochs. Part 2 writes the **evaluation loop**, plots **train-vs-validation** curves, builds a **confusion matrix** you can interpret astrophysically, and saves the trained weights to `galaxy_model.pth`. By the end you have a real, reusable galaxy classifier.

---

## The Task

> **Part 1 — CNN + training loop.** Rebuild your Week-1 `DataLoader`s, define a `GalaxyCNN` (`Conv → ReLU → Pool` blocks + a `Linear` head), forward-pass a batch to confirm shapes, then write the five-step training loop and run it for 5–10 epochs. Record the per-epoch training loss and plot a clearly **decreasing** loss curve.
>
> **Part 2 — Evaluation + interpretation.** Write an `evaluate` function using `model.eval()` and `torch.no_grad()`. Track **training and validation** loss together across epochs and plot both. Report **test** accuracy and confirm it **beats your Week-2 baseline**. Build and label a **confusion matrix**, read off the per-class confusions, and `torch.save(model.state_dict(), 'galaxy_model.pth')` — then reload it and verify the round-trip.

Part 1 produces a model that *learns*. Part 2 proves it learned something *real*, tells you *where* it fails, and ships the weights.

---

## Starter and Solution Notebooks

There are **two notebook pairs** this week — one per part. Attempt the starter before opening the solution.

| Part | Starter | Solution |
|------|---------|----------|
| 1 — CNN & training | 🟢 [`notebooks/week3_cnn_starter.ipynb`](notebooks/week3_cnn_starter.ipynb) | 🟡 [`notebooks/week3_cnn_solution.ipynb`](notebooks/week3_cnn_solution.ipynb) |
| 2 — Eval & confusion matrix | 🟢 [`notebooks/week3_eval_starter.ipynb`](notebooks/week3_eval_starter.ipynb) | 🟡 [`notebooks/week3_eval_solution.ipynb`](notebooks/week3_eval_solution.ipynb) |

Both are designed to open in Colab directly. They reuse the Week-1 data pipeline — see "Reusing your Week-1 data" below. **Switch the runtime to GPU** (`Runtime → Change runtime type → GPU`) before you start; this is the week where it actually matters for speed.

---

## Part 1 — Step-by-Step Checklist

Each step maps to a cell (or small group) in `week3_cnn_starter.ipynb`.

- [ ] **Step 0.** Open the starter in Colab with a **GPU runtime**, and re-create your Week-1 `train_loader` / `val_loader` / `test_loader` and `num_classes` (copy the data-prep cells from `week1_data_solution.ipynb`).
- [ ] **Step 1.** Define `GalaxyCNN(nn.Module)`: two `Conv2d(…, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)` blocks (channels `3→16→32`), then `Flatten → Linear(32*16*16, 128) → ReLU → Linear(128, num_classes)`. (See [`02-building-a-cnn.md`](02-building-a-cnn.md).) Remember `super().__init__()`.
- [ ] **Step 2.** Instantiate the model with the correct `num_classes` and `.to(device)` it.
- [ ] **Step 3.** Forward-pass **one real batch** and confirm the output is `(batch_size, num_classes)`. Print the parameter count and note how the convolutions are cheap vs the `Linear` head (contrast with the Week-2 MLP).
- [ ] **Step 4.** Define `criterion = nn.CrossEntropyLoss()` and `optimizer = optim.Adam(model.parameters(), lr=1e-3)`. Sanity-check the untrained loss is near `ln(num_classes)`.
- [ ] **Step 5.** Write the training loop: for each epoch, loop over `train_loader` running the five steps (`zero_grad → forward → loss → backward → step`), accumulate a running loss, and append the epoch average to a `train_losses` list. (See [`03-the-training-loop.md`](03-the-training-loop.md).)
- [ ] **Step 6.** Run for **5–10 epochs**. Print the loss each epoch and confirm the trend is **down**.
- [ ] **Step 7.** Plot `train_losses` vs epoch with matplotlib. Label the axes.

---

## Part 2 — Step-by-Step Checklist

Each step maps to a cell in `week3_eval_starter.ipynb`.

- [ ] **Step 0.** Re-create the Week-1 `DataLoader`s and your `GalaxyCNN` (re-train it here, or load Part 1's `galaxy_model.pth` if you saved it).
- [ ] **Step 1.** Write `evaluate(model, loader, criterion, device)` that uses `model.eval()` + `torch.no_grad()` and returns `(avg_loss, accuracy)`. (See [`05-evaluation-and-overfitting.md`](05-evaluation-and-overfitting.md).)
- [ ] **Step 2.** Re-run training while calling `evaluate(model, val_loader, …)` each epoch; record `train_losses` and `val_losses`.
- [ ] **Step 3.** Plot training and validation loss on the same axes. Comment (Markdown) on whether you see overfitting.
- [ ] **Step 4.** Compute **test** accuracy with `evaluate(model, test_loader, …)`. Compare it to your **Week-2 baseline** number and state the improvement.
- [ ] **Step 5.** Collect predictions over the test set and build a **confusion matrix** with `ConfusionMatrixDisplay`, using `display_labels=test_ds.classes`. (See [`06-confusion-matrix-and-metrics.md`](06-confusion-matrix-and-metrics.md).)
- [ ] **Step 6.** Print `classification_report` (per-class precision/recall/F1). In a Markdown cell, name the most-confused pair and link it to the astronomy ([`04`](04-spiral-structure-and-star-formation.md), [`08`](08-lenticulars-mergers-and-evolution.md)).
- [ ] **Step 7.** `torch.save(model.state_dict(), 'galaxy_model.pth')`, reload into a fresh `GalaxyCNN`, and **assert** the reloaded test accuracy matches. (See [`07-saving-and-loading-models.md`](07-saving-and-loading-models.md).)

---

## Acceptance Criteria

Your submission counts as complete when, on a fresh Colab session:

1. **Both** notebooks run top-to-bottom (`Runtime → Restart and run all`) **without errors**.
2. **Part 1:** `GalaxyCNN` forward-passes a real batch to `(batch_size, num_classes)` logits; the training loop runs for ≥5 epochs; and you show a **decreasing** training-loss curve.
3. **Part 2:** you plot **train vs validation** loss; you report a **test accuracy that beats the Week-2 baseline**; you display a **labelled confusion matrix** and name at least one specific confusion; and `galaxy_model.pth` is saved and verified by a reload round-trip.
4. Both notebooks are saved to your Drive *or* pushed to your fork so a mentor can review them. (Don't commit the `.pth` or the dataset to the repo — see Submission.)

---

## Stretch Goals (Optional but Recommended)

### 1. Data augmentation (the natural galaxy win)

Add `transforms.RandomHorizontalFlip()` and `transforms.RandomRotation(180)` to the **training** transform only (keep val/test clean). Galaxies have no preferred orientation, so this is almost-free extra signal. Re-train and compare the train/val gap — augmentation usually narrows it. (See [`05-evaluation-and-overfitting.md`](05-evaluation-and-overfitting.md).)

### 2. Go deeper / wider

Add a third `Conv → ReLU → Pool` block (spatial `64→32→16→8`) or bump channels to `32→64`. Recompute the flattened size and the parameter count. Does accuracy improve, or just training time?

### 3. Save the *best* model (early stopping)

Track the lowest validation loss and overwrite `galaxy_model_best.pth` only when validation improves. Compare the best-epoch model's test accuracy to the last-epoch model's. (See [`07-saving-and-loading-models.md`](07-saving-and-loading-models.md).)

### 4. Add dropout, and prove `eval()` matters

Put `nn.Dropout(0.3)` in the classifier head. Evaluate **with** and **without** `model.eval()` and watch the accuracy wobble when you forget it — concrete proof of why eval mode exists.

### 5. Split the spiral class (the granularity experiment)

The Week-1 pipeline already separates `spiral` and `spiral_barred`. Train once with them merged into a single `spiral`, then again split, and compare confusion matrices. Can the CNN recover the bar feature on its own? (This is the experiment promised in the [track overview](../README.md).)

### 6. Look at what the model got wrong

Plot a grid of the test galaxies the model **misclassified**, titled with "true → predicted". Do *you* find them ambiguous too? This is the most fun cell in the whole track.

---

## Reusing Your Week-1 Data

Both notebooks assume you can produce the Week-1 `train_loader` / `val_loader` / `test_loader`. The simplest path:

1. Copy the dataset-download and `galaxy_data/{train,val,test}/<class>/` build cells from [`week1_data_solution.ipynb`](../Week-1/notebooks/week1_data_solution.ipynb).
2. Re-create the `transforms.Compose([...])`, the three `ImageFolder`s, and the three `DataLoader`s exactly as in Week 1.
3. Keep `num_classes = len(train_ds.classes)` and `test_ds.classes` handy — Part 1 needs the count, Part 2 needs the names for the confusion matrix.

> If you mounted Google Drive and saved `galaxy_data/` there in Week 1, skip re-downloading — re-mount Drive and point `ImageFolder` at the saved folders. Bump `PER_CLASS` (e.g. 200 → 1000) for a stronger model once the pipeline runs.

---

## Opening the Notebooks in Colab

### Option A — From a GitHub fork (recommended)

1. Fork the [CSoT repo](../../../README.md) on GitHub.
2. In Colab: **File → Open notebook → GitHub** tab.
3. Open the relevant notebook under `ML-Astronomy/Week-3/notebooks/`.
4. Edits autosave to Drive; push back with **File → Save a copy in GitHub**.

### Option B — Direct GitHub link

Paste a URL like this into Colab's "GitHub" tab (swap in your username):

```
https://github.com/YOUR_USER/csot-ml-astronomy/blob/main/ML-Astronomy/Week-3/notebooks/week3_cnn_starter.ipynb
```

### Option C — Upload

Download the `.ipynb` from this folder, then in Colab use **File → Upload notebook**.

---

## Submission

Submission logistics are confirmed via the official CAIC channels. Default expectation:

- Push both completed notebooks into a `submissions/<your-name>/week3/` folder in your fork, **or**
- Share Colab "share links" (with comment access) with your mentor.

> **Do not commit `galaxy_model.pth` or the `galaxy_data/` images to the repo.** This is a teaching repository with no binary assets — save weights to your Drive or attach them to your submission another way. Mentors can re-train from your notebook.

Ask in the mentor channel **before** the week ends if anything is unclear.

---

## Reflection Questions

Write 2–3 sentences each in a Markdown cell at the bottom of the relevant notebook:

1. What test accuracy did your CNN reach, and by how much did it beat the Week-2 baseline? What does that gap say about the value of preserving spatial structure?
2. Did your train and validation curves show overfitting? How could you tell, and what would you try to reduce it?
3. Which two classes did the confusion matrix muddle most? Tie that confusion to the astronomy from [`04`](04-spiral-structure-and-star-formation.md) and [`08`](08-lenticulars-mergers-and-evolution.md) — is it a model failure, a real physical ambiguity, or both?
4. You saved and reloaded the model and the accuracy matched. Why is that round-trip check worth doing rather than assuming it worked?

These won't be graded — they exist so you can look back at the end of the track and see how far you've come.

---

⬅️ Previous: [`08-lenticulars-mergers-and-evolution.md`](08-lenticulars-mergers-and-evolution.md) | 📚 See also: [`resources.md`](resources.md)
