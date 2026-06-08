# 02 — A Baseline with Scikit-Learn

> Now that our galaxies are flattened NumPy rows, we train the simplest classifiers that exist — **K-Nearest Neighbours** and **Logistic Regression** — and record their test accuracy. That single number is the **baseline**: the bar every neural network in the rest of the track must clear to justify its complexity.

---

## What "Baseline" Means

In machine learning, a **baseline** is a deliberately simple model you build *first*, on purpose, so you have something to compare against. It's the control group of your experiment.

A good baseline is:

- **Simple** — a few lines, no tuning, no GPU.
- **Honest** — evaluated on data the model never trained on.
- **Cheap** — runs in seconds.

The two we'll use are classics from the scikit-learn toolbox. Neither knows anything about images; they just see rows of 12 288 numbers.

---

## Two Reference Points Before Any Model

Even before KNN, know the two "do-nothing" baselines so you can sanity-check everything else:

| "Model" | What it does | Accuracy you'd expect |
|---|---|---|
| **Random guess** | Picks a class uniformly at random. | `1 / num_classes` (e.g. 33% for 3 balanced classes). |
| **Majority class** | Always predicts the most common class. | `count(biggest class) / total`. |

scikit-learn ships this as `DummyClassifier`:

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print("Majority-class accuracy:", dummy.score(X_test, y_test))
```

> **Why this matters.** If your classes are imbalanced — say 70% spirals — a model that *only ever says "spiral"* already scores 70%. Any real model must beat **that**, not 50%. Always know your majority-class accuracy before celebrating.

---

## Classifier 1 — K-Nearest Neighbours (KNN)

KNN is the most intuitive classifier in existence. To classify a new galaxy:

1. Measure the distance from its 12 288-number vector to **every** galaxy in the training set.
2. Find the `k` closest training galaxies (its "nearest neighbours").
3. Take a **majority vote** of their labels. That vote is the prediction.

That's the entire algorithm. There is no "training" beyond memorising the training set.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)   # k = 5
knn.fit(X_train, y_train)                    # just memorises the data
acc = knn.score(X_test, y_test)              # fraction correct on the test set
print(f"KNN accuracy: {acc:.3f}")
```

### What KNN "sees" in our images

Two galaxy images are "close" in KNN's world if their flattened pixel vectors have similar values position-by-position. This picks up coarse signals — overall brightness, overall colour, how much of the frame is lit — but it is extremely fragile:

- Shift a galaxy a few pixels and every pixel value changes; the "distance" can balloon even though it's the same galaxy.
- Rotate it and, again, the vector changes wildly while the morphology is identical.

This fragility is *exactly* the spatial-blindness from [`01-tensors-to-numpy-and-flattening.md`](01-tensors-to-numpy-and-flattening.md), showing up as a concrete weakness. KNN has no notion that a shifted or rotated galaxy is "the same thing".

> **The curse of dimensionality.** In 12 288 dimensions, "distance" becomes a weak signal — almost everything is far from everything else, and nearest neighbours aren't reliably *near* in any meaningful sense. KNN works far better on a few dozen features than on twelve thousand raw pixels. This is another reason the baseline will be beatable.

---

## Classifier 2 — Logistic Regression

Despite the name, **logistic regression is a classifier**, not a regressor. It learns a **weighted sum** of the input features for each class, then turns those sums into class probabilities. For our flattened images it learns one weight per pixel-channel per class — a linear decision boundary in 12 288-dimensional space.

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)   # more iters: high-dim data is slow to converge
logreg.fit(X_train, y_train)
acc = logreg.score(X_test, y_test)
print(f"Logistic Regression accuracy: {acc:.3f}")
```

A few practical notes:

- `max_iter=1000` (or more) is often needed — with 12 288 features the default 100 iterations may not converge, and sklearn will warn you.
- It helps to have the inputs on a similar scale. Our Week-1 `Normalize([0.5]*3,[0.5]*3)` already roughly centres them, so we're fine.
- Logistic regression is, in a real sense, **a single-layer neural network with no hidden layer** — a stack of linear weights followed by a softmax. Keep that in your back pocket: when we build an MLP in [`04-building-models-with-nn-module.md`](04-building-models-with-nn-module.md), we're essentially adding hidden layers *on top of* this idea.

---

## Measuring Accuracy Honestly

**Accuracy** is the simplest metric: the fraction of test samples the model labels correctly.

```
accuracy = (number of correct predictions) / (total predictions)
```

The single most important rule of evaluation:

> **Never score a model on data it trained on.** A KNN that memorised the training set scores ~100% on it and tells you *nothing*. Always report accuracy on the held-out **test** set — galaxies the model has never seen.

This is why Week 1 split the data into `train` / `val` / `test`. For the baseline we fit on `train` and score on `test`. (We'll bring `val` into play in Week 3 when we have hyperparameters to tune.)

### Beyond a single number (preview)

Accuracy hides *where* a model fails. A model can be 75% accurate while getting every single irregular galaxy wrong, if irregulars are rare. The tool that reveals this is the **confusion matrix** — a grid of predicted-vs-true labels — which we build properly in Week 3. For now, accuracy is enough to set the bar.

---

## What Numbers to Expect

Exact values depend on your class set and how many images per class you sampled, but the *shape* of the result is reliable:

| Model | Rough accuracy on flattened 64×64 galaxies | Interpretation |
|---|---|---|
| Random guess | `1 / num_classes` | The absolute floor. |
| Majority class | `largest class fraction` | The "lazy" floor. |
| KNN (k=5) | a bit above majority | Captures coarse colour/brightness; blind to shape. |
| Logistic Regression | similar to / slightly above KNN | A linear boundary on raw pixels. |
| **CNN (Week 3)** | **clearly higher** | Sees spatial structure — this is the payoff. |

The point is **not** to make the baseline good. The point is to make it *honest and reproducible*, so that when the CNN beats it in Week 3, you can prove the improvement came from architecture, not luck.

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Accuracy ≈ 100% on training data, much lower on test | You reported the *training* score. | Always call `.score(X_test, y_test)`. |
| `ConvergenceWarning` from LogisticRegression | Too few iterations for 12 288 features. | Raise `max_iter` (1000+); ensure inputs are normalised. |
| KNN is painfully slow | It compares against every training point at predict time. | Use a smaller subset, or fewer features (e.g. downsample images). |
| Suspiciously high accuracy | Train/test leak, or you forgot images are normalised consistently. | Audit your splits; use the same transform for train and test. |
| Baseline "beats" your later CNN | CNN bug, or baseline benefiting from a trivial cue (e.g. brightness). | Re-check the CNN; inspect a confusion matrix in Week 3. |

---

## Quick Self-Check

1. What is a baseline, and why build one before a neural network?
2. Your three classes split 60/30/10. What's the majority-class accuracy, and why must a real model beat *that* and not 33%?
3. In one sentence, how does KNN make a prediction?
4. Why is logistic regression sometimes described as "a neural network with no hidden layer"?
5. Why must accuracy be measured on the test set, never the training set?

<details>
<summary>Answers</summary>

1. A deliberately simple model used as a comparison point; it tells you how hard the problem is and gives a number every fancier model must beat.
2. 60% (the biggest class). A model that always predicts the majority class already scores 60%, so beating random (33%) isn't enough — you must beat the lazy 60%.
3. It finds the `k` training samples closest to the new sample and takes a majority vote of their labels.
4. Because it's just a linear weighted sum of inputs followed by a softmax — exactly one layer of weights, with no intermediate ("hidden") layer in between.
5. Because a model can memorise its training data and score near 100% on it without having learned to generalise; only held-out test data gives an honest estimate.

</details>

---

## External Resources

- 📘 [scikit-learn — Nearest Neighbors guide](https://scikit-learn.org/stable/modules/neighbors.html) and [`KNeighborsClassifier` docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
- 📘 [scikit-learn — Logistic Regression guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
- 📘 [scikit-learn — `DummyClassifier` docs](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).
- 📘 [scikit-learn — model evaluation: accuracy and beyond](https://scikit-learn.org/stable/modules/model_evaluation.html).
- 📺 [StatQuest — K-nearest neighbors, clearly explained](https://www.youtube.com/watch?v=HVXime0nQeI).
- 📺 [StatQuest — Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8).
- 📘 [Wikipedia — Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) — why distances weaken in 12 288 dimensions.

---

⬅️ Previous: [`01-tensors-to-numpy-and-flattening.md`](01-tensors-to-numpy-and-flattening.md) | ➡️ Next: [`03-surface-brightness-and-isophotes.md`](03-surface-brightness-and-isophotes.md)
