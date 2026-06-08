# Week 2 — Curated Resources

> The links in each numbered concept markdown are the *minimum* set you need. This page gathers extras for the curious, organised by theme. Nothing here is required; everything here is good.

---

## Scikit-Learn & Classical ML (Part 1)

### Official
- 📘 [scikit-learn — Getting Started](https://scikit-learn.org/stable/getting_started.html) — the `(n_samples, n_features)` mental model.
- 📘 [scikit-learn — `KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) and the [Nearest Neighbors guide](https://scikit-learn.org/stable/modules/neighbors.html).
- 📘 [scikit-learn — Logistic Regression guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
- 📘 [scikit-learn — `DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) — your random/majority floors.
- 📘 [scikit-learn — Model evaluation (accuracy, confusion matrix, and beyond)](https://scikit-learn.org/stable/modules/model_evaluation.html).

### Friendlier intros
- 📺 [StatQuest — K-nearest neighbors, clearly explained](https://www.youtube.com/watch?v=HVXime0nQeI).
- 📺 [StatQuest — Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8).
- 📘 [Google — Machine Learning Crash Course: classification](https://developers.google.com/machine-learning/crash-course/classification/video-lecture).
- 📘 [Wikipedia — Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) — why distances weaken in 12 288-D.

---

## PyTorch Models, Losses & Optimisers (Part 2)

### Official
- 📘 [PyTorch — Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).
- 📘 [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) and [`torch.nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
- 📘 [`torch.nn` — all layers and activations](https://docs.pytorch.org/docs/stable/nn.html).
- 📘 [`nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — read the "expects unnormalized logits" note.
- 📘 [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html) and the [Optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

### Friendlier intros
- 📺 [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk).
- 📺 [3Blue1Brown — Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w).
- 📺 [StatQuest — Cross Entropy clearly explained](https://www.youtube.com/watch?v=6ArSys5qHAU).
- 📘 [Daniel Bourke — Learn PyTorch: neural network classification](https://www.learnpytorch.io/02_pytorch_classification/) — free, hands-on.
- 📘 [CS231n — Optimisation notes](https://cs231n.github.io/optimization-1/) and [linear classification](https://cs231n.github.io/linear-classify/).

### Reference
- 📄 [Kingma & Ba 2014 — Adam optimiser (arXiv)](https://arxiv.org/abs/1412.6980).
- 📘 [Wikipedia — Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem).
- 📘 [Wikipedia — Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).

---

## Surface Brightness, Isophotes & the Sérsic Profile (Astronomy)

- 📘 [Wikipedia — Surface brightness](https://en.wikipedia.org/wiki/Surface_brightness) (and its distance-independence).
- 📘 [Wikipedia — Isophote](https://en.wikipedia.org/wiki/Isophote).
- 📘 [Wikipedia — Sérsic profile](https://en.wikipedia.org/wiki/Sersic_profile) — plots for several `n`.
- 📘 [Wikipedia — de Vaucouleurs' law](https://en.wikipedia.org/wiki/De_Vaucouleurs%27s_law) — the `n = 4` case.
- 📄 [Graham & Driver 2005 — A concise reference to the Sérsic model (arXiv)](https://arxiv.org/abs/astro-ph/0503176).
- 📄 [Sérsic 1963 — the original profile paper (ADS)](https://ui.adsabs.harvard.edu/abs/1963BAAA....6...41S/abstract).
- 📘 [Caltech Ay127 — galaxy photometry lecture notes](https://www.astro.caltech.edu/~george/ay127/).

---

## Stellar Populations & Galaxy Colour (Astronomy)

- 📘 [Wikipedia — Stellar classification](https://en.wikipedia.org/wiki/Stellar_classification) — the blue-massive / red-low-mass story.
- 📘 [Las Cumbres Observatory — the lives of stars](https://lco.global/spacebook/stars/) and [galaxies](https://lco.global/spacebook/galaxies/).
- 📘 [Wikipedia — Red and dead / quenching](https://en.wikipedia.org/wiki/Galaxy_color%E2%80%93magnitude_diagram) — the colour–magnitude bimodality.
- 📺 [PBS Space Time — galaxy colours and stellar populations](https://www.youtube.com/@pbsspacetime).
- 📺 [Dr. Becky — galaxies playlist](https://www.youtube.com/@DrBecky/playlists).

### Image galleries (build intuition)
- 🖼️ [ESA/Hubble image archive — Galaxies](https://esahubble.org/images/archive/category/galaxies/).
- 🌌 [SDSS SkyServer Explore Tool](https://skyserver.sdss.org/) — inspect real galaxies and their catalogued fits.
- 🖼️ [Astronomy Picture of the Day archive](https://apod.nasa.gov/apod/archivepix.html).

---

## ML × Astronomy (Where We're Headed)

- 📄 [Dieleman, Willett, Dambre 2015 — Rotation-invariant CNNs for Galaxy Morphology (arXiv)](https://arxiv.org/abs/1503.07077) — note how their CNN beats any flatten-and-classify baseline.
- 📄 [Walmsley et al. — Galaxy Zoo DECaLS (arXiv search)](https://arxiv.org/a/walmsley_m_1).
- 📘 [`astroNN` — ML in astronomy in Python](https://github.com/henrysky/astroNN).
- 📘 [`statmorph` — non-parametric morphology + Sérsic fitting](https://statmorph.readthedocs.io/) — the tool astronomers use to measure `n`, concentration, asymmetry.

---

## Communities

- 💬 [PyTorch discussion forums](https://discuss.pytorch.org/) — official, very active.
- 💬 [scikit-learn discussions / Stack Overflow `scikit-learn` tag](https://stackoverflow.com/questions/tagged/scikit-learn).
- 💬 [r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/).
- 💬 [Astronomy Stack Exchange](https://astronomy.stackexchange.com/).
- 💬 The official **CAIC CSoT channels** — your first stop for track-specific questions and mentor help.

---

## How to Use This List

You will not read all of this. Suggested workflow for Week 2:

1. **Part 1:** skim the scikit-learn [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html) and [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) guides, then do the self-checks in pages 01–02.
2. **Watch** one StatQuest video (KNN or Cross Entropy) for intuition.
3. **Part 2:** read the PyTorch [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) tutorial end-to-end, then do the self-checks in pages 04–05.
4. **Astronomy:** skim the [Sérsic profile](https://en.wikipedia.org/wiki/Sersic_profile) page and look at two galaxies on [SkyServer](https://skyserver.sdss.org/).
5. **Bookmark** this page; come back as questions arise.

⬅️ Back to [Week 2 README](README.md)
