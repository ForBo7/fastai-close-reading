The lesson opens by picking up where Lesson 5 left off — the Titanic dataset and the OneR algorithm, which finds the single best binary split. Jeremy quickly extends this into a TwoR model: take the two groups from the best split (males and females), remove the splitting variable, and run OneR again on each group separately. For males, the biggest predictor of survival turns out to be Age (younger or older than six); for females, it's Pclass. This gives a decision tree — a series of binary splits that gradually partition the data into leaf nodes with increasingly strong predictions. Doing this manually would be tedious, so Jeremy introduces scikit-learn's `DecisionTreeClassifier`, which automates the process entirely.

The tree visualization is walked through carefully. Sex ≤ 0.5 looks odd, but binary features are coded as 0 or 1. Each node shows the number of samples, the value breakdown (died vs. survived), and a metric called **Gini impurity** — a measure of how mixed a node is. In the binary case, Gini is $1 - p^2 - (1-p)^2$: a perfectly pure node has Gini 0, and a 50-50 node has Gini 0.5. The tree's splits are identical to the manual ones, confirming that the algorithm found the same structure. On this small dataset, the four-leaf tree's accuracy (0.224 MAE) is actually slightly worse than OneR (0.215), illustrating the noise floor of small data. A deeper tree with `min_samples_leaf=50` yields a richer, uneven structure — some branches go deeper than others — and achieves 0.183 MAE.

Jeremy then submits to Kaggle, emphasizing a core philosophy: **submit early and often**, regardless of quality. Decision trees require almost no preprocessing — no dummy variables, no normalization, no log transforms. All a tree cares about is the ordering of values, so monotonic transforms like log are irrelevant to the splits. This robustness is why Jeremy recommends decision tree–based approaches as the default starting point for any tabular problem.

The lesson pivots to **bagging**, Leo Breiman's key insight. A single tree is imperfect but unbiased — on average, it predicts correctly without systematic over- or under-estimation. If you build many such trees, each on a different random subset of data, their errors are uncorrelated. Averaging uncorrelated, unbiased predictions yields an estimate whose average error is zero — the errors cancel out. This is the theoretical foundation of the **Random Forest**: build many decision trees on random subsets, average their predictions. Jeremy builds one from scratch in about seven lines of Python — `get_tree()` samples 75% of rows, fits a `DecisionTreeClassifier`, and a list comprehension creates 100 trees. Stacking and averaging their predictions gives the Random Forest prediction. The "real" version also randomly selects a subset of columns at each split point, further decorrelating the trees. Scikit-learn's `RandomForestClassifier` wraps this up neatly.

**Feature importance** is demonstrated next. For each tree, at each split, the Gini improvement is recorded per column. Summing across all trees produces a feature importance plot showing which variables drive predictions. On the Titanic data, Sex dominates, Pclass is a distant second, and everything else barely matters. Jeremy recounts using this technique on a credit scoring dataset with 7,000 columns — a Random Forest identified the ~30 useful columns in two hours, matching the results of a multi-year, multi-million-dollar consulting project.

The lesson covers how the number of trees affects accuracy: more trees always help, with diminishing returns. Jeremy's rule of thumb is rarely more than 100 trees. **Out-of-Bag (OOB) error** is introduced — since each tree uses only ~75% of the data, the remaining 25% serves as a built-in validation set for that tree. Averaging predictions across trees where a given row was *not* in the training set gives the OOB error, sometimes eliminating the need for a separate validation set.

Five model interpretation techniques are laid out: (1) **prediction confidence** via standard deviation across trees, (2) **explaining individual predictions** via tree interpreter (tracking Gini changes along the path for a single row), (3) **feature importance** (already covered), (4) **redundant feature detection**, and (5) **partial dependence plots**. The partial dependence technique is model-agnostic: set one column to a fixed value across all rows, predict, repeat for each value, and plot the averages. This isolates the effect of one variable while holding everything else at its actual observed values, avoiding confounding. The Blue Book for Bulldozers dataset is used to show that newer bulldozers fetch higher prices, with the relationship clearly visualized.

Jeremy then introduces **Gradient Boosting** as an alternative to bagging. Instead of building full trees on random subsets and averaging, boosting builds very small trees sequentially — each one predicting the residual error of all previous trees — and sums their predictions. Gradient Boosting Machines (GBMs) are generally more accurate than Random Forests but can overfit, making them less safe as a first choice. Jeremy points to the explain.ai three-part series on gradient boosting for deeper understanding.

The second half of the lesson shifts to a Kaggle walkthrough: the **Paddy Disease Classification** competition. Jeremy demonstrates his iterative workflow — start with the fastest possible model, submit immediately, then improve. He uses `fastkaggle` to automate data downloading and notebook publishing. The `fastcore.parallel` module quickly checks image sizes across the dataset.

Image preprocessing is explored: squish (distorts aspect ratio), crop (loses content), and pad with zeros (preserves everything, adds blank pixels). All three perform similarly here. Jeremy picks `resnet26d` from the "Best Vision Models for Fine-Tuning" study (a collaboration with Thomas Capelle testing ~100 architectures from Ross Wightman's timm library) — it's the fastest option, good for rapid iteration even if not the most accurate. The learning rate finder is run, Jeremy picks 0.01 (more aggressive than the conservative built-in suggestions), and fine-tunes for three epochs in about a minute.

The first Kaggle submission lands in the bottom 20% — expected for one minute of training. Jeremy pre-resizes images to fix a 4× speed bottleneck on Kaggle's two-vCPU machines, then switches to `convnext_small` — which the benchmarking study showed has the best overall speed-accuracy trade-off — cutting error rate from 12% to 4.5%. **Test Time Augmentation (TTA)** further improves this to ~3.9% by averaging predictions over multiple augmented versions of each test image. Using larger rectangular images (matching the dataset's native 4:3 aspect ratio) and 12 epochs of training drops error below 2%. A vectorized pandas/numpy indexing trick for mapping prediction indices to disease name strings is shown as an example of the kind of fluency worth developing.

The lesson closes with a Q&A covering TTA mechanics (it's specifically an inference technique that reuses training augmentation), why squares are the default (diverse aspect ratios in most datasets), and why zero-padding often works better than reflection padding (the model prefers a straightforward "this is the edge" signal). Jeremy explains his aversion to AutoML and hyperparameter sweeps — preferring intentional, hypothesis-driven experimentation — and his low-tech notebook organization: duplicate, rename, annotate failures, use the file system.

---

- **Lesson Challenges**
  - Build a Random Forest from scratch (sampling subsets, building trees, averaging predictions)
  - Implement feature importance by tracking Gini improvement per column across trees
  - Enter the [Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification) Kaggle competition and iterate to improve your score

- **Potential Research Directions**
  - Leo Breiman, ["Statistical Modeling: The Two Cultures"](https://www.semanticscholar.org/paper/Statistical-modeling%3A-The-two-cultures-Breiman/e5df6bc6da5653ad98e754b08f63326c2e52b372) — the philosophical case for algorithmic modeling over data modeling
  - Judea Pearl & Dana Mackenzie, *The Book of Why* — causality vs. correlation in model interpretation
  - Partial dependence plots for deep learning models and two-way interactions
  - Batching images by aspect ratio rather than forcing squares
  - Ensembling deep learning models via bagging (Jeremy notes this is rarely done but viable)
  - The explain.ai [Gradient Boosting series](https://explained.ai/gradient-boosting/) for deeper GBM understanding

- **Homework**
  - Complete the [How Random Forests Really Work](https://www.kaggle.com/code/jhoward/how-random-forests-really-work/) notebook
  - Work through the [Road to the Top, Part 1](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1) notebook
  - Read [Chapter 9](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb) of the fastbook
  - Submit to the Paddy Disease Classification competition on Kaggle

- **Things Jeremy Says You Should Do**
  - **Submit to Kaggle every day**, no matter how rough — you need to see how you're going
  - Start every tabular project with a Random Forest — they're nearly impossible to mess up and give fast insight
  - Create a feature importance plot as early as possible to find which columns matter
  - Use `show_batch()` on every dataset to see what the data looks like before modeling
  - Jump into modeling quickly — models are a great way to understand your data
  - Iterate rapidly: train models in ~1 minute so you can try 80 things, not one perfect thing
  - Create public Kaggle notebooks to practice communicating results clearly
  - Organize notebooks with careful naming, duplication, and folders — low-tech but effective
  - Spend time playing with NumPy/pandas vectorized indexing — it's a pattern worth mastering
  - Think like a scientist: have hypotheses, test them, draw conclusions — don't just grid-search blindly

- **Resources**
  - **Notebooks**
    - [How Random Forests Really Work](https://www.kaggle.com/code/jhoward/how-random-forests-really-work/)
    - [Road to the Top, Part 1](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)
    - [Small Models: Road to the Top, Part 2](https://www.kaggle.com/code/jhoward/small-models-road-to-the-top-part-2)
    - [The Best Vision Models for Fine-Tuning](https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning)
  - **Book**: [Chapter 9 — Tabular Modeling Deep Dive](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb)
  - **Readings**
    - [How to Explain Gradient Boosting](https://explained.ai/gradient-boosting/) (Jeremy Howard & Terence Parr)
    - [Statistical Modeling: The Two Cultures](https://www.semanticscholar.org/paper/Statistical-modeling%3A-The-two-cultures-Breiman/e5df6bc6da5653ad98e754b08f63326c2e52b372) — Leo Breiman
    - *The Book of Why* — Judea Pearl & Dana Mackenzie
  - **Libraries**: scikit-learn, timm (PyTorch Image Models), fastcore.parallel, fastkaggle, Tree Interpreter
  - **Competition**: [Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification)
  - **Course page**: [Lesson 6](https://course.fast.ai/Lessons/lesson6.html)
  - **Forums**: [Lesson 6 official topic](https://forums.fast.ai/t/lesson-6-official-topic/100561)
