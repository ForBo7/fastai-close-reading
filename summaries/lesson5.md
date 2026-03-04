We're at the stage now where we go deeper into how these networks actually work. The lesson returns to the Titanic dataset — each row a real passenger, columns recording survival, class, sex, age, family, fare, and embarkation city — and builds a linear model, a neural net, and then a deep learning model entirely from scratch in Python and PyTorch, before showing why you'd use a framework in practice, and finally introducing the machinery behind random forests.

The notebook begins with data cleaning. Calling `df.isna().sum()` reveals that Age is sometimes missing, Cabin is almost always missing, and Embarked has two gaps. Rather than dropping rows or columns (which you should basically never do), every missing value is replaced with the mode of its column — the simplest imputation that always works and is good enough nearly all the time. A quick `describe()` reveals that Fare has a long-tailed distribution, so we take the log (using `np.log1p` to handle zeros), turning the skewed histogram into something much more sensibly centered. The rule of thumb: anything with a dollar sign is a candidate for logging. Categorical columns — Sex, Pclass, Embarked — are turned into dummy variables with `pd.get_dummies`, producing one boolean column per level. Jeremy notes he prefers keeping all n levels rather than n−1, so there's no need for a separate constant term.

With all columns now numeric, the data is converted into PyTorch tensors. A key concept is introduced: the **shape** of a tensor and its **rank** (number of dimensions). Random coefficients are created, and then comes the line `indeps * coeffs` — element-wise multiplication via **broadcasting**, an idea from APL that lets a length-12 vector multiply across all 891 rows of a 891×12 matrix in one optimized operation. Before summing, the products are inspected: Age values dominate because their magnitude dwarfs the 0/1 dummies. The fix is normalization — dividing each column by its maximum — so all values sit in a comparable range.

Summing across columns gives predictions; comparing predictions to actuals via mean absolute error gives a loss. Both steps are wrapped into small functions (`calc_preds`, `calc_loss`). Jeremy emphasizes his coding style: explore interactively, step by step, then copy-paste those steps into a function — and keep the exploration in the notebook as documentation for your future self and colleagues.

Gradient descent is demonstrated one step at a time. Calling `requires_grad_()` on the coefficients tells PyTorch to track operations. After computing the loss, `loss.backward()` fills in `.grad` on the coefficients. Subtracting `grad * lr` in-place with `sub_()` completes one gradient descent step, and the loss drops from 0.54 to 0.52.

Training is then wrapped into `train_model()`: initialize random coefficients, loop over epochs, print the loss each time. After 18 epochs the loss falls to about 0.295 and accuracy reaches ~79%. The coefficients are inspected: Age is negative (older → less likely to survive), Sex_male is negative — both sensible.

The **sigmoid function** σ(x) = 1/(1+e^(−x)) is introduced as a way to squish predictions into the 0–1 range, which makes optimization much easier for a binary target. Redefining `calc_preds` to apply `torch.sigmoid` at the end — leveraging Python's dynamic dispatch so the rest of the code doesn't change — allows the learning rate to jump from 0.1 to 2, and accuracy improves to nearly 83%. The lesson emphasizes: whenever you have a binary dependent variable, always put a sigmoid at the very end.

The model is submitted to Kaggle, landing right at the 50th percentile — a respectable baseline from a linear model built entirely from scratch.

Next, `calc_preds` is rewritten to use matrix multiplication (`@`) instead of element-wise multiply plus sum, and the coefficients become a column vector (n_coeff × 1 matrix) rather than a 1-D vector. The dependent variable gains a trailing unit axis via `[:, None]`. This changes nothing mathematically but sets up the transition to a neural network.

The neural network adds a hidden layer of 20 units: the first weight matrix maps from input to hidden, ReLU zeros out negatives, then a second matrix maps from hidden to 1 output, a constant term is added, and sigmoid is applied. Getting the initialization right is fiddly — dividing the first layer's random values by `n_hidden`, subtracting 0.3 instead of 0.5 for the second layer. The result: about the same accuracy, confirming that for this tiny dataset a neural net doesn't magically outperform a linear model.

Deep learning generalizes this to n hidden layers. `init_coeffs` now creates a list of weight matrices and constant terms for an arbitrary number of layers, and `calc_preds` loops through them — matrix multiply, add constant, ReLU (except the last layer, which uses sigmoid). Again the accuracy is about the same. Jeremy notes that deep learning isn't necessarily better for very small tabular datasets with few simple features.

The lesson then shifts to using **fastai as a framework**. Feature engineering is borrowed from a Kaggle tutorial — extracting Deck from Cabin, computing Family size, Alone flag, TicketFreq, and Title from Name. A `TabularPandas` object handles all preprocessing (categorification, missing-value imputation, normalization) in one call. `lr_find()` sweeps learning rates to find a good range, and `learn.fit()` trains the model. At inference time, `test_dl()` ensures test-set preprocessing exactly matches training — one line of code that eliminates a whole class of bugs.

**Ensembling** is introduced: train five models with different random initializations, average their predictions, and submit. This simple trick takes the submission from the 50th percentile to the top 25%.

The final act introduces **random forests**, starting from first principles. A **binary split** divides passengers into two groups — for instance, male vs. female — and checks whether the groups have very different survival rates. The quality of a split is scored by the weighted standard deviation of the dependent variable on each side (lower is better). Trying all columns and all thresholds, Sex emerges as the single best split — a result known as **1R** (one rule), which a 1990s review found competitive with far more complex methods. Jeremy notes that he was "Mr. Random Forests" for years and that they're extremely hard to mess up compared to, say, logistic regression. The lesson ends having laid the groundwork for next week: recursively applying binary splits to build decision trees, and then ensembling trees into a random forest.

---

**Lesson Challenges**

- Build a linear model, single-hidden-layer neural net, and deep learning model from scratch on the Titanic dataset
- Submit predictions to the [Titanic Kaggle competition](https://www.kaggle.com/competitions/titanic)
- Experiment with the interactive binary-split widget to find good split points for each column

**Potential Research Directions**

- The [OneR paper](https://link.springer.com/article/10.1023/A:1022631118932) and when the simplest possible model is competitive
- Chris Deotte's approach of using only the Name column for Titanic (feature engineering from string data)
- Initialization strategies for from-scratch neural networks (why subtracting 0.3 vs 0.5 matters)
- Broadcasting semantics and their origin in APL / Ken Iverson's notation
- Comparison of imputation methods beyond mode (and fastai's trick of adding a boolean "was this missing?" column)

**Homework**

- Go through the clean notebook: before running each cell, predict why it's there and what output to expect; if unsure, change things and see what happens; if still stuck, search the forums or ask
- Become familiar with the basic APIs of PyTorch, NumPy, and pandas (Jeremy recommends Wes McKinney's book for pandas)
- Look up and understand NumPy broadcasting rules
- Copy the deep learning `calc_preds` code out of its function, split into cells, and run layer by layer to understand the shapes

**Things Jeremy Says You Should Do**

- Never throw away data — don't drop rows or columns with missing values
- Always start with the simplest possible approach; don't do complicated things until you know you need them
- Keep your exploratory steps in the notebook as documentation rather than deleting them
- Run your models a few times without a fixed seed to build intuition for how stable your results are
- Always include a simple baseline (like 1R) alongside more complex models
- Eyeball your coefficients after training to sanity-check that they make sense
- For binary targets, always apply sigmoid as the final activation — and check that your framework is doing this
- Look up NumPy/PyTorch broadcasting rules and make sure you understand them

**Resources**

- [Linear model and neural net from scratch](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch) (Kaggle notebook)
- [Why you should use a framework](https://www.kaggle.com/code/jhoward/why-you-should-use-a-framework) (Kaggle notebook)
- [How random forests really work](https://www.kaggle.com/code/jhoward/how-random-forests-really-work/) (Kaggle notebook)
- [Chapter 4 — fastbook](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb)
- [Chapter 9 — fastbook](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb)
- [Deep Learning for Coders with fastai and PyTorch](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) (the book)
- [OneR paper](https://link.springer.com/article/10.1023/A:1022631118932) (Holte, 1993)
- Titanic notebooks: [Exploring Survival on the Titanic](https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic) · [Titanic WCG XGBoost](https://www.kaggle.com/code/cdeotte/titanic-wcg-xgboost-0-84688/notebook) · [Divide and Conquer](https://www.kaggle.com/code/pliptor/divide-and-conquer-0-82296) · [Titanic Using Name Only](https://www.kaggle.com/code/cdeotte/titanic-using-name-only-0-81818/notebook)
- [fast.ai forums](https://forums.fast.ai/)
- Wes McKinney's *Python for Data Analysis* (recommended for pandas/NumPy fundamentals)
- [SymPy](https://www.sympy.org/) (symbolic math in Python, used in the lesson to plot the sigmoid curve)
