This final lesson of Part One picks up where Lesson 7 left off—inside the collaborative filtering notebook—and uses it as a launchpad to demystify what's really happening under the hood of PyTorch modules, embeddings, and convolutional neural networks.

The lesson opens by stressing that you should be comfortable with the `05_linear_model_and_neural_net_from_scratch` notebook before proceeding, because everything from here builds abstractions on top of functionality already built by hand. Jeremy shows a class `T` subclassing `Module` with an attribute `a = torch.ones(3)`. Asking for its `.parameters()` returns an empty list—PyTorch doesn't know those are trainable. The fix is wrapping them in `nn.Parameter`, a near-trivial class whose only real job is to signal "this tensor should be optimized." Once wrapped, the parameter appears in `.parameters()` with `requires_grad=True` by default. Most of the time you never call `nn.Parameter` directly because building blocks like `nn.Linear` handle it internally.

From there, Jeremy builds an embedding from scratch. A helper function `create_params(size)` creates a zero tensor, fills it in-place with small normally-distributed random values via `.normal_()`, and wraps it in `nn.Parameter`. The `DotProductBias` module stores user factors, movie factors, user bias, and movie bias as parameters, and its `forward()` indexes into them, computes dot products, adds biases, and applies `sigmoid_range`. Training confirms it works identically to the PyTorch `Embedding`-based version from Lesson 7.

The lesson then turns to interpreting what the trained model actually learned. Sorting `movie_bias` reveals the lowest-bias movies are widely disliked films like "Lawnmower Man 2"—movies people rate poorly even relative to their genre. Sorting descending yields universally appreciated films like "L.A. Confidential" and "Silence of the Lambs"—movies people enjoy even when they don't normally like that genre. The 50-dimensional `movie_factors` matrix is compressed to two dimensions via PCA and plotted: one axis roughly separates mainstream blockbusters from critically acclaimed films, the other separates action/sci-fi from dialogue-driven dramas. None of this genre information was in the training data—SGD discovered it by optimizing predictions.

Jeremy then demonstrates fast.ai's `collab_learner`, which produces an `EmbeddingDotBias` model with nearly identical source code to the from-scratch version. Embedding distance via `CosineSimilarity` finds that "Dial M for Murder" is closest to "Silence of the Lambs"—a sensible result. The bootstrapping problem (cold-start recommendations) is mentioned but deferred to the book.

Collaborative filtering can also be done with a neural network instead of dot products. The `CollabNN` module concatenates user and item embeddings and passes them through a simple sequential network (linear → ReLU → linear → sigmoid). fast.ai's `get_emb_sz` provides a rule of thumb for embedding dimensions. Passing `use_nn=True` to `collab_learner` builds this automatically. The dot-product version performs better here because it encodes domain knowledge, but in practice companies often combine both approaches—the neural net component is especially useful when metadata about users or items is available. Jeremy notes a subtlety about collaborative filtering: small groups of obsessive users (like anime fans) can dominate recommendations, requiring normalization.

The discussion broadens to show that embeddings are not just for collaborative filtering. In NLP, words are assigned integer IDs and looked up in an embedding matrix—exactly the same operation. Jeremy demonstrates this in an Excel spreadsheet using the poem "I Am Sam," showing how each word gets mapped through a vocabulary lookup to a row in an embedding matrix, producing the numerical input a neural network needs. The same interpretation techniques (bias analysis, PCA visualization) apply to word embeddings.

Embeddings are equally central to tabular data. Jeremy revisits the bulldozer auction price prediction from Lesson 6, this time using a neural net. Categorical variables (product size, coupler system, tire size, etc.) each get their own embedding. `tabular_learner` creates a `TabularModel` that embeds every categorical variable, concatenates them with continuous variables, and passes everything through linear layers with batch norm and dropout. Jeremy walks through the `TabularModel` source code, noting how much of it is now recognizable. He references the "Entity Embeddings of Categorical Variables" paper by Guo and Berkhahn, whose third-place Kaggle team used exactly this technique with far less feature engineering than competitors. Strikingly, their region embeddings reproduced the geography of Germany without any location data, and their day-of-week and month embeddings clustered temporally adjacent values together. They also showed that trained embeddings can be extracted and fed into Random Forests or Gradient Boosted Trees for significant accuracy improvements.

The lesson then dives into convolutions—the heart of CNNs. Using an Excel spreadsheet on an MNIST digit (a handwritten 7), Jeremy shows that a convolution slides a small kernel (a 3×3 matrix of coefficients) across the image, computing a dot product at each position followed by a ReLU. A top-edge-detecting kernel has ones on top, zeros in the middle, and minus ones on the bottom—it produces high activations only where there's a transition from dark above to light below. A left-edge kernel transposes this idea. These are the kinds of features the first layer of a CNN learns.

The second convolutional layer takes multiple input channels (the outputs of the first layer) and uses a 3×3×(number of input channels) filter for each output channel. The filters are stacked to produce multiple output channels, combining features across input channels. In practice, filter values start random and are optimized by SGD.

Jeremy explains two approaches for reducing spatial dimensions. The older method is max pooling—a sliding window that takes the maximum value, typically 2×2, halving each dimension. The modern approach is stride-2 convolutions, where the kernel skips every other position, achieving the same size reduction while still learning parameters. At the very end, rather than a large dense layer, modern architectures use average pooling over the final grid (typically 7×7 for ImageNet) to produce one value per feature. fast.ai's concat pooling combines both max and average pooling for robustness—average pooling works well for objects that fill the frame, while max pooling catches small objects in a corner.

Matt Kleinsmith's visualization proves that a convolution is mathematically equivalent to a matrix multiplication where certain weights are forced to zero and others are tied together. The sliding-window implementation is faster, but conceptually it's just a constrained linear layer.

Dropout is introduced via another Excel sheet. A dropout mask is generated by comparing random floats against a threshold (the dropout probability). Multiplying the mask by the activations randomly zeroes out a fraction of them. Each mini-batch gets a different mask, forcing the network to learn robust representations rather than overfitting to specific activation patterns. Jeremy frames it as data augmentation for activations. The dropout probability controls the tradeoff: more dropout means worse training performance but better generalization. The technique was developed by Hinton's group (Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov), notably rejected from NIPS and disseminated via arXiv—one of the most impactful papers ever.

The lesson closes with a summary of neural network building blocks: inputs can be continuous numbers, one-hot encodings, or embeddings (a computational shortcut for one-hot); the middle consists of matrix multiplications (including convolutions and embeddings as special cases) sandwiched with activation functions (ReLU and others, which don't matter much in practice); the output uses softmax or sigmoid; and loss functions include cross-entropy, MSE, or MAE. Jeremy emphasizes that the internals of a wide range of neural networks are now understandable.

An extended AMA covers motivation ("you don't have to know everything—pick a sub-area"), the perception that bigger models are always needed (fast.ai's DAWNBench victory disproved this), homeschooling with apps like DragonBox Algebra 5+, live-coding walkthroughs as a teaching method, turning models into businesses via Eric Ries's Lean Startup approach (solve a real problem, fake it with an MVP, iterate), and Jeremy's productivity philosophy: spend half your time learning new things (building exponentially compounding expertise), don't overwork, get good sleep, and have the tenacity to finish things properly. He highlights community member Diganta Misra's creation of the Mish activation function as an example of what's possible. Jeremy recommends Radek Osmulski's book "Meta Learning" and urges students to practice, write blog posts, join study groups, and build projects before returning for Part 2.

---

- **Lesson Challenges**
  - Rebuild the embedding module from scratch using `nn.Parameter` and `create_params()`
  - Interpret trained collaborative filtering biases and latent factors
  - Visualize movie embeddings with PCA
  - Understand the convolution Excel spreadsheet (`conv-example.xlsx`) by tracing the formulas
  - Experiment with different dropout levels in the dropout spreadsheet

- **Potential Research Directions**
  - The bootstrapping / cold-start problem in recommendation systems (discussed in the book)
  - Combining dot-product and neural-net collaborative filtering with metadata
  - Entity embeddings for categorical variables in non-deep-learning models (Guo & Berkhahn, 2016)
  - Concat pooling (max + average) vs pure average pooling
  - The dominance problem in collaborative filtering (e.g., anime bias) and normalization techniques
  - PCA and other dimensionality reduction methods for interpreting embeddings
  - Convolution as constrained matrix multiplication—implications for architecture design

- **Homework**
  - Work through chapter 9 (tabular) and chapter 13 (convolutions) of the [fastbook](https://github.com/fastai/fastbook)
  - Complete the [chapter 8 questionnaire](https://forums.fast.ai/t/fastbook-chapter-8-questionnaire-solutions-wiki/69926)
  - Redo the `05_linear_model_and_neural_net_from_scratch` notebook if any part is unclear
  - Experiment with the [collaborative filtering](https://github.com/fastai/course22/blob/master/xl/collab_filter.xlsx) and [convolution](https://github.com/fastai/course22/blob/master/xl/conv-example.xlsx) spreadsheets

- **Things Jeremy Says You Should Do**
  - Make sure you're comfortable with the neural-net-from-scratch notebook before proceeding
  - Read Radek Osmulski's book [Meta Learning](https://rosmulski.gumroad.com/l/learn_machine_learning)
  - Practice and write—don't just watch videos on 2x speed
  - Watch the lessons multiple times; code along as you watch
  - Write blog posts about what you're learning
  - Spend time on the forums helping others and reading answers
  - Join or create a study group (e.g., on Discord or Zoom)
  - Build hobby projects and passion projects—don't just put the videos away
  - Spend significant time learning new tools and skills (Jeremy spends half his working day on this)
  - Don't overwork—get good sleep, eat well, exercise
  - Be tenacious: finish things properly rather than giving up early
  - Make the things you want to do easier so you do them more (tooling matters)
  - Like the video on YouTube to help others find the course
  - Help beginners on forums.fast.ai

- **Resources**
  - [Lesson 8 video](https://www.youtube.com/watch?v=htiNBPxcXgo)
  - [Lesson 8 course page](https://course.fast.ai/Lessons/lesson8.html)
  - [Collaborative Filtering Deep Dive notebook](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive/notebook)
  - [Collab filtering & embeddings spreadsheet](https://github.com/fastai/course22/blob/master/xl/collab_filter.xlsx)
  - [Convolution example spreadsheet](https://github.com/fastai/course22/blob/master/xl/conv-example.xlsx)
  - [Chapter 13: Convolutions (fastbook)](https://github.com/fastai/fastbook/blob/master/13_convolutions.ipynb)
  - [Chapter 8 questionnaire solutions](https://forums.fast.ai/t/fastbook-chapter-8-questionnaire-solutions-wiki/69926)
  - [Jeremy AMA thread](https://forums.fast.ai/t/jeremy-ama/97238)
  - [Data ethics bonus lesson](https://www.youtube.com/watch?v=krIVOb23EH8)
  - [Entity Embeddings of Categorical Variables (Guo & Berkhahn, 2016)](https://arxiv.org/abs/1604.06737)
  - [Dropout paper (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html)
  - [CNNs from different viewpoints (Matt Kleinsmith)](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)
  - [Computational Linear Algebra course (fast.ai)](https://github.com/fastai/numerical-linear-algebra)
  - [Meta Learning by Radek Osmulski](https://rosmulski.gumroad.com/l/learn_machine_learning)
  - [The Lean Startup by Eric Ries](https://theleanstartup.com/)
  - [DragonBox Algebra 5+](https://dragonbox.com/)
  - [Neural net from scratch notebook (05)](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch)
  - [fast.ai forums](https://forums.fast.ai/)
  - [Mish activation function (Diganta Misra)](https://arxiv.org/abs/1908.08681)
