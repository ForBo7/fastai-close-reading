The lesson opens by continuing the Rice Paddy competition through the lens of "Road to the Top," Parts 3 and 4, before pivoting into collaborative filtering for the second half.

The lesson begins by addressing a practical constraint: GPU memory. Larger models (more parameters, trickier features, better accuracy) consume more memory for activations and gradients, and GPUs aren't as clever as CPUs at managing memory — when they run out, they just crash. Jeremy demonstrates a quick hack for estimating memory usage: train on the smallest class only (337 images of bacterial panicle blight), since memory consumption depends on image and batch size, not dataset length. A helper function using `torch.cuda.list_gpu_processes()` reports usage, and calling `gc.collect()` plus `torch.cuda.empty_cache()` between runs keeps the GPU clean.

This leads directly into **gradient accumulation**, the lesson's first major technique. The problem: scaling to larger architectures (ConvNeXt large, ViT large, SwinV2 large, Swin large) causes out-of-memory errors at the default batch size of 64. The solution is disarmingly simple. Instead of processing 64 images and then updating weights, process 32 (or 16, or even 1) at a time, but don't zero out the gradients between mini-batches — PyTorch *adds* new gradients to old ones by default. Once enough images have been accumulated to match the target effective batch size, then update and zero. The resulting training loop is nearly mathematically identical to the original (exactly identical for architectures without batch normalization, like ConvNeXt), and you never need to change learning rates or other hyperparameters. In fastai, this is a one-liner: divide the batch size by `accum`, add the `GradientAccumulation(64)` callback, done. Jeremy pointedly notes that this means there's very little reason to buy expensive high-memory GPUs — a cheaper card with less memory and gradient accumulation gets you just as far.

With all large models fitting in 16 GB via `accum=2`, Jeremy builds a dictionary of architectures (ConvNeXt large, ViT large, SwinV2 large, Swin large) paired with their preprocessing configurations, loops through them all, and collects TTA predictions. He then **ensembles** the results by averaging the predicted probabilities across all models — a form of bagging. Different architectures with different preprocessing and different random training subsets (no `seed=` parameter) produce diverse predictions; averaging smooths out individual weaknesses. The ensemble of large models with TTA reached the top of the Kaggle leaderboard, and double-weighting the best-performing ViT models pushed the score even higher. Jeremy notes that Transformer-based vision models (ViT, Swin, SwinV2) require fixed square input sizes, unlike ConvNeXt.

Part 4 of "Road to the Top" introduces **multi-target models** — predicting both the disease and the rice variety from each image. This is done through fastai's DataBlock API: instead of `(ImageBlock, CategoryBlock)`, you pass `(ImageBlock, CategoryBlock, CategoryBlock)` with `n_inp=1`, so one image produces two categorical targets. The `get_y` parameter takes a list of two functions — `parent_label` for disease and a custom `get_variety` function that looks up the variety from a DataFrame indexed by filename.

This multi-target setup forces us to confront what `vision_learner` was doing for us silently: choosing the loss function and the number of output activations. With two targets, we must set `n_out=20` (10 diseases + 10 varieties) and provide our own loss function. This is the gateway to a thorough explanation of **cross-entropy loss**.

Jeremy walks through the mechanics in a spreadsheet. A model predicting 5 categories outputs 5 raw numbers. **Softmax** converts them to probabilities: exponentiate each output, then divide by the sum of all exponentials. The result: all values between 0 and 1, summing to 1, with larger outputs pushed closer to 1. Softmax enforces "pick one thing" — it cannot express "none of these," which is why a bear classifier will always pick a bear type even for a cat photo. If you want multiple independent predictions, you skip softmax and treat each output separately.

**Cross-entropy** then compares these probabilities to the one-hot encoded target. The formula $-\sum y_j \log(p_j)$ looks intimidating but collapses to a single lookup: find the probability the model assigned to the correct class and take its negative log. Low probability for the right answer → high loss. PyTorch's `nn.CrossEntropyLoss` (or equivalently `F.cross_entropy`) combines log-softmax and cross-entropy in one numerically stable step. The **binary cross-entropy** special case handles two-class problems with the familiar formula involving $y \log(p) + (1-y)\log(1-p)$.

For the multi-target model, `disease_loss` applies `F.cross_entropy` to the first 10 output columns against the disease target; `variety_loss` applies it to the last 10 columns against the variety target; `combined_loss` sums both. That's it — the gradients naturally guide the first 10 columns to specialize in disease and the last 10 in variety. Training shows both error rates dropping. Counterintuitively, predicting a second target can actually *improve* the primary prediction, because shared features (textures, variety-disease correlations) provide additional learning signal.

The lesson then shifts to **collaborative filtering**, presented as a chapter of the book that needed no revision. The MovieLens dataset provides 100,000 user-movie-rating triples. Cross-tabulated into a matrix, most cells are empty — the task is matrix completion.

The key idea is **latent factors**: we don't know what makes users like movies, but we assume there are hidden dimensions (maybe "how sci-fi is it," "how action-y," "how old"). For each user and each movie, we assign a vector of random numbers — 5 latent factors each — and predict a rating as the **dot product** of the user's vector with the movie's vector. We then optimize these vectors with gradient descent (demonstrated first in Excel's Solver, then in PyTorch) to minimize mean squared error against known ratings.

Jeremy then reveals that this array-lookup operation is what's called an **embedding** — a word that sounds far more sophisticated than it is. An embedding is simply an array lookup. It's computationally equivalent to multiplying by a one-hot encoded vector, but without ever constructing the one-hot vector. This connects directly back to the dummy variables from earlier lessons.

The `DotProduct` model in PyTorch inherits from `Module`, creates user and movie embedding matrices in `__init__`, and in `forward` looks up the relevant embeddings and computes their dot product. Training is fast — 100,000 rows in seconds on CPU.

Improvements follow. **Sigmoid range** squashes predictions between 0 and 5.5 so they can't exceed the rating scale. **Bias terms** — a single extra number per user and per movie — capture the tendency of some users to rate everything high and some movies to be universally loved. These are just additional 1-dimensional embeddings added to the dot product.

The model overfits, which introduces **weight decay** (L2 regularization). The idea: add $\text{wd} \times \sum w_i^2$ to the loss function, penalizing large weights. In practice this is implemented not by modifying the loss but by adding $\text{wd} \times w$ directly to the gradients (since the derivative of $w^2$ is $2w$, and the factor of 2 is absorbed into the `wd` constant). Weight decay asks the model to keep weights as small as possible while still making good predictions. In fastai, you pass `wd=0.1` to `fit_one_cycle`. The collaborative filtering model's loss improves steadily with weight decay applied. Jeremy notes that for vision models, fastai's defaults handle weight decay well, but for tabular and collaborative filtering problems you should experiment — start at 0.1 and divide by 10 a few times.

The lesson closes by defining **regularization** broadly: any technique that reduces overfitting by limiting model complexity without reducing the number of parameters. Weight decay is one form; dropout (covered later) is another.

---

- **Lesson Challenges**
  - Try building a multi-target model on the Titanic dataset (predict two things, e.g. sex and survival, or class and survival)
  - Experiment with the cross-entropy spreadsheet and the collaborative filtering spreadsheet
  - Add weight decay to your Titanic model to experiment with its effects
  - Run the "Road to the Top" Part 3 and Part 4 notebooks yourself

- **Potential Research Directions**
  - Teacher/student models and model distillation (using a smaller model to mimic a larger one — mentioned as a Part 2 topic)
  - Whether multi-target learning reliably improves single-target accuracy and under what conditions
  - Workarounds for Transformer-based vision models' fixed input size requirements
  - Alternative embedding size heuristics beyond fastai's rule of thumb
  - The interplay between batch normalization and gradient accumulation

- **Homework**
  - Replicate the multi-target model on the Titanic dataset
  - Run and experiment with the collaborative filtering notebook
  - Explore the softmax/cross-entropy and collaborative filtering Excel spreadsheets
  - Try different weight decay values on your collaborative filtering model
  - Read Chapter 8 of the fastbook

- **Things Jeremy Says You Should Do**
  - Use gradient accumulation instead of buying bigger GPUs — there's very little reason to spend on high-memory cards
  - When you get a CUDA out of memory error, restart your notebook first (these errors are tricky to recover from)
  - After each model training run, call `gc.collect()` and `torch.cuda.empty_cache()` to prevent GPU memory fragmentation
  - Try to submit something to Kaggle every day to build intuition
  - Use batch sizes that are multiples of 8 for performance; use the largest batch size your GPU can handle
  - If investing in a GPU, Nvidia consumer RTX cards are much cheaper but just as good as enterprise cards
  - Experiment with weight decay by starting at 0.1 and dividing by 10 to find the best value
  - Go back to earlier from-scratch models and try building multi-target versions

- **Resources**
  - **Notebooks:**
    - [Scaling Up: Road to the Top, Part 3](https://www.kaggle.com/code/jhoward/scaling-up-road-to-the-top-part-3)
    - [Multi-target: Road to the Top, Part 4](https://www.kaggle.com/code/jhoward/multi-target-road-to-the-top-part-4)
    - [Collaborative Filtering Deep Dive](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive/notebook)
  - **Spreadsheets:**
    - [Softmax and cross-entropy](https://github.com/fastai/course22/blob/master/xl/entropy_example.xlsx)
    - [Collaborative filtering and embeddings](https://github.com/fastai/course22/blob/master/xl/collab_filter.xlsx)
  - **Book:** [Chapter 8 — Collaborative Filtering](https://github.com/fastai/fastbook/blob/master/08_collab.ipynb) from *Deep Learning for Coders with fastai and PyTorch*
  - **Articles:**
    - [Things that confused me about cross-entropy](https://chris-said.io/2020/12/26/two-things-that-confused-me-about-cross-entropy/) — Chris Said
    - [Label Smoothing Explained using Microsoft Excel](https://amaarora.github.io/posts/2020-07-18-label-smoothing.html) — Aman Arora
  - **Forums:** [Lesson 7 official topic](https://forums.fast.ai/t/lesson-7-official-topic/100534)
  - **Course page:** [https://course.fast.ai/Lessons/lesson7.html](https://course.fast.ai/Lessons/lesson7.html)
