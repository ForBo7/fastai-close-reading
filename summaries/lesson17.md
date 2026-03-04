The lesson opens with a couple of housekeeping changes to the miniai library. The `Callback` class now has a `__getattr__` that forwards `model`, `opt`, `batch`, and `epoch` to `self.learn`, saving repetitive typing. The four training methods from `MomentumLearner` have been pulled into a `TrainLearner` subclass (with `zero_grad`), so `MomentumLearner` now just adds momentum on top. A new `HooksCallback` wraps the old `Hooks` class into a callback that auto-creates hooks in `before_fit` and removes them after — and a subclass `ActivationStats` appends means, standard deviations, and histograms, powering `color_dim`, `dead_chart`, and `plot_stats` methods. The message is clear: there is now zero excuse not to visualize what's happening inside your model, because it's one line of code.

The goal for the next lesson or two is announced: get Fashion-MNIST to 90% accuracy without architectural tricks like ResNets. A deliberately simple model is built — pure convolutions, channels doubling from 8 → 16 → 32 → 64, grid sizes halving from 14×14 down to 1×1, with a `Flatten` at the end. This model is so badly initialized that the learning rate finder is useless at normal settings; even after fiddling with the `gamma` multiplier, training at lr=0.2 is unstable — the colorful dimension plot shows the classic pattern of activations growing and crashing.

A practical aside on GPU memory in Jupyter: Python's underscore variables (`_`, `__`, `_16`) hold onto CUDA tensors, and exception tracebacks do the same. The `clean_ipython_hist` and `clean_mem` functions clear both, plus garbage-collect and empty the CUDA cache — no notebook restart needed. It's also noted that the Fashion-MNIST classifier is a stepping-stone: the tools built here will be needed later for a proper autoencoder.

The lesson then explains *why* we need mean 0 and standard deviation 1 throughout a network. A 50-layer toy experiment — just repeated matrix multiplies by `randn(100,100)` — produces NaN. Scaling down to 0.01 produces zeros. Both failures reflect floating-point's inability to represent very large or very small numbers, and the additional problem that numbers far from zero lose discriminating precision. The fix comes from Xavier Glorot and Yoshua Bengio's paper: initialize with bounds $\frac{1}{\sqrt{n}}$ where $n$ is the number of inputs. With 100 inputs, multiplying by 0.1 keeps activations reasonable through 50 layers.

A statistics refresher follows. Variance is the mean of squared deviations from the mean; standard deviation is its square root. The mean absolute deviation is similar but gives less weight to outliers. A handy shortcut: $\text{Var}(X) = E[X^2] - (E[X])^2$. Covariance measures how two variables move together, and the Pearson correlation coefficient is covariance scaled by both standard deviations.

Glorot initialization is derived from the observation that products of independent mean-zero random variables maintain mean 0 and std 1 when divided by $\sqrt{n}$. But this breaks with ReLU: after applying ReLU to Glorot-initialized layers, everything vanishes to zero, because ReLU kills all negatives and systematically shrinks the distribution.

Kaiming He's paper ("Delving Deep into Rectifiers") solves this: multiply by $\sqrt{2/n}$ instead of $1/\sqrt{n}$. The extra factor of $\sqrt{2}$ compensates for ReLU zeroing out roughly half the distribution. Using `nn.init.kaiming_normal_` via `model.apply`, the learning rate finder now works without hacks.

The rationale for doubling channels at each stride-2 convolution is explained: halving the grid in both dimensions reduces activations by 4×, but doubling channels adds 2× back, giving a 2× net reduction — a gentle compression that forces the model to learn efficient representations while keeping compute roughly constant across layers.

Even with Kaiming init, training reaches only 70% and the stats still aren't right at the start. The culprit: the input data itself has mean 0.28 and std 0.35, far from the 0 and 1 that initialization assumes. A `BatchTransformCB` callback normalizes every batch by subtracting the dataset mean and dividing by the dataset std. With that single fix, accuracy jumps to 85% and the colorful dimension plot finally looks beautiful — smooth, even activation distributions across layers.

But the stats still aren't *perfectly* at mean 0 and std 1, even from the very first layer. The reason: ReLU outputs can never have mean 0 because there are no negatives. This leads to the invention of **General ReLU** — a standard ReLU with two additions: a *leak* (small slope for negatives, i.e. leaky ReLU) and a *subtraction* (shifting the whole function down so the output can be negative). Plotted with leak=0.1 and sub=0.4, it's a normal line above zero pushed down by 0.4, with a gentle negative slope below. A new `conv` function and `get_model` accept configurable activation functions and filter counts; `init_weights` is updated to pass the leak factor to `kaiming_normal_`'s `a` parameter. With General ReLU, accuracy reaches 87%, means start near 0, standard deviations near 0.8, and the dead-unit percentages are tiny.

The lesson then introduces **LSUV** ("All You Need Is a Good Init" by Dmytro Mishkin). It's a completely general initialization method: pass a single batch through the model, then for each layer, iteratively subtract the mean from the bias and divide the weights by the std until both converge to 0 and 1. This is implemented with hooks and a while loop. After LSUV, every layer is correctly calibrated regardless of what activation function you use. Training gives 86% — comparable to the manual approach but with no need to derive the correct scaling factor.

The motivation for **normalization layers** is the original Batch Normalization paper by Ioffe and Szegedy, whose graph showed dramatic training speedups on ImageNet. The core idea: even if we initialize correctly, the distribution of each layer's inputs shifts during training (internal covariate shift), forcing lower learning rates and careful initialization. The fix is to normalize activations *during* training as part of the model.

**Layer Normalization** (Ba, Kiros, Hinton) is presented first for its simplicity. For each item in the mini-batch, compute the mean and variance over channel, height, and width, normalize, then scale by learnable `mult` (initialized to 1) and shift by learnable `add` (initialized to 0). The key insight: initially this just normalizes, but since `mult` and `add` are learnable, SGD can adjust them freely. What normalization really does is distill the overall scale and shift — previously distributed across every weight — into just two parameters per layer, making the optimization landscape much easier. With layer norm on every layer except the last, accuracy reaches 87.3%.

**Batch Normalization** is bigger. `mult` and `add` are now vectors (one per channel), and the mean/variance are computed over the batch *and* spatial dimensions — one per channel. Additionally, batch norm maintains exponentially weighted moving averages of means and variances (stored as buffers, not parameters) using `torch.lerp`. During inference, these saved statistics are used instead of per-batch statistics. With batch norm, the learning rate can be pushed to 0.4 — double what was possible before.

A comparison of normalization types uses the Group Norm paper's NCHW diagram: **Batch Norm** averages over N, H, W (one value per channel); **Layer Norm** over C, H, W (one per batch item); **Instance Norm** over H, W only (one per channel per batch item); **Group Norm** groups channels and averages over groups plus H, W. The lesson notes that normalization layers add complexity and recent trends favor reducing their use, making correct initialization increasingly important.

Dropping batch size from 1024 to 256 (with lr=0.2) and using PyTorch's `BatchNorm2d` reaches 87.8% in 3 epochs. Then, decreasing the learning rate and continuing for one more epoch fine-tunes to 89.9% — tantalizingly close to 90%.

The lesson moves to **Accelerated SGD**. First, a custom SGD class is built with `step` and `zero_grad`. **Weight decay** (L2 regularization) adds $\text{wd} \cdot \sum w^2$ to the loss; its gradient simplifies to multiplying weights by $(1 - \text{lr} \times \text{wd})$ at each step.

**Momentum** is explained with a visual demonstration: an exponentially weighted moving average (EWMA) of noisy data. At $\beta = 0.5$ the red line is slightly smoother; at $\beta = 0.9$ it's smooth but lags; at $\beta = 0.99$ it barely follows the signal. The sweet spot smooths out the bumps in the loss landscape without too much lag. Implementation: for each parameter, store a `grad_avg` attribute and lerp it with each new gradient; step by `grad_avg × lr` instead of `grad × lr`. With momentum, learning rate can be pushed to 1.5, reaching 87.6% with the smoothest colorful dimension plots yet. Yann LeCun's view is mentioned: ideal batch size is 1 (maximum updates), so use the smallest batch you can tolerate.

**RMSProp**, announced informally by Hinton in a Coursera lecture (~2012–2013, never formally published), lerps on squared gradients instead of gradients. Dividing by $\sqrt{\text{moving avg of grad}^2} + \epsilon$ scales down updates where gradients are noisy and scales up where they're consistent. A practical detail: initializing the squared average to the first batch's actual squared gradient avoids an enormous initial learning rate from dividing by near-zero.

**Adam** combines momentum and RMSProp: maintain both a gradient EWMA ($\beta_1$) and a squared-gradient EWMA ($\beta_2$), and apply an unbiasing correction $1/(1 - \beta^i)$ for the first few batches. With Adam at lr=0.01, accuracy is 86.5% — slightly below momentum alone, but tunable. The lesson closes with the promise that 90% will be achieved in the next lesson, and the reminder that everything has been built from scratch with full understanding.

---

- **Lesson Challenges**
  - Get Fashion-MNIST to ≥90% accuracy without architectural changes (ResNets, etc.)
  - Understand and fix initialization so that colorful dimension plots look smooth and flat

- **Potential Research Directions**
  - Deriving correct initialization scaling for novel activation functions beyond ReLU
  - Reducing or eliminating normalization layers while maintaining training stability
  - Exploring the interaction between batch size, learning rate, and momentum
  - The distinction between L2 regularization and weight decay in adaptive optimizers (they diverge for Adam — see [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101))
  - Internal covariate shift vs. the smoothing-the-loss-landscape explanation of batch norm (see [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604))

- **Homework**
  - Turn LSUV into a proper callback (being careful it doesn't re-initialize on every call to `fit`)
  - Experiment with different leak and subtraction values for General ReLU
  - Try different β₁ and β₂ values for Adam to beat momentum SGD's accuracy

- **Things Jeremy Says You Should Do**
  - Always use `ActivationStats` / colorful dimension plots — even for models you think are training well, you might be surprised
  - Call `clean_mem` when you hit CUDA OOM errors instead of restarting the notebook
  - Normalize your inputs — it's trivially easy and critically important
  - Care about initialization: most people don't, and most frameworks don't even let you visualize whether it's wrong

- **Resources**
  - [Lesson 17 course page](https://course.fast.ai/Lessons/lesson17.html)
  - [Lesson 17 video](https://youtu.be/vGsc_NbU7xc)
  - [Lesson 17 official topic (forums)](https://forums.fast.ai/t/lesson-17-official-topic/102602)
  - [Understanding the difficulty of training deep feedforward neural networks — Xavier Glorot, Yoshua Bengio](http://proceedings.mlr.press/v9/glorot10a)
  - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification — Kaiming He et al.](https://arxiv.org/abs/1502.01852)
  - [All You Need Is a Good Init (LSUV) — Dmytro Mishkin, Jiri Matas](https://arxiv.org/abs/1511.06422)
  - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift — Sergey Ioffe, Christian Szegedy](https://arxiv.org/abs/1502.03167)
  - [Layer Normalization — Ba, Kiros, Hinton](https://arxiv.org/abs/1607.06450)
  - [Group Normalization — Yuxin Wu, Kaiming He](https://arxiv.org/abs/1803.08494)
  - Notebooks: `13_initializing.ipynb`, `14_augmentation.ipynb` (miniai library), `07_accel_sgd.ipynb`
