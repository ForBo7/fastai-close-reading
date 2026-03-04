The lesson opens not in a Jupyter notebook but in Microsoft Excel, where a spreadsheet called `graddesc` from the `course22p2` repo lays out accelerated SGD approaches in full view. The task is deliberately simple: learn the slope and intercept of a linear regression where the true values are $a = 2$ and $b = 30$, starting from random guesses of 1 and 1. For the first data point ($x = 14$, $y = 58$), the prediction $\hat{y} = 1 \times 14 + 1 = 15$ is way off. The squared error and its derivatives are computed two ways: by finite differencing (nudging the parameter by 0.01 and measuring the change) and analytically ($2(\hat{y} - y)$ for the intercept, $2(\hat{y} - y) \cdot x$ for the slope). The two agree closely, and the point is made that finite differencing is slow but invaluable as a sanity check — always test analytic gradients against it. The parameters are updated with a learning rate of 0.0001 using online SGD (batch size one), and after one epoch the estimates improve from (1, 1) to roughly (1.06, 2.57). A VBA macro automates the tedious copy-paste across epochs, and a chart shows RMSE declining — but painfully slowly, with near-constant increments toward the true intercept of 30.

This motivates **momentum**. On a new sheet with $\beta = 0.9$, the update becomes a lerp between the previous momentum and the current gradient: $v_t = \beta v_{t-1} + (1-\beta) g_t$. When gradients consistently point the same direction, momentum builds up and the parameters jump further. The VBA macro now also copies the momentum state back to the top each epoch. Training is noticeably faster.

**RMSProp** follows on its own sheet. Instead of tracking an exponentially weighted moving average of the gradients, it tracks the moving average of gradients squared, then divides the gradient by the square root of that average. The effect: when gradients vary little, the step size grows; when they're noisy, it shrinks. This normalizes step sizes across parameters.

**Adam** combines both: a momentumized gradient and the RMSProp normalization. After just 10 epochs the estimates jump close to the correct values, but then oscillate — a clear sign the learning rate needs to decrease. This leads to **Adam with automatic annealing**, a novel experiment right in the spreadsheet. The average squared gradients per epoch are tracked, and when the ratio of current to historical minimum halves, the learning rate drops by a factor of 4. The theory is that flatter loss landscape regions signal readiness for a smaller step. Running it, the learning rate automatically decreases several times and the parameters converge almost exactly. The suggestion is made that students should try implementing an automatic annealer in miniai.

The lesson then shifts to PyTorch. Using `dir(lr_scheduler)` and a quick list comprehension filtering for class names, all available schedulers are enumerated — something not easily found in the official docs. The anatomy of PyTorch optimizers is explained: parameter groups (collections of parameters that can have different hyperparameters) and the state dictionary (keyed by parameter tensors, storing optimizer state like moving averages). This contrasts with the miniai approach of storing state as attributes directly on parameters. Parameter groups matter for transfer learning, where earlier and later layers often want different learning rates.

A **cosine annealing** scheduler is created with `T_max` set to 3 × the number of mini-batches. A small helper function steps through the scheduler and records learning rates for plotting. The learning rate starts high, stays high, then drops — and if iterated past `T_max`, it rises again because it's a cosine. Training with this batch-level cosine schedule and Adam yields close to 90% accuracy in just 3 epochs. An epoch-level scheduler is also demonstrated, producing just 3 discrete steps.

The **1-Cycle policy** from Leslie Smith replaces cosine annealing. Training for 5 epochs reaches **90.6%** — a new record. The recorder plots both learning rate and momentum (Adam's beta): the LR starts low (warmup), ramps up high, then decreases; momentum mirrors it. The rationale is explained carefully: low LR with high momentum at the start prevents bad weight jumps from imperfect initialization; high LR with low momentum in the middle explores efficiently; low LR with high momentum at the end fine-tunes. A brief aside mentions the 2019 Fixup paper and T-Fixup, which showed that with proper initialization, warmup and batch norm become unnecessary — but most practitioners don't initialize well enough.

A code quality improvement is introduced: callbacks now receive `learn` as a parameter instead of storing `self.learn`, avoiding ugly `self.learn.model` chains and reference cycles. A tip for tracking repo changes: append `/compare` to a GitHub URL and use `master@{7.days.ago}...master` to see a week's diffs. fastcore's `@patch` decorator is used to add `lr_find` directly to the Learner class. The `fit` method gains a `callbacks` parameter for temporary callbacks that are removed after fitting.

Attention turns to **ResNets**. The existing model — 4 conv layers going 1→8→16→32→64 channels, all stride-2 — sits at 90.6%. A simple change, making the first conv stride-1 (adding one more layer and reaching 128 channels), pushes accuracy to **91.7%**. But deeper networks run into trouble: the famous He et al. result shows a 56-layer plain network performing worse than a 20-layer one on the *training set*. The insight is that since a 56-layer net is a strict superset of a 20-layer net (the extra layers could just be identity), the problem is purely one of optimization, not capacity.

The solution is **skip connections**: instead of $\text{out} = f(\text{in})$, compute $\text{out} = f(\text{in}) + \text{in}$. If the conv weights start near zero, the block initially acts as identity, so a deep network begins life behaving like a shallow one. The convolutions learn the *residual* — the difference between output and input. When input and output shapes differ (different channel counts or stride), a 1×1 identity convolution and/or average pooling are placed on the skip path.

The implementation initializes the second conv's **batch norm weights to zero**, making the entire conv block output zeros at the start without touching the conv weights themselves. The activation function is applied after the residual addition, not inside individual convs. A `get_model` function is identical to the plain CNN version except every `conv` is replaced with `ResBlock`, and the final 1×1 conv becomes a flatten + linear (since they're mathematically identical at that size). A `_print_shape` hook and a `summary` patch for Learner display the data flow and parameter counts per layer.

Training the ResNet with `lr_find` at 0.02 gives **92.2%**. Comparing against Ross Wightman's **timm** library, the best timm ResNet (ResNet18d) only managed 92.0% — the hand-crafted architecture beats it. Going wider with a 5×5 first kernel and doubling up to 512 channels reaches **92.7%**. The `summary` patch is extended to show **MFLOPs**, revealing that the first layer (on the full 28×28 grid) dominates compute despite having few parameters — a reminder that parameter count and speed are poorly correlated. Removing the 512-channel layer drops parameters from 4.9M to 1.2M with no accuracy loss; replacing the first ResBlock with a single conv drops MFLOPs from 18.3 to 13.3 with no accuracy loss either.

Training for 20 epochs without regularization overfits (training accuracy 0.999, validation drops). The claim is made that **weight decay doesn't regularize with BatchNorm**: the batch norm coefficients can simply scale up to compensate for shrunk weights, since BN has very few parameters and is barely affected by weight decay. Any effect is a weird second-order learning-rate interaction, not true regularization.

**Data augmentation** is the proper remedy. A `BatchTransform` callback with `RandomCrop(28, padding=1)` and `RandomHorizontalFlip` runs on the GPU. Because it's a batch transform, every image in the batch gets the same augmentation — less variety per batch, but no CPU bottleneck. With just 1 pixel of padding and horizontal flip, 20 epochs of OneCycle training reaches **93.8%**. Jeremy challenged the world on Twitter to beat this in 20 epochs; nobody got close.

**Test Time Augmentation** applies a deterministic horizontal flip at validation time, averages the two prediction sets, and pushes accuracy to **94.2%** — like a mini ensemble at zero extra training cost.

**Random erasing** is implemented from scratch. A small patch of each image is replaced with Gaussian noise matching the dataset's mean and standard deviation. The first attempt produces visible artifacts: the noise has the right statistics but the wrong range, with values outside the original min/max. The fix is to **clamp** the random values. An extension randomly places up to 4 erased blocks per image. With all three augmentations and 50 epochs, accuracy reaches **94.6%**.

A creative variant called **random copying** replaces erased regions with pixels copied from elsewhere in the same image, guaranteeing correct pixel distributions without clamping. This is noted as related to CutMix, which copies from different images.

**Ensembling** two 25-epoch models (94.1% and 94.0%) by averaging their predictions gives 94.3% — better than either alone but not beating the 50-epoch record. On Papers with Code, the 94.6% result is competitive with the best Fashion-MNIST models, all of which use 250+ epochs.

---

- **Lesson Challenges**
  - [Fashion-MNIST challenge](https://forums.fast.ai/t/a-challenge-for-you-all/102656): beat 93.8% (20 epochs) or 94.6% (50 epochs) on Fashion-MNIST
  - Try implementing an automatic learning rate annealer in miniai based on the squared-gradient-ratio idea from the spreadsheet

- **Potential Research Directions**
  - Automatic learning rate annealing based on gradient statistics (the spreadsheet experiment)
  - Proper initialization removing the need for batch norm and warmup ([Fixup Initialization](https://arxiv.org/abs/1901.09321), [T-Fixup](https://arxiv.org/abs/2002.02862))
  - Random copying as an alternative to random erasing and its relationship to CutMix
  - The relationship (or lack thereof) between parameter count, MFLOPs, and actual training speed
  - Whether weight decay provides any meaningful regularization in the presence of BatchNorm

- **Homework**
  - Create your own schedulers that work with PyTorch's optimizers: implement cosine annealing from scratch, then 1-Cycle, and make sure they work with the batch scheduler callback. Study the PyTorch optimizer API carefully.
  - Try to beat Jeremy on Fashion-MNIST at 5, 20, or 50 epochs, ideally using miniai with things you've added yourself. If you grab another library and it helps, try to re-implement it.

- **Things Jeremy Says You Should Do**
  - Always test analytic gradients against finite differencing as a sanity check
  - Investigate objects in a REPL environment — look at what's inside, run things independently, plot anything you can plot; this is how to learn APIs
  - When building models, think about architecture thoughtfully rather than grabbing the biggest off-the-shelf model
  - Be skeptical of "fewer parameters" claims — parameter count doesn't determine speed
  - Don't start by writing functions; start by writing single lines of code you can run and check independently, looking at pictures to verify correctness
  - Expect to get frustrated with homework — that's the journey

- **Resources**
  - [Lesson 18 discussion thread](https://forums.fast.ai/t/lesson-18-official-topic/102750)
  - [Fashion-MNIST challenge thread](https://forums.fast.ai/t/a-challenge-for-you-all/102656)
  - [Excel optimisers spreadsheet](https://github.com/fastai/course22p2/blob/master/xl/graddesc.xlsm)
  - Notebooks: `17_accel_sgd.ipynb`, `17_resnet_augment.ipynb` (in [course22p2 repo](https://github.com/fastai/course22p2))
  - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) — Leslie Smith
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Kaiming He et al.
  - [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321)
  - [T-Fixup (Improving Transformer Optimization Through Better Initialization)](https://arxiv.org/abs/2002.02862)
  - [Fashion-MNIST benchmarks on Papers with Code](https://paperswithcode.com/sota/image-classification-on-fashion-mnist)
  - [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models) — Ross Wightman
  - [fastcore](https://github.com/fastai/fastcore) (`@patch` decorator)
