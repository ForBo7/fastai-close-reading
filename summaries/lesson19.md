Lesson 19 opens on New Year's Eve 2022 with Jeremy sharing quick updates from the Fashion-MNIST challenge. Christopher Thomas on the forum broke new ground with Dropout, and Piotr spotted a bug in Jeremy's ResBlock code where the BatchNorm parameter wasn't being passed along. With both fixes applied, results improved to 93.2% in just 5 epochs.

Jeremy then teaches Dropout from scratch. The idea is simple: with some probability *p* (here 0.1), randomly zero out activations during training. He builds a `Dropout` class that creates a binomial distribution object with probability 1−*p*, samples a mask of ones and zeros, and multiplies it by the activations. A subtle but important detail: the surviving activations are scaled up by 1/(1−*p*) to compensate for the ~10% reduction in magnitude. During evaluation, dropout is turned off entirely. Johno catches a bug live — `probs = 1-p` should be `1-self.p`. Jeremy places a regular Dropout layer before the final linear layer and a Dropout2d layer at the start. Dropout2d differs in that it drops entire channel grids together rather than individual positions independently. Christopher Thomas later found that removing Dropout2d and keeping only regular Dropout yielded even better results over 50 epochs — the first to break 95%.

An interesting tangent arises when Johno asks about using dropout at test time. Jeremy reveals he wrote a callback that keeps Dropout in training mode during evaluation, enabling a kind of confidence measure — running the model multiple times with dropout on gives varied predictions when the model is uncertain. He notes this has been used in medical models and deserves more study.

The lesson then pivots dramatically as Tanishq takes over to build DDPM (Denoising Diffusion Probabilistic Models) from scratch — everything except the U-Net architecture itself, which will come next lesson. Tanishq begins by framing the goal of generative modeling: learning p(x), the probability distribution over data, so that new samples can be drawn from it. He uses height as a one-dimensional analogy — you wouldn't sample heights uniformly between 3' and 7', but according to a bell curve centered around 5'10".

Tanishq then walks through the DDPM paper (Ho et al., 2020, UC Berkeley), explaining the two core processes. The **forward process** progressively adds noise to an image across T timesteps, governed by a noise schedule β_t that increases with t. The transition at each step is a Gaussian: q(x_t | x_{t−1}) = N(x_t; √(1−β_t) · x_{t−1}, β_t·I). As t grows, the mean shrinks toward zero (losing the image) while the variance grows (adding noise), until x_T is pure Gaussian noise. Because each step depends only on the previous one, a closed-form expression exists for jumping directly from x_0 to any x_t: q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1−ᾱ_t)·I), where ᾱ_t is the cumulative product of (1−β_t).

The **reverse process** is where the neural network lives. The paper's key simplification is fixing the variance to constants and having the network predict only the mean — which is further reparametrized so the network actually predicts the noise ε that was added. The simplified training objective becomes a plain MSE loss: ||ε − ε_θ(x_t, t)||². Tanishq explains the intuition: noise prediction tells you the direction back toward the data distribution. Adding noise takes you away from it; predicting and subtracting noise brings you back.

Moving to code, Tanishq implements everything as a miniai callback called `DDPMCB` that inherits from `TrainCB`. The `__init__` sets up β as a linspace from β_min to β_max across T timesteps, then computes α = 1−β, ᾱ = cumulative product of α, and σ = √β. Jeremy shows plots of these variables to build intuition. The `before_batch` method does the noising: it takes clean images x_0, samples random timesteps, generates noise ε from N(0,I), and creates noisy images via x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε. The batch is then set to ((x_t, t), ε) — input tuple and target noise. The `predict` method unpacks the tuple with `*` and calls `.sample` on the Hugging Face model output. Tanishq walks through how this matches Algorithm 1 from the paper exactly.

For the model itself, they temporarily import a U-Net from Hugging Face's Diffusers library — cheating that will be rectified next lesson. The rest of the training loop is standard miniai with MSE loss. Jeremy emphasizes how elegant it is that the same framework trains both classifiers and diffusion models, differing only by one callback.

The `sample` method implements Algorithm 2: starting from pure noise x_T, it iterates backward through all T timesteps. At each step it predicts the noise, computes an estimate of x_0, then takes a weighted average of that estimate and the current x_t, plus a small amount of fresh noise. As timesteps decrease, the weight on the x_0 estimate increases. After just 5 epochs (~4 minutes of training) on Fashion-MNIST, the generated 32×32 images show recognizable shirts, shoes, and pants with visible texture and detail. Jeremy notes this far surpasses what they achieved years ago with Wasserstein GANs after hours of training.

Visualizing the sampling progression reveals a limitation: between timesteps 0 and 800, the images are essentially just noise with very little visible change — all the interesting denoising happens in the last 200 steps. Tanishq explains this is a known weakness of the linear noise schedule used in the original DDPM paper, particularly on small images, and points to the Improved DDPM paper which proposes better schedules.

Jeremy then shows his own exploration notebook (notebook 17) where he refactored Tanishq's code. He extracted the `before_batch` logic into a standalone `noisify` function, making it easier to experiment with outside a class. He took an alternative architectural approach: instead of inheriting from `TrainCB` and overriding `predict`, he inherited from `UNet2DModel` directly and overrode `forward` to handle the tuple unpacking. He visualizes noisified images at various timesteps to verify correctness — a practice he strongly advocates, since he gets things wrong the first several attempts.

For speed improvements, Jeremy applies custom initialization: zeroing every second conv layer in ResBlocks (so skip connections initially act as identity), orthogonal weights for downsamplers, and zeroing the final output layer. Crucially, he changes Adam's epsilon from the default 1e-8 to 1e-5, which prevents the effective learning rate from exploding when squared gradients are very small. He also halves all channel sizes while adjusting group norm groups accordingly, achieving a loss of 0.016 in 5 epochs.

The lesson closes with Jeremy teasing mixed precision training — using 16-bit floats where possible for massive speedups on modern GPUs — but runs out of time. He frames the path ahead: just as the previous lessons journeyed from bad to excellent Fashion-MNIST classification, now they'll journey from decent to excellent Fashion-MNIST generation, eventually reaching Stable Diffusion and beyond.

---

**Lesson Challenges**

- Implement Dropout2d from scratch and devise a way to test that it's actually working correctly (not just appearing to work)
- Try implementing alternative noise schedules (e.g. from the Improved DDPM paper) using Tanishq's notebook as a starting point
- Experiment with different β ranges and numbers of timesteps
- Try halving channel sizes and adjusting group norm groups

**Potential Research Directions**

- Test-time dropout as a confidence/uncertainty measure — calibrating dropout variance against known confident/uncertain predictions; applications in medical AI
- Alternative noise schedules (cosine schedule from Improved DDPM) for better utilization of all timesteps, especially on small images
- The emerging class of non-diffusion iterative refinement models (referenced by Jeremy as potentially superior to DDPM-style diffusion)
- Better initialization strategies for diffusion model U-Nets (the Diffusers library uses PyTorch defaults)
- The effect of Adam's epsilon on training stability at high learning rates

**Homework**

- Implement Dropout2d from scratch with a testing strategy
- Explore the Fashion-MNIST generation challenge: try to improve on the baseline DDPM results
- Read the DDPM paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Read the Improved DDPM paper for alternative noise schedules

**Things Jeremy Says You Should Do**

- Always draw pictures of everything along the way — the first six or seven times you do anything, you do it wrong, so visualize to catch mistakes
- Always use a small and simple dataset for R&D and experiments until it's no longer challenging
- When stuff is inside a class and hard to explore, copy the logic into a standalone function to understand what the actual parameters are
- Don't assume library defaults (like PyTorch init or Adam epsilon) are optimal for your specific model
- Check the forums for the latest results on challenges

**Resources**

- [Lesson 19 video](https://youtu.be/ItyO8s48zdc)
- [Lesson 19 official topic (forums)](https://forums.fast.ai/t/lesson-19-official-topic/103201)
- [Notebook: 15_DDPM.ipynb](https://github.com/fastai/course22p2/blob/master/nbs/15_DDPM.ipynb) — Tanishq's DDPM from scratch notebook
- [Notebook: 17_DDPM_Jeremy.ipynb](https://github.com/fastai/course22p2/blob/master/nbs/17_DDPM_Jeremy.ipynb) — Jeremy's exploration/refactoring notebook
- [DDPM paper — Ho et al. 2020](https://arxiv.org/abs/2006.11239) — "Denoising Diffusion Probabilistic Models"
- [Improved DDPM paper](https://arxiv.org/abs/2102.09672) — "Improved Denoising Diffusion Probabilistic Models" (alternative noise schedules)
- [Hugging Face Diffusers library](https://github.com/huggingface/diffusers) — source of the temporary U-Net used in this lesson
- [miniai framework](https://github.com/fastai/course22p2) — the from-scratch training framework built throughout Part 2
