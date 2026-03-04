Jonathan walks you through the Stable Diffusion Deep Dive notebook, peeling apart the high-level pipeline into its individual components so you can see — and modify — exactly what happens when an image is generated. The code is adapted directly from the `__call__` method of the default Hugging Face Stable Diffusion pipeline, now laid bare in a notebook you can run alongside.

The first component is the **variational autoencoder (VAE)**. Stable Diffusion is a *latent* diffusion model: it never touches pixels directly. Instead, a pre-trained VAE compresses a 512×512 image down to a 4×64×64 latent representation — a factor-of-8 reduction per spatial dimension, roughly 64× less data — yet the decoded reconstruction is nearly indistinguishable from the original. This is what makes high-resolution generation tractable: the diffusion model only ever works with these compact 64×64 latents. The input dimensions need not be 512; any multiple of 8 will do, and the same factor-of-8 reduction applies.

Next comes **noise and the scheduler**. During training, noise is added to latents at varying amounts controlled by a schedule (parameterized by `beta_start`, `beta_end`, and `beta_schedule`). At sampling time we use far fewer steps — say 15 instead of 1000 — mapped to a subset of training timesteps. The `sigmas` attribute shows the actual noise level at each sampling step, starting high and tapering to zero. The `add_noise` function is deceptively simple: `noisy_samples = original_samples + noise × σ`. The **image-to-image pipeline** exploits this directly: encode an input image, noise it to some intermediate timestep, then denoise from there with a new prompt. The "strength" parameter simply controls which step you start from — how much of the schedule you skip — giving you control over how much of the original composition survives.

The **text encoding process** is dissected layer by layer. A prompt is tokenized into exactly 77 tokens (padded or truncated), then each token is looked up in a learned embedding table (~50,000 entries, each 768-dimensional). These token embeddings are position-independent, so a separate learned positional embedding (one per position out of 77) is added element-wise. The combined embeddings are then fed through a stack of Transformer encoder blocks — attention, feed-forward, normalization, skip connections — producing the final encoder hidden states (output embeddings) that condition the UNet. Breaking this into steps enables powerful manipulation: you can swap a token embedding to change the subject, average two token embeddings (e.g. "puppy" and "skunk") to create chimeras, or blend final output embeddings from two entirely different prompts.

**Textual inversion** is the main practical application of token-level manipulation. A small learned embedding file (`embeds.bin`) — just a single 768-dimensional vector — captures a new concept or style. Jonathan demonstrates "birb style," trained on watercolor bird paintings, producing a charming mouse-in-an-apron in that style simply by inserting the learned embedding at the right position in the token sequence. Over 1,400 community-contributed concepts exist in the Stable Diffusion Concepts Library, and training your own is straightforward.

The **UNet and classifier-free guidance (CFG)** section reveals what the diffusion model actually predicts. The UNet takes noisy latents, the current timestep, and text embeddings, and outputs a noise prediction of the same shape as the latents. The predicted denoised image is computed as x̂₀ = xₜ − σ·ε_θ. Progress visualizations show the predicted output starting blurry and gradually sharpening, while the latents themselves change only incrementally per step. CFG works by running *two* copies through the model — one conditioned on the prompt, one unconditional (empty prompt) — and combining predictions: ε_final = ε_uncond + s·(ε_cond − ε_uncond). The guidance scale *s* controls how aggressively the model follows the prompt; higher values produce more prompt-faithful but potentially less natural results.

Jonathan then steps away from the notebook for a **paper-and-pen explanation of sampling**. He introduces the space of all possible images, manifold theory (real images occupy a lower-dimensional surface within that vast space), and the score function (predicting how to get back to the manifold from a noisy point). Single-step denoising fails because from a random starting point the model can only predict a blurry average. The solution is iterative ODE solving: remove a little noise, get a better prediction, repeat. **First-order solvers** (Euler's method) use simple linear steps. **Second-order solvers** account for curvature by estimating how the prediction changes, enabling larger steps at the cost of multiple model evaluations per step. **Linear multi-step (LMS) sampling** is a practical hybrid — it keeps a buffer of past predictions to approximate curvature without extra model calls. Finally, Jonathan presents an alternative framing: **sampling as optimization**, where the predicted noise amount serves as a loss and standard optimizer tricks (adaptive learning rates, momentum) drive the latents toward a plausible image. Not yet mainstream, but a thought-provoking perspective.

The lesson concludes with **loss-based guidance**. Beyond text and image-to-image control, you can define an arbitrary loss function on the predicted denoised output (decoded to image space) and use its gradient to nudge the latents. A simple "make it blue" loss demonstrates the principle: every few iterations, decode the predicted output, compute the loss, trace gradients back to the latents, and subtract ∇(loss)·σ². The result is a distinctly blue campfire painting. Caveats: decoding and gradient tracing are expensive (hence applying every 5th step), and the shortcut of not tracing through the UNet itself trades accuracy for memory. For more precise gradients, full backprop through decoder and UNet is possible with gradient checkpointing. The technique generalizes to color palettes, CLIP-based text guidance, classifier-based steering, and more.

---

- **Lesson Challenges**
  - Experiment with different noise timesteps to see how the noised image changes at different stages
  - Try different schedulers from the diffusers library
  - Explore mixing embeddings from different prompts at different blend factors
  - Tweak the `blue_loss_scale` to see how guidance strength affects the output
  - Try applying the loss every iteration instead of every 5th, with a lower scale

- **Potential Research Directions**
  - Sampling as optimization: applying modern optimizer techniques (Adam, momentum, weight decay) to the denoising loop
  - Dynamic step-count estimation during sampling
  - More efficient gradient-based guidance via gradient checkpointing through the full UNet
  - Combining textual inversion with loss-based guidance for fine-grained style + content control
  - Exploring what information each of the four latent channels encodes

- **Homework**
  - Run through the notebook yourself, modifying prompts, guidance scales, and sampling steps
  - Train your own textual inversion concept using the linked community notebooks
  - Implement a custom loss function (e.g. CLIP-based or style-transfer-based) for guided generation

- **Things Jeremy Says You Should Do**
  - *(This lesson is presented by Jonathan; no explicit Jeremy recommendations are given in this segment.)*

- **Resources**
  - [Stable Diffusion Deep Dive Notebook](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb) — the notebook walked through in this lesson
  - [Hugging Face Diffusers Library](https://github.com/huggingface/diffusers) — pipelines, schedulers, and models used throughout
  - [Stable Diffusion Concepts Library](https://huggingface.co/sd-concepts-library) — 1,400+ community-contributed textual inversion embeddings
  - [Textual Inversion training notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb) — for training your own learned embeddings
  - [AI Coffee Break: GLIDE video](https://www.youtube.com/watch?v=lvv4N2nf-HU) — recommended explanation of classifier-free guidance
  - [Score-based generative modeling (Yang Song's blog)](https://yang-song.net/blog/2021/score/) — background on the SDE/ODE framing of diffusion
