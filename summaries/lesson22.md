The lesson opens with Jeremy, Johno, and Tanishq diving into notebook 22, a refinement of the DDPM–DDIM pipeline for Fashion-MNIST. The first significant change is the elimination of the discrete step count *T*. Instead of saying "this is step 500 out of 1,000," the time step is now a float between 0 and 1 representing the percentage through the forward diffusion process. ᾱ becomes a function you call — specifically the cosine schedule, `abar(t) = cos(t·π/2)²` — rather than something you look up in a list. Jeremy shows that you can also invert this to recover *t* from ᾱ. The `noisify` function changes accordingly: it samples a random float between 0 and 0.999 instead of a random integer, and calls `abar()` directly. Sampling likewise switches from `range` to `linspace`, stepping linearly from 0.999 down to 0. DDIM with 100 steps yields a FID of 3, noticeably better than the previous discrete-step version.

Jeremy then visualizes what a single denoising step actually looks like at varying noise levels — something rarely seen because few people work in interactive notebooks. At 25% noise the reconstruction is crisp; at 95% it's a blur. Even at 45%, where the image looks like pure noise to the eye, the model finds the underlying garment. The quality of these single-step predictions is genuinely impressive.

Next comes an excursion into original research: predicting the noise level itself. Jeremy builds a regression model that takes a noised image and predicts ᾱₜ (expressed as its logit, which is critical — without the logit transform the model fails entirely, since the difference between 0.999 and 0.99 is highly significant). The model achieves an MSE of 0.075, dramatically better than the random baseline of ~3.5, confirming that the network can figure out how much noise is present without being told. This matters because Jeremy wants to explore whether passing *t* to the UNet is actually necessary.

He tests this by training a standard noise-predicting UNet but always passing 0 for *t* — giving the model no information about the noise level. Training loss barely changes (0.034 vs 0.033). But sampling produces garbage (FID 22) because the model consistently removes slightly the wrong amount of noise, and the errors compound. The fix is clever: use the noise-level predictor (tmodel) during sampling to estimate the actual ᾱ at each step, clamp predictions to stay near the batch median, and use these estimates instead of the assumed values. This brings FID down to 3.88 — close to the with-*t* result of 3.2. Jeremy considers this encouraging for a couple days' work versus a decade of *t*-based approaches, and mentions that Robert (a lead author of the Stable Diffusion paper) has been fine-tuning a real Stable Diffusion model without *t* with promising results.

The discussion turns to a timely paper by Ting Chen at Google Brain, "On the Importance of Noise Scheduling for Diffusion Models," which Tanishq found the morning of the lesson. This paper demonstrates that the optimal noise schedule depends on image size — higher resolution images have more pixel redundancy, so the same noise level destroys less information. Their key strategies are adjusting the schedule function itself and scaling the input data by a constant factor *b*. Jeremy notes that their Figure 4c (b = 0.5) is exactly what their own accidental bug had been doing. The paper shows that with the right input scaling, they achieve state-of-the-art 1024×1024 class-conditioned ImageNet generation using a third of the training steps.

The heart of the lesson is notebook 23, implementing ideas from the Karras et al. paper "Elucidating the Design Space of Diffusion-Based Generative Models." Karras introduces a unified framework with greatly simplified notation — no more α̅, α, β, β̅, just σ (though confusingly, σ here plays the role α̅ used to). The core insight is that predicting pure noise is easy when the image is mostly noise, and predicting the clean image is easy when the image is mostly clean. Karras introduces c_skip to interpolate: the training target is a lerp between the clean image and the noise, weighted by σ, so the problem is equally difficult regardless of noise level. Jeremy visualizes this — at low σ the target looks like "predict the noise," at high σ it looks like "predict the image," and in between it's a blend.

To solve the input-scaling problem, Karras introduces c_in (inverse of total standard deviation, ensuring unit variance inputs) and c_out (ensuring unit variance outputs). Jeremy walks through the paper's Appendix B6 to show these are derived from basic variance math — the same principle used throughout the entire course: make inputs and outputs have unit variance. Johno emphasizes this is the recurring theme: nearly every improvement in deep learning comes down to getting mean-zero, unit-variance inputs and activations.

For sampling σ during training, Karras uses a log-normal distribution (exponentiated normal with mean −1.2, std 1.2), concentrating samples where the model can actually learn — the medium-noise range — rather than wasting effort on nearly-clean or nearly-pure-noise cases.

Sampling becomes dramatically simpler. The Karras sigma schedule uses a ρ=7 power law, spending more time on fine-grained denoising at the end and less on the noisy early steps. The Euler sampler is just three lines: denoise, compute the slope (x − denoised)/σ, step by (σ₂ − σ₁). This achieves a FID of 1.98. The Ancestral Euler variant, which adds controlled randomness, improves further to 1.53. Heun's method — which computes the slope at both the current and predicted next point, then averages — reaches FID 0.97 in 50 steps (100 model evaluations), and even at 20 steps (40 evaluations) beats Euler at 100. The LMS sampler takes a different approach, storing recent slopes and using polynomial interpolation, achieving competitive results in just 20 model evaluations. Jeremy notes that real training data gives a FID around 0.5, so 0.97 is remarkably close.

The lesson wraps with Jeremy showing the Karras paper's Table 1, which maps different papers' design choices into a common parameterization — demonstrating that VP, VE, iDDPM, and Karras's own approach are all instances of the same framework with different parameter values. Every notebook from DDPM onwards has gotten simpler to understand and produced better results. Tanishq draws connections: deterministic DDIM ≈ Euler sampler, stochastic DDPM ≈ Ancestral Euler. Next lesson will move to 3-channel 64×64 Tiny ImageNet and building a proper UNet from scratch.

---

**Lesson Challenges**

- Predict the noise level (α̅ₜ) of a noised image without being given *t*, and use the logit transform for the target
- Train a standard noise-predicting UNet without passing *t*, then fix sampling by using a noise-level predictor with median clamping
- Implement the Karras preconditioning (c_skip, c_out, c_in) and verify that noised inputs have unit variance

**Potential Research Directions**

- Training diffusion models without *t* conditioning — Jeremy's early experiments suggest this is viable; Robert is fine-tuning Stable Diffusion without *t*
- Adjusting mean (not just variance) of inputs/outputs to zero — Jeremy notes Karras only cared about unit variance, not zero mean
- Improving composition quality in early sampling steps where the Karras ρ schedule spends little time
- Exploring the relationship between image resolution and optimal noise schedule / input scaling (cf. [Ting Chen's paper](https://arxiv.org/abs/2301.10972))
- Investigating why σ_data = 0.33 (a bug) outperforms the correct value of 0.66

**Homework**

- Study and run notebooks 22_cosine and 23_karras
- Understand the derivation of c_in from Appendix B6 of the Karras paper
- Experiment with different σ schedules and input scaling factors on Fashion-MNIST
- Compare the various samplers (Euler, Ancestral Euler, Heun, LMS) at different step counts

**Things Jeremy Says You Should Do**

- Use interactive notebook environments to visualize intermediate denoising steps — it builds crucial intuition
- Always check a random/average baseline before evaluating model quality ("is this better than random?")
- Think carefully about target transformations (like logit) — without this the noise-prediction model appeared to fail, when really it was just a scaling issue
- Be lazy when doing research — don't build the fancy version until you know the simple version works
- Read the scary-looking math appendices in papers — they often turn out to be straightforward
- Do research on small datasets first, then scale up at the end

**Resources**

- [Notebook 22: Cosine schedule](https://github.com/fastai/course22p2/blob/master/nbs/22_cosine.ipynb)
- [Notebook 22: Noise-spread (predicting noise level)](https://github.com/fastai/course22p2/blob/master/nbs/22_noise-spread.ipynb)
- [Notebook 22: No-t experiment](https://github.com/fastai/course22p2/blob/master/nbs/22_no-t.ipynb)
- [Notebook 23: Karras](https://github.com/fastai/course22p2/blob/master/nbs/23_karras.ipynb)
- [Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (2022)](https://arxiv.org/abs/2206.00364)
- [Ting Chen, "On the Importance of Noise Scheduling for Diffusion Models" (2023)](https://arxiv.org/abs/2301.10972)
- [Kat Crowson's k-diffusion repo](https://github.com/crowsonkb/k-diffusion)
- [Lesson 22 official topic (forums)](https://forums.fast.ai/t/lesson-22-official-topic/103586)
- [Lesson 22 video](https://youtu.be/6Bta1tXRUfM)
