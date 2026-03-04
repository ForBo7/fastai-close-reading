The lesson opens with Johno presenting exciting progress: scaling up from the tiny Fashion-MNIST dataset to CIFAR-10 using the exact same miniai code. CIFAR-10 is a popular benchmark for both classification and generative modeling — the smallest dataset you'll see reported in diffusion papers. The images are now three-channel RGB instead of single-channel grayscale, but thanks to PyTorch's broadcasting, the noisify function and everything else works without modification. Jeremy reminisces about winning a global competition training CIFAR-10 classification for 26 cents of AWS compute back in 2018, and admits he actually hates the dataset — but it's useful for quick testing before moving on to something better.

Johno then motivates experiment tracking. With the default diffusers UNet now at 274 million parameters (versus the 15 million they'd been using), training runs take long enough that you can't just sit and watch. He introduces **Weights and Biases (W&B)**, a free experiment tracking service. The W&B callback he wrote inherits from `MetricsCB`, overriding `_log` to send metrics to W&B's servers. It logs training loss every batch, validation metrics every epoch, and generates sample images as matplotlib figures. Jeremy is impressed by how little code this requires — the callback system they built in miniai makes it trivial. Jeremy also offers a counterpoint: he personally doesn't use experiment tracking because he prefers to stay disciplined and directed rather than running hundreds of undirected hyperparameter sweeps. The consensus is that W&B is excellent (and as of early 2023, the best tool with the best fastai integration), but careful hypothesis-driven experimentation matters more than tooling.

The lesson then pivots to a major topic: **how do you measure the quality of generated images?** Jeremy explains the challenge — when generated samples start looking decent, it's very hard to tell visually whether you're improving. He introduces two metrics: **FID (Fréchet Inception Distance)** and **KID (Kernel Inception Distance)**. The idea behind both is to run images through a pre-trained classifier, grab the activations from a late pooling layer (yielding a 512-dimensional feature vector per image), and compare the distribution of these features between real and generated images. Jeremy builds this up step by step in the `18_fid.ipynb` notebook. He loads the DDPM2 model trained last lesson and a pre-trained Fashion-MNIST classifier, then demonstrates two ways to extract intermediate features: using hooks (via `HooksCallback`) or simply deleting the final layers of a Sequential model and calling `capture_preds`.

With features in hand, Jeremy explains the **covariance matrix**: a 512×512 matrix where each entry captures how correlated two feature channels are across a batch. The diagonal is the variance of each channel; the off-diagonals capture relationships (e.g., "pointy heels" correlating with "no flowing fabric"). The FID compares the means and covariance matrices of real versus generated features. Computing the FID requires a **matrix square root**, which Jeremy implements from scratch using the **Newton-Schulz iterative method** — the matrix analog of Newton's method for scalar square roots. Once implemented, he also demonstrates using `scipy.linalg.sqrtm` as the practical alternative.

Jeremy explains FID's caveats: it is biased by sample count (fewer samples → inflated FID), and the standard practice of using the **Inception** network (an obsolete ImageNet classifier) means all images are resized to 299×299, which is a terrible fit for 28×28 or 32×32 images. He wraps an `InceptionV3` model to handle single-channel images by replicating channels, and shows that his custom Fashion-MNIST classifier actually gives a much better FID ratio than Inception. He then introduces **KID**, which uses Maximum Mean Discrepancy with a polynomial kernel and doesn't have the bias problem — but in practice has very high variance, making it unreliable. Jeremy wraps everything into an `ImageEval` class for convenient use. He also plots FID across sampling steps — a novel visualization showing that quality plateaus around step 900 — and takes the FID of real data as a baseline, which is something he hasn't seen others do but considers very informative.

The lesson takes a dramatic turn with **DDPM_v3**. Jeremy tried to fix a "bug" — everyone feeds UNets images in the range \([-1, 1]\), but they'd been using \([0, 1]\). When he switched to \([-1, 1]\), results got *worse*. Days of painful debugging followed. He discovered they hadn't been shuffling training data (fixed, but unrelated), and eventually asked a deeper question: is there actually any mathematical reason for the \([-1, 1]\) range? Nobody could find one. The insight: the issue wasn't centering, it was that \([-1, 1]\) has range 2 while \([0, 1]\) has range 1. A smaller range means the noise schedule is more effective. So he tried \([-0.5, 0.5]\) — range 1, centered at zero — and it was dramatically better. This became **DDPM_v3**. Jeremy draws the lesson that bugs can reveal accidental discoveries, and researchers should always question "standard" choices that lack theoretical justification. He notes that Robin Rombach (Stable Diffusion co-author) discovered similar things — using a lower β_max of 0.012, and a mysterious 0.18 scaling factor on latents, both of which reduce effective range.

Jeremy then experiments with **noise schedules**. He compares the standard linear schedule (β_max = 0.02) with the **cosine schedule**, plotting both ᾱ curves and their slopes. The linear schedule wastes many steps with ᾱ near zero, while the cosine schedule distributes noise smoothly. But simply reducing β_max to 0.01 in the linear schedule produces nearly the same curve as cosine — so he sticks with the linear schedule with this smaller β_max. With DDPM_v3 trained for 25 epochs with doubled channels, 512 generated samples achieve **FID 8.1** versus 6.6 for real data — nearly indistinguishable. Jeremy also experiments with skip-sampling (calling the model every 3rd step plus the last 50), achieving roughly 3× speedup with minimal FID degradation.

The final major topic is **DDIM (Denoising Diffusion Implicit Models)**. Jeremy first uses the diffusers library's DDPM and DDIM schedulers to verify results, showing how to trick pickle loading by redefining a stub `UNet` class. With 2,048 samples and the diffusers DDPM scheduler, they achieve FID 3.7 (real data: 1.9). Switching to DDIM, they can reduce to 333 steps, 200 steps, or even 100 steps with barely any FID degradation. At 25 steps the images become too smooth — losing texture and fine detail.

Jeremy then re-implements DDIM from scratch. The key equation from the paper gives the next sample as: √ᾱ_{t-1} times predicted x₀, plus √(β̄_{t-1} - σ²) times the predicted noise, plus σ times random noise. Tanishq provides an excellent walkthrough of the DDIM paper's math, explaining how both DDPM and DDIM share the same predicted x₀ equation but differ in their sampling distributions. The critical insight: DDIM introduces a parameter **η** that controls stochasticity. When η=0, sampling is completely deterministic; when η=1, it matches DDPM. This is purely a change in the sampling algorithm — no retraining needed. Tanishq draws the iterative process on a whiteboard: starting from noise, following the score function to estimate x₀, stepping back to an intermediate point, re-estimating, and repeating. The DDIM formulation is simpler because it only requires ᾱ (not individual α or β), making accelerated sampling with arbitrary step subsets straightforward. The lesson closes with Jeremy declaring they've reached a milestone: Stable Diffusion–quality unconditional sampling, all from scratch, and they're about to go beyond.

---

- **Lesson Challenges**
  - Build a W&B callback for experiment tracking using miniai's callback system
  - Implement FID and KID from scratch
  - Implement the Newton-Schulz matrix square root
  - Re-implement DDIM sampling from scratch, matching or beating the diffusers version

- **Potential Research Directions**
  - Investigating optimal input ranges for diffusion models (why does [-0.5, 0.5] work better than [-1, 1]?)
  - Developing unbiased, low-variance metrics for generated image quality (neither FID nor KID is ideal)
  - Exploring the η parameter in DDIM — deterministic vs stochastic sampling and its effect on sample diversity
  - Custom noise schedules beyond linear and cosine
  - Plotting FID across sampling steps to diagnose sampling efficiency
  - Using domain-specific classifiers instead of Inception for FID calculation

- **Homework**
  - Train DDPM_v3 with the corrected input range and reduced β_max, and verify FID improvements
  - Implement DDIM from scratch and experiment with different numbers of sampling steps and η values
  - Scale up to CIFAR-10 using the same miniai code and compare results
  - Try writing a callback that logs experiments to a SQLite database and build a simple front end to browse them

- **Things Jeremy Says You Should Do**
  - Always question "standard" choices that lack theoretical justification — don't blindly copy what everyone else does
  - When a bug fix makes things worse, suspect a compensating bug elsewhere
  - Be very directed in experimentation — carefully thought-out hypotheses rather than sweeping hyperparameter searches
  - Build things step by step, checking shapes at each stage, then paste into functions
  - Take the FID of real data as a baseline to know how good you could possibly get
  - Use consistent sample sizes when comparing FID scores
  - Put screenshots of paper equations directly in notebooks next to the implementing code
  - Consider using W&B for experiment tracking, but don't go crazy on undirected experiments

- **Resources**
  - [Lesson 21 video](https://youtu.be/PXiD7ZjOKhA)
  - [Lesson 21 forum topic](https://forums.fast.ai/t/lesson-21-official-topic/103528)
  - [Lesson 21 course page](https://course.fast.ai/Lessons/lesson21.html)
  - Notebook: `18_fid.ipynb` — FID and KID implementation
  - Notebook: DDPM_v3 / DDIM experiments (continuation of the DDPM series)
  - [DDIM paper — Denoising Diffusion Implicit Models (Song et al., 2020)](https://arxiv.org/abs/2010.02502)
  - [DDPM paper — Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
  - [Improved DDPM / Cosine schedule paper (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)
  - Library: [Weights and Biases (W&B)](https://wandb.ai/)
  - Library: `pytorch_fid` (for Inception V3 model)
  - Library: `diffusers` (HuggingFace — DDPMScheduler, DDIMScheduler)
  - Library: `scipy.linalg` (for matrix square root)
