The final lesson of Part 2 opens with Jeremy explaining that one piece of Stable Diffusion remains — the CLIP embeddings — which will be covered in the upcoming NLP-focused Part 3. Given the explosion of interest in GPT-4 since the course began, they've decided to pivot toward NLP and delay CLIP accordingly. With that framing, Jeremy hands the floor to Jono for a creative application of pixel-level diffusion: generating bird calls.

Jono walks through treating audio as images. A microphone samples air pressure thousands of times per second — this dataset uses 32,000 samples per second — producing a massive 1D waveform. Working directly with waveforms is impractical: they're enormous, they mix wildly different frequency scales together, and nearby samples don't relate to each other the way nearby pixels do. The solution is the spectrogram, which maps time on the x-axis and frequency on the y-axis, with intensity shown as brightness. Jono demonstrates live with a spectrogram visualizer: a pure tone produces a single peak, speech produces many overtones, and a chord shows distinct harmonics. A bird call, rendered as a spectrogram, reveals beautiful spatial patterns — patterns that look like something a diffusion model could learn.

The specific variant used is the **Mel spectrogram**, which warps the frequency axis to match human hearing perception and applies a log transformation so that intensity corresponds to perceived loudness. The `Mel` class from diffusers handles splitting audio into chunks of the right size for a target resolution (128×128), converting to a spectrogram image, and crucially, converting back. That inverse conversion uses the **Griffin-Lim algorithm** — an iterative optimization that guesses the missing phase information, since the spectrogram is a lossy representation that discards how much each frequency component is shifted in time. The result isn't perfect, but it works.

With audio now represented as 128×128 grayscale images, Jono plugs them straight into the same Simple Diffusion pipeline from earlier notebooks. The model is tiny — five channel levels (128→64→32→16→8), no attention, just Transformer blocks in the middle of the U-Net where resolution is smallest. After 15 epochs and DDPM sampling with 100 steps, the generated spectrograms convert back into sounds that genuinely resemble bird calls. Jono notes that the **Riffusion** project does something similar for music, remarkably by fine-tuning the original Stable Diffusion image model on spectrograms — evidence that visual representations learned for photographs transfer meaningfully to audio.

The lesson then pivots to the VAE. Jeremy motivates it by recalling that in Lesson 1, they saw the Stable Diffusion VAE compress a 256×256×3 image down to 32×32×4 latents — a dramatic reduction that makes diffusion training far more tractable. Before building a VAE, they revisit a plain autoencoder on Fashion MNIST: flatten 28×28 images to 784-dimensional vectors, compress through a hidden layer of 400 to a 200-dimensional bottleneck, and decode back. After training with Adam for 20 epochs, reconstructions are passable but crude. The real test is generation: decoding random noise through the decoder produces nothing recognizable, because the 200-dimensional space has no structure — the model was only trained to reconstruct specific inputs, not to make the space navigable.

The VAE addresses this by replacing the encoder's single output with two parallel heads: one producing **μ** (means) and one producing **log variance**. Latents are sampled as z = μ + σ·ε, where ε is standard normal noise and σ = exp(½·lv). A second loss term — **KL divergence** — is added alongside the reconstruction loss. The KLD loss pushes μ toward zero and variance toward one, measuring the divergence between the learned distribution and a standard normal. Jeremy plots x − eˣ to show it has a minimum at x = 0, corresponding to unit variance. Jono clarifies the intuition: the VAE is being asked to decode not just each exact encoded point, but the *neighborhood around* that point, and to keep the overall space organized around zero mean and unit variance. This is a harder problem — reconstruction quality drops from 0.26 BCE to 0.31 — but generation from random noise now produces recognizable items of clothing. The latent space has been conditioned to be smooth and navigable.

Jeremy then asked Bing/GPT-4 to explain the variance extremes. Very low variance makes the latent space peaked and non-generative; very high variance makes it noisy and incoherent. The sweet spot is variance of exactly one — precisely what KL divergence enforces.

Moving to real-scale generation, they turn to the **Stable Diffusion VAE** and the **LSUN Bedrooms** dataset (millions of bedroom photos, with a 20% subset hosted on AWS). The dataset class uses PyTorch's `read_image` for speed, crops to 256×256, and feeds batches of 64 through the pre-trained VAE encoder. The encoded latents have shape [16, 4, 32, 32] — 48× smaller than the originals. Visualizing the first three channels shows blurry but recognizable geometry; decoding recovers the originals almost perfectly, with only minor losses in small text and fine facial detail.

For efficient training, Jeremy advocates encoding everything once and saving to **memory-mapped numpy files** (`np.memmap`). The OS handles caching transparently — only the needed portions reside in RAM at any moment. After encoding all 303,125 images, they flush to disk, reload with `mode='r'`, and verify the round trip. A memmap array has the same interface as a list, so it serves directly as a PyTorch dataset with no wrapper.

The latents' standard deviation is much higher than one, so they divide by ~5 to normalize. Training proceeds with batch size 128, the same Simple Diffusion architecture adapted for 4-channel input instead of 3. After a few hours on a single GPU, the model generates 256×256 bedroom images — some quite convincing, others with artifacts. Tanishq's 100-epoch run (~15 hours on an A100) produces noticeably better results. Jeremy also tries the diffusers UNet, getting marginally better loss but no obvious visual improvement.

The lesson's most forward-looking segment is Jeremy's experiment with a **pre-trained backbone on latents**. He trains an ImageNet classifier directly on VAE-encoded latents using a pre-activation ResNet with compute concentrated at low resolution (mirroring Simple Diffusion's philosophy). Data augmentation — random padding, cropping, random erasing — works on 4-channel latents just as well as on RGB images. After a single epoch, the classifier hits 25% accuracy on 1,000 ImageNet classes; after 40 epochs, it reaches **66%** — surpassing AlexNet's historic ~63% and approaching ResNet-34's ~74%. This opens intriguing possibilities: latent-space classifiers for guidance, latent-CLIP models, latent-perceptual-loss, knowledge distillation from pixel-space models — all computationally cheaper than decoding.

Their colleague Molly demonstrates the same pipeline on CelebA-HQ, producing convincing face generations. Jeremy wraps up by congratulating everyone on completing the course and emphasizing that understanding these techniques from scratch puts students far ahead of most practitioners. The next part will tackle NLP, GPT-4-style models, and finally complete Stable Diffusion from scratch with CLIP.

---

- **Lesson Challenges**
  - Try generating audio with the spectrogram diffusion approach on different audio datasets
  - Experiment with the VAE on Fashion MNIST — observe the effect of KL divergence weight
  - Train latent diffusion on LSUN Bedrooms or CelebA-HQ and compare results across epochs
  - Train an ImageNet classifier on latents and compare accuracy to pixel-space baselines

- **Potential Research Directions**
  - Pre-trained latent-space backbones for diffusion (analogous to pre-trained backbones for super-resolution)
  - Latent-CLIP: distilling a CLIP model that operates directly on VAE latents
  - Latent-perceptual-loss: whether perceptual loss adds value when the VAE was already trained with it
  - Knowledge distillation from pixel-space classifiers to latent-space classifiers for richer training signal
  - Latent-FID: computing FID scores directly in latent space
  - Using diffusers UNet with Stable Diffusion's channel settings for extended training
  - Better phase reconstruction (deep learning replacements for Griffin-Lim)
  - Combining pre-trained ResNet cross-connections with the Simple Diffusion U-Net

- **Homework**
  - Rebuild the audio diffusion pipeline and the latent diffusion pipeline from scratch
  - Experiment with different datasets (CelebA-HQ, your own audio, etc.)
  - Try the pre-trained latent backbone idea with your own diffusion model

- **Things Jeremy Says You Should Do**
  - If you binged the videos, go back and actually build everything yourself — experiment
  - Don't feel unqualified; if you understand diffusion from scratch, you are far ahead of most people in this space
  - Combine these techniques with your own domain expertise
  - Join the fast.ai Discord "generative" channel and collaborate
  - Take the experimentalist approach: try things, share results, iterate

- **Resources**
  - [Lesson 25 discussion thread](https://forums.fast.ai/t/lesson-25-official-topic/104573)
  - [02_diffusion_for_audio.ipynb (HuggingFace diffusion-models-class)](https://github.com/huggingface/diffusion-models-class/blob/main/unit4/02_diffusion_for_audio.ipynb)
  - [Jono's Simple Diffusion for audio (Colab)](https://colab.research.google.com/drive/1b3CeZB2FfRGr5NPYDVvk34hyZFBtgub5?usp=sharing)
  - [Notebook 29: VAE](https://github.com/fastai/course22p2/blob/master/nbs/29_vae.ipynb)
  - [Notebook 30: LSUN diffusion latents](https://github.com/fastai/course22p2/blob/master/nbs/30_lsun_diffusion-latents.ipynb)
  - [Notebook 31: ImageNet latents (wide)](https://github.com/fastai/course22p2/blob/master/nbs/31_imgnet_latents-widish.ipynb)
  - [Riffusion demo](https://www.riffusion.com/) | [Riffusion repo](https://github.com/riffusion/riffusion)
  - [Xeno-canto (bird call recordings)](https://xeno-canto.org/)
  - [LSUN dataset](https://www.yf.io/p/lsun)
  - [fast.ai Discord](https://discord.gg/fastai)
  - [fast.ai Forums](https://forums.fast.ai/)
