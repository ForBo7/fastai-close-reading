The lesson opens with a showcase of student work from the past week. @puru demonstrates spherical linear interpolation (slerp) between two latent noise starting points, producing smooth transitions — first between otter images, then morphing an old car into a modern Ferrari. @namrata extends this idea by interpolating from a dinosaur to a bird, yielding a striking intermediate "dino-bird." John Richmond earns "dad of the week" by turning his daughter's dog into a unicorn using img2img at varying strength values — too low and the dog barely changes, too high and the dog vanishes entirely, but at 0.3 he finds the sweet spot. Maureen applies different artist style prompts to Johno's parrot image, discovering that a Frida Kahlo prompt produces Frida Kahlo herself (because she appears in virtually all her own paintings), while the Jackson Pollock version retains a ghostly parrot shape. Jeremy reminds everyone to watch Johno's Stable Diffusion walkthrough videos and the Wasim–Tanishq video on the math of diffusion, quoting a student named Alex who assumed the math video would be incomprehensible but found it entirely accessible. Jason Antic, creator of DeOldify, has joined the research team and, within a single week, generated high-quality face images from scratch on a single GPU in just a few hours using classic deep learning optimizers rather than differential equation solvers — a research direction Jeremy considers extremely promising.

A quick recap of Lesson 9 follows, now presented as polished slides converted from hand-drawn OneNote notes. The core idea is restated: take a digit (say a seven), add noise, feed the noisy image to a U-Net that predicts the noise, compare that prediction to actual noise via a loss function, and update the U-Net's weights. An embedding of the digit label can be passed in as conditioning so the model later generates specific digits on demand. The VAE/latents piece is glossed over as a computational shortcut — compressing images into a smaller latent space. For richer prompts beyond single digits, CLIP provides the bridge: an image encoder and a text encoder are trained so that matching image–text pairs produce similar feature vectors, using contrastive loss (the "CL" in CLIP). At inference time, we feed a text embedding plus random noise into the U-Net, which predicts noise to subtract. We subtract only a little, then repeat — progressively denoising across many steps. The lesson shows intermediate results at steps 0, 6, 12, 18, 24, 30, 36, 42, where a face gradually emerges from static. The noisy images look strange (not like Gaussian noise) because they are decoded VAE latents.

Jeremy then introduces three recent papers. The first, "Progressive Distillation for Fast Sampling of Diffusion Models," uses a teacher–student distillation scheme. The fully trained Stable Diffusion model (teacher) runs two denoising steps; a student model learns to jump directly from the input to the two-step result. That student then becomes the new teacher, and a fresh student learns to match two of the previous student's steps — effectively doubling the step size each round (1 → 2 → 4 → 8), collapsing what originally took hundreds of steps down to just a handful. The second paper, "On Distillation of Guided Diffusion Models," extends this to classifier-free guided diffusion (CFGD). Normally CFGD requires running the U-Net twice per step (once with the prompt, once with an empty prompt) and combining the outputs via a guidance scale. This paper folds guidance into the distillation process: the student receives noise, prompt, and guidance scale as inputs, and learns to replicate the teacher's guided output across a range of guidance values — reducing 60 steps down to as few as 4. Jeremy encourages watching Johno's paper walkthrough video and notes that the actual algorithms, once you look past the math, are mostly primary-school arithmetic turned into a few lines of code. The third paper, "Imagic: Text-Based Real Image Editing with Diffusion Models," released just hours before the lesson, demonstrates editing real photographs via text prompts — making a bird spread its wings, a dog sit down, a person give a thumbs up — while preserving identity and context. The method has three steps: (A) optimize the text embedding to make the diffusion model reconstruct the input image, (B) fine-tune the entire model to output the exact input image given that optimized embedding, and (C) interpolate between the original target embedding and the optimized embedding, then pass it through the fine-tuned model. Jeremy emphasizes the societal implications: anyone can now generate believable photos of events that never occurred.

The lesson then dives into the Stable Diffusion pipeline code, built from scratch using the individual Hugging Face components. The text encoder (CLIP's `clip-vit-large-patch-14`), VAE (`CompVis/stable-diffusion-v1-4`), and U-Net are loaded via `from_pretrained`. A scheduler (Katherine Crowson's LMS discrete scheduler) maps time steps to noise levels. The full inference loop is walked through step by step: tokenize the prompt (padding to fixed length — GPUs prefer uniform-sized batches), encode via CLIP to get a 77×768 embedding tensor, do the same for the empty string and concatenate both (a trick so the GPU processes guided and unguided predictions simultaneously). Random latents are created at shape 1×4×H/8×W/8 (the factor-of-8 compression comes from the VAE), scaled by the scheduler's initial noise sigma. The denoising loop iterates over 70 time steps (jumping from 999 down by ~14 each time — Jeremy stresses that "time step" is a terrible name since these are really just noise-level indices, not even integers). Inside the loop: concatenate the two latent copies, scale, run through the U-Net, chunk the output into unconditional and text predictions, apply the guidance scale formula, and call `scheduler.step` to update latents. After the loop, the VAE decoder converts latents back to pixel space (dividing by 0.18215, a magic number from the paper), values are clamped from [-1,1] to [0,1], reordered for PIL, and converted to a displayable image. Jeremy then consolidates everything into two compact cells and introduces single-letter variable names (g for guidance scale, u for unconditional prediction, t for text prediction) to make the code look as close to the paper's equations as possible. He generates "a photograph of an astronaut riding a horse" and "an oil painting of an astronaut riding a horse in the style of Grant Wood." The code is further refactored into three small functions that fit on one screen.

The second half of the lesson marks a major pivot: building everything from the foundations toward Stable Diffusion. The foundations are defined as Python, the Python standard library, matplotlib, and Jupyter/nbdev — nothing else. Once a feature is reimplemented from scratch, the real library version is then permitted. The plan is to create smaller but architecturally identical versions of the VAE, U-Net, and CLIP encoder, collectively forming a library called "mini-ai." Jeremy works with MNIST — 50,000 handwritten digit images, 28×28 pixels, loaded via gzip and pickle, unpacked using destructuring assignment.

He pauses to advocate strongly for reading the Python documentation, demonstrating Jupyter's Shift+Tab for inspecting signatures, `?` for docs, and `??` for source code, plus keyboard shortcuts like Ctrl+Shift+Hyphen to split cells and Alt+Enter to create new ones. The flat 784-element pixel list is reshaped into 28×28 using a custom `chunks` generator function — a two-line function using `yield`, which Jeremy explains as one of Python's most underused features. He demonstrates `itertools.islice`, the two-argument form of `iter()` (callable plus sentinel), and the general power of iterators for streaming data without loading everything into memory.

To enable two-dimensional indexing (`image[20, 15]` instead of `image[20][15]`), Jeremy builds a minimal `Matrix` class with `__init__` and `__getitem__` dunder methods — just enough to justify graduating to PyTorch tensors, which provide the same indexing plus vastly more. The `map` function converts all data to tensors in one line. He explains `reshape` (including the `-1` shorthand), then takes a historical detour into APL — the array-oriented language created by Ken Iverson in the late 1950s, inspired by tensor analysis. APL called them "arrays," NumPy inherited that name, and PyTorch inexplicably switched to "tensors" — but they are all the same thing: rectangular blocks of numbers. Rank is just the number of dimensions. Jeremy mentions the fast.ai forums' APL study section with 17 sessions covering the entire language.

The lesson's final deep dive is into random number generation from scratch. True randomness requires physical sources: the ANU quantum vacuum generator, Cloudflare's wall of lava lamps, or Intel's RDRAND instruction — all too slow for deep learning workloads. Instead, pseudo-random number generators (PRNGs) use deterministic math that produces sequences appearing random. Jeremy implements the Wichmann–Hill algorithm (Python's PRNG before version 2.3), showing how `seed()` initializes global state as a tuple of three integers, and each call to `rand()` transforms and replaces that state. The output passes both visual tests: a scatter plot shows no obvious correlation between successive values, and a histogram shows uniform distribution.

Then comes a critical practical warning. When `os.fork()` creates a child process, both parent and child inherit identical copies of the random state — and therefore produce identical "random" numbers. Jeremy demonstrates this bug live: his custom `rand()`, PyTorch's `torch.rand()`, and NumPy's `np.random.rand()` all fail after a fork, producing matching values in parent and child. Only Python's built-in `random.random()` correctly reinitializes. This matters enormously in deep learning, where data loaders commonly fork worker processes for parallel augmentation. fastai itself once had this exact bug. Jeremy benchmarks his custom PRNG against PyTorch's: 3ms vs 73µs for generating 7,840 numbers — so from here on, the real PyTorch version is used. The lesson concludes by creating a 784×10 tensor of random numbers — the weight matrix for a linear classifier mapping 28×28 pixel inputs to 10 digit classes.

---

**Lesson Challenges**

- Implement negative prompts in the from-scratch Stable Diffusion pipeline
- Implement image-to-image generation in the from-scratch pipeline
- Add a callback system to the from-scratch pipeline
- Reimplement `chunks` and the `Matrix` class to understand iterators and dunder methods
- Explore the `os.fork()` random number bug with different PRNG libraries

**Potential Research Directions**

- Using classic deep learning optimizers instead of differential equation solvers for diffusion (Jason Antic's work)
- Progressive distillation to reduce sampling steps in diffusion models
- Distilling classifier-free guided diffusion into a single model
- Text-based real image editing (Imagic) — societal implications and extensions
- Spherical linear interpolation in latent space for smooth image transitions
- Better handling of random state across forked processes in PyTorch/NumPy

**Homework**

- Implement negative prompts in your version of the pipeline
- Implement image-to-image generation
- Implement callbacks for the inference loop
- Read the PyTorch tensor documentation end-to-end
- Read the Python standard library documentation for every method you use
- Watch Johno's Stable Diffusion walkthrough and paper walkthrough videos
- Watch Wasim and Tanishq's "Math of Diffusion" video
- Explore the "Share your work here" thread on the forums

**Things Jeremy Says You Should Do**

- Read the Python documentation for every single method you use and look at every single option it takes, then practice in Jupyter
- Read the PyTorch tensor documentation — scroll through the whole thing to know roughly what exists
- Use Shift+Tab in Jupyter to inspect signatures; use `?` and `??` for docs and source
- Watch Johno's paper walkthrough video to see how an expert reads papers (skipping most math, focusing on the algorithm)
- Watch the Wasim and Tanishq video on the math of diffusion, even if you don't think of yourself as a math person
- Check out the "Share your work here" thread on the forums
- Try to have all important code fit on one screen at once
- Use single-letter variable names when experimenting with equations, matching the paper's notation as closely as possible
- Understand iterators and generators — they can replace huge pieces of enterprise software
- Be aware that PyTorch and NumPy random number generators do not correctly reinitialize after `os.fork()`

**Resources**

- [Lesson 10 official topic (forums)](https://forums.fast.ai/t/lesson-10-official-topic/101171)
- [course22p2 repo](https://github.com/fastai/course22p2)
- [diffusion-nbs repo](https://github.com/fastai/diffusion-nbs) — `stable_diffusion.ipynb`
- [Paper walkthrough video by Johno](https://www.youtube.com/watch?v=ZXuK6IRJlnk) — "Progressive Distillation for Fast Sampling of Diffusion Models"
- [Johno's Stable Diffusion walkthrough videos](https://www.youtube.com/watch?v=ZXuK6IRJlnk) (linked from course page and forums)
- [Wasim & Tanishq: Math of Diffusion video](https://youtu.be/mYpjmM7O-30)
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) — Tim Salimans, Jonathan Ho
- [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.09671) — Chenlin Meng et al.
- [Imagic: Text-Based Real Image Editing with Diffusion Models](https://arxiv.org/abs/2210.09276) — Bahjat Kawar et al.
- [Fashion-MNIST reimplementation of the lesson by @strickvl](https://mlops.systems/computervision/fastai/parttwo/2022/10/24/foundations-mnist-basics.html)
- [APL & Array Programming section on fast.ai forums](https://forums.fast.ai/c/array-programming/56)
- [TryAPL (online APL interpreter)](https://tryapl.org/)
- [Cloudflare lava lamp wall (randomness blog post)](https://blog.cloudflare.com/randomness-101-lavarand-in-production/)
- [ANU Quantum Random Number Generator](https://qrng.anu.edu.au/)
- [DeOldify by Jason Antic](https://github.com/jantic/DeOldify)
- [CLIP model: openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- [Stable Diffusion model: CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- Python standard library: `pathlib`, `gzip`, `pickle`, `itertools`, `urllib.request`, `os.fork()`
- Ken Iverson's "Notation as a Tool of Thought" (1979 Turing Award lecture)
