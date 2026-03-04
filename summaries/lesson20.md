The lesson opens with a small but satisfying refactoring: eliminating the DDPM callback entirely by moving `noisify` into a collation function. Since the collation function is what transforms individual dataset rows into batched tensors—and since `before_batch` in the old callback operated on exactly the output of `default_collate`—the noisification step slots in naturally. A helper `ddpm_dl` wraps this up, and `DataLoaders` is constructed directly from training and test loaders rather than using `DataLoaders.from_dd`. No callback needed. This isn't required for mixed precision; it's flexing the framework muscles.

Mixed precision training arrives next as a new callback. The standard PyTorch recipe—`torch.autocast` as a context manager, `GradScaler` for loss scaling—is decomposed into callback hooks. `autocast.__enter__` fires before the batch, `__exit__` fires after the loss, and the scaler wraps `backward` and `step`. New hooks (`after_predict`, `after_loss`, `after_backward`, `after_step`) are added to the learner to make this possible. Once the callback exists, mixed precision can be dropped into any training run. To actually benefit from it, the GPU must stay busy: batch size is quadrupled, learning rate pushed to 1e-2, epochs increased from 5 to 8, and the optimizer epsilon set to 1e-5 with proper initialization. Training runs about twice as fast for the same quality.

HuggingFace's Accelerate library, created by Sylvain Gugger and maintained with Zach Mueller, wraps this pattern into a single `Accelerator` object. You call `accelerator.prepare()` on the model, optimizer, and data loaders, and it handles mixed precision, multi-GPU, and TPU training transparently. The only code change is replacing `loss.backward()` with `accelerator.backward(loss)`. Jeremy also redesigns the data flow: instead of returning nested tuples from `noisify`, a flat tuple of three elements is returned, and `TrainCB` gains an `n_inputs` parameter that controls how many elements are model inputs vs. loss targets. A tiny `DDPMCB2` extracts `.sample` from UNet2DModel predictions—everything else is generic and decoupled.

A sneaky trick for slow data loading: `MultDL` wraps any data loader and yields each batch multiple times. Each epoch loads data once but produces two (or more) epochs' worth of gradient updates. The GPU stays busy instead of waiting on CPU-bound data loading—particularly useful on Kaggle with two GPUs but only two CPUs.

Homework ideas are floated: experiment with these techniques on other datasets, try different noise schedules, reduce diffusion steps below 1000 (Johno suggests most important activity is in the final 200), and explore non-uniform timestep sampling during training.

Johno then takes over screen sharing and opens the Style Transfer notebook. The goal is to create artistic combinations where one image contributes structure and another contributes style. This was the first-ever fast.ai generative modeling lesson, and many students found it deeply instructive for understanding deep learning more broadly.

The approach starts with the simplest possible version: optimizing an image's raw pixels to match a target via MSE. A `TensorModel` wraps a random tensor as an `nn.Parameter`—no neural network at all, just pixels to optimize. A `LengthDataset` returns dummy values so the learner has something to iterate over. SGD drives the noisy pixels toward the content image. An `ImageLogCB` logs intermediate results, showing noise progressively resolving into the target. Jeremy praises this methodology: build the absolute simplest version first, verify it works, then layer on complexity.

To move beyond pixel-level matching, features are extracted from VGG16, a simple older architecture whose `.features` attribute gives the convolutional body as a flat `nn.Sequential`. The Distill.pub Feature Visualization article illustrates how early layers detect edges and textures while deeper layers capture high-level semantics. Matching features at different layers gives a tunable spectrum: early layers preserve texture and color; deep layers preserve semantic content while allowing visual divergence. The image must be normalized to ImageNet statistics, which requires careful broadcasting—`imagenet_mean[:, None, None]` adds the spatial dimensions needed for the channel-first layout.

A `calc_features` function runs the input through VGG16 layer by layer, collecting activations at specified target layers. The Content Loss (perceptual loss) computes MSE between the generated image's features and the target's features at those layers. This uses VGG16 inside the loss function—not as the model being optimized, but as a learned similarity metric. Optimizing with deep target layers (18, 25) preserves the person's shape but not color; shallow layers (1, 6) produce something much closer to the original. Jeremy warns never to use a mutable list as a default parameter—always use a tuple.

For style transfer proper, the spatial dimension must be removed from feature maps so that texture comparisons are location-independent. The Gram Matrix achieves this: flatten the spatial dimensions of a feature map, then multiply the result by its own transpose. The output is a (features × features) matrix encoding which features co-occur and how frequently, with no spatial information. Jeremy draws connections to Word2Vec co-occurrence, CLIP loss, and covariance matrices—all variations on multiplying a matrix by its transpose, differing only in what matrix. Scaling by (width × height) normalizes for image size.

The Style Loss uses Gram Matrices: compute Gram matrices of the style image at target layers, then during optimization compute MSE between the generated image's Gram matrices and the target's. The total loss combines content loss (preserving structure) and style loss (transferring texture). Starting from the content image rather than noise, optimization progressively adds the style image's textures while maintaining the content's semantic structure. The result is strikingly intelligent—spider webs drape naturally along the subject's arm while the face, important for recognition, remains largely untouched.

Experimentation possibilities abound: vary target layers, change the content-to-style loss weighting, start from noise or the style image, try different architectures (ConvNeXt, ResNet), and apply perceptual/style losses to practical tasks like super resolution or diffusion guidance.

The lesson's final act introduces Neural Cellular Automata (NCA). Conway's Game of Life demonstrates how simple local rules produce complex emergent behavior. Each cell sees only its immediate neighbors, yet large-scale patterns arise—mirroring biological systems like ant colonies and slime molds. The key innovation: replace the hard-coded update rule with a tiny neural network.

Alexander Mordvintsev's "Growing Neural Cellular Automata" from Google Brain is demonstrated—a pixel grid where each cell runs an identical neural network, seeing only its 3×3 neighborhood plus hidden channels. Starting from a single seed pixel, the system grows into a target image (a lizard emoji) and can repair damage, despite no cell knowing the global structure. Applications span swarm robotics, subterranean rescue, nanotechnology, and self-assembling systems.

Rather than reproducing the lizard paper, Johno follows a related paper training NCA to match textures via style loss—perfect for the Gram Matrix approach just developed. The implementation uses circular padding (wrapping edges for seamless tiling), hard-coded perception filters (identity, horizontal/vertical gradients, Sobel filter—inspired by biological gradient sensing), and 1×1 convolutions as efficient per-pixel MLPs. The model has only 168 parameters. A stochastic update mask (like dropout) breaks symmetry so uniform initial states can evolve.

Training uses a pool of 256 grids. Most training steps start from previous outputs rather than the initial state, forcing the model to maintain the target texture over time rather than just growing into it once. The loss combines style loss and an overflow penalty. Gradient normalization controls exploding gradients from the 50+ chained update steps. The result: starting from random noise, the tiny network produces seamless tileable textures matching the style target.

Johno demonstrates exporting the trained NCA weights into a WebGL fragment shader—a GLSL program running per-pixel in the browser at real-time speeds. One example uses style loss plus CLIP guidance for "glowing dragon scales." The simulation is interactive: destroying part of the grid triggers self-repair. Jeremy calls it one of the coolest things he's seen.

---

- **Lesson Challenges**
  - Get rid of the DDPM callback using a collation function
  - Implement mixed precision training as a callback
  - Use HuggingFace Accelerate for mixed precision and multi-GPU
  - Implement `MultDL` for repeated batch yielding
  - Optimize raw pixels to match a target image via MSE
  - Implement content loss using VGG16 features at various layers
  - Implement the Gram Matrix for style loss
  - Combine content and style loss for full style transfer
  - Build and train a Neural Cellular Automata model for texture synthesis
  - Export NCA weights to a WebGL shader for real-time interactive rendering

- **Potential Research Directions**
  - Training ResNets without global pooling (using VGG-style flattening)—potentially unexplored territory
  - What special properties of VGG make it effective for style transfer vs. other architectures (ConvNeXt, ResNet)
  - Using perceptual/style loss as guidance functions in diffusion models
  - Denoising, stylizing, and image-restoration cellular automata
  - NCA at different scales and with video inputs
  - Non-uniform timestep probability distributions during diffusion training
  - Reducing diffusion steps to ~200 (where most important activity occurs)
  - The relationship between changing β schedules and changing timestep sampling probabilities

- **Homework**
  - Run your own experiments with mixed precision on other datasets
  - Try different noise schedules and fewer diffusion steps
  - Experiment with different layer combinations for content and style loss
  - Try starting style transfer from random noise or the style image
  - Vary the content-to-style loss ratio, training duration, and learning rate
  - Try different style/content image pairs
  - Implement feature extraction using hooks instead of the sequential iteration approach
  - Experiment with more NCA channels and hidden neurons for richer textures

- **Things Jeremy Says You Should Do**
  - Build the absolute simplest version first and fully verify it before adding complexity
  - Never use a mutable list as a default parameter in Python—use a tuple instead
  - Read the original Gatys et al. paper and practice mapping Johno's visual explanations back to the paper's language
  - Use the right layer of abstraction for what you're doing—don't jump to the highest level, but don't start at the lowest either
  - Reread the datasets notebook (notebook 5) to remind yourself about collation functions

- **Resources**
  - [Lesson 20 video](https://youtu.be/PdNHkTLU2oQ)
  - [Lesson 20 discussion thread](https://forums.fast.ai/t/lesson-20-official-topic/103322)
  - [Lesson 20 course page](https://course.fast.ai/Lessons/lesson20.html)
  - [15_StyleTransfer notebook](https://github.com/fastai/course22p2/blob/master/nbs/15a_StyleTransfer.ipynb)
  - [15B_NCA notebook](https://github.com/fastai/course22p2/blob/master/nbs/15b_NCA.ipynb)
  - [09_learner notebook (updated version with new callback hooks)](https://github.com/fastai/course22p2/blob/master/nbs/09_learner.ipynb)
  - [Gatys et al., "A Neural Algorithm of Artistic Style" (arXiv:1508.06576)](https://arxiv.org/abs/1508.06576)
  - [Distill.pub — Feature Visualization](https://distill.pub/2017/feature-visualization/)
  - [Growing Neural Cellular Automata (Mordvintsev et al., Google Brain)](https://distill.pub/2020/growing-ca/)
  - [Self-Organising Textures (Mordvintsev et al.)](https://distill.pub/selforg/2021/textures/)
  - [HuggingFace Accelerate library](https://huggingface.co/docs/accelerate)
  - [PyTorch mixed precision training docs](https://pytorch.org/docs/stable/amp.html)
  - [Shadertoy (WebGL shader platform)](https://www.shadertoy.com/)
  - [Zeiler and Fergus — Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.1901)
