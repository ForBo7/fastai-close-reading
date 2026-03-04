The lesson opens with Jeremy, joined by Johno and Tanishq, confessing to a double bug in the previous Karras notebook's FID measurement. The data loaders fed images scaled to $(-0.5, +0.5)$ while the model expected $(-1, +1)$, and the sampling path made the same error in reverse — so both real and generated features looked equally low-contrast, producing misleadingly low FID scores. Once fixed, FIDs land around 5–6 (reals at 2.5), and the cosine schedule model — which accidentally scaled to $(-0.5, +0.5)$ then multiplied by 2 post-sampling — actually outperforms Karras. The takeaway: Karras's unit-variance scaling isn't necessarily optimal, and the dependent variable choice (raw $\mathcal{N}(0,1)$ noise vs. the Karras $c$-mix) matters too.

With Fashion MNIST retired, the lesson moves to Tiny ImageNet ($64 \times 64$, 200 classes), downloaded from Stanford's servers. Jeremy builds the dataset pipeline entirely by hand: a `TinyDS` class using `glob` with `**/*.JPEG` and `recursive=True`, a `TinyValDS` subclass that looks up labels from `val_annotations.txt` via a dictionary built from a generator comprehension (`dict(iterable)` from key-value pairs), and a general-purpose `TransformDataset` wrapper applying `transform_x` and `transform_y`. WordNet IDs are converted to integers via a `{v: k}` dictionary comprehension over `enumerate`. Images are read as RGB (forcing three channels for occasional grayscale images), divided by 255, and normalized with per-channel mean and standard deviation computed from one batch. The `words.txt` file maps WordNet codes to human-readable synsets — Egyptian cat, guacamole, monarch butterfly — and the categories are noted to be deliberately tricky.

For data augmentation, `RandomResizedCrop` is rejected for low-res images (it blurs), replaced by small padding plus `RandomCrop(64)`, `RandomHorizontalFlip`, and `RandomErasing`. These are composed via `nn.Sequential` and applied at batch level through a `BatchTransformSpatial` callback. The initial model reuses `get_drop_model` from Fashion MNIST — kernel-size-5 opening conv followed by res blocks — and the model summary shows constant megaflops per layer (channels doubling as grid halves), confirming even compute distribution. Training with AdamW and mixed precision for 25 epochs reaches ~59% accuracy, with training at 91% indicating overfitting pressure.

Benchmarking against Papers with Code reveals that 90%+ results all use pretrained models; the best from-scratch result is ~72%. A published comparison table shows ResNet-18 achieving 63–65% with various mix-up strategies. Jeremy deepens the model by replacing single res blocks with `res_blocks` — a `Sequential` of multiple blocks per stage (e.g. `[3, 2, 2]`) with stride 2 only on the last — roughly doubling megaflops. This reaches 62% in 25 epochs with less overfitting, suggesting longer training would help with more augmentation.

TrivialAugment, from Frank Hutter's lab, is introduced as a reaction to the enormous compute cost of AutoAugment and RandAugment. The algorithm simply picks one random augmentation per image at a random strength — beautifully simple, tuning-free, and at least as effective. It's built into PyTorch as `T.TrivialAugmentWide()`. Jeremy finds batch-level TrivialAugment is harmful (one bad augmentation poisons the whole batch), so augmentation moves to per-item in the dataset. The pipeline becomes: `Image.open` → PIL augmentations → `to_tensor` → normalize → random erase (after normalization, since it uses unit Gaussian noise).

Pre-activation ResNets are then introduced via He Kaiming's follow-up paper. The original ResBlock applies activation after the skip addition, breaking the pure identity path. Pre-activation reorders to norm → act → conv → norm → act → conv → add, giving a clean identity skip. The model can't start with a pre-act block (it would discard half the data via activation), so a plain conv opens the network, and a final activation + batch norm closes it. Training for 50 epochs yields 65%, and 200 epochs reaches 67.5% — competitive with published mix-up results on larger models, all built from scratch. In discussion, Tanishq reports pre-activation was actually worse for Fashion MNIST and shallower models, which makes theoretical sense (the identity path matters more for very deep networks) but remains an open experimental question.

The lesson pivots to super resolution. The independent variable is the $64 \times 64$ image resized down to $32 \times 32$ then `interpolate`d back up (pixels doubled), so input and output are the same spatial size. Augmentation (crop, flip) must be applied identically to both input and target. Random erasing is applied only to the input, making the task harder and teaching the model more about what pictures look like. The blocky inputs visibly lose fine details — a cat loses its eyes entirely at $32 \times 32$.

An autoencoder baseline (res blocks striding down, `nn.Upsample` + res blocks going up, MSE loss) produces terrible results after 5 epochs (loss 0.207) — going through a tiny bottleneck destroys too much information. The UNet architecture, originally from 2015 medical imaging and now used in Stable Diffusion, solves this with skip connections that copy activations from each downsampling level directly across to the corresponding upsampling level. The downsampling path uses `nn.ModuleList` with activations saved manually in `forward`. The upsampling path adds back saved activations: `x = x + layers[n-i-1]`. The very first input is also saved and added at the very end. Johno notes that addition (rather than concatenation) suits super resolution — you're saying "this pixel is basically right, just modify it slightly." Tanishq frames it as a boosting intuition: the skip passes the original, the network produces an update.

For initialization, Jeremy zeros out the weights of the last up-path res block and final conv so the untrained model outputs its input unchanged — a sensible starting point for super resolution. Tanishq raises symmetry-breaking concerns, but Jeremy explains that with different weights in all previous layers, gradients will differ regardless. The UNet's MSE loss after 1 epoch is already 0.086 (vs. the autoencoder's 0.207 after 5), reaching 0.073 after 5 epochs. Results are decent but blurry — MSE encourages predicting the average when uncertain.

Perceptual loss addresses this. The idea parallels FID: compare intermediate features of a pretrained classifier on the output and target images. Jeremy truncates the 25-epoch Tiny ImageNet classifier at the fourth res block and uses its features (shape `[batch, 124, H, W]`) in a combined loss: $\mathcal{L} = \text{MSE}(\hat{y}, y) + \frac{1}{10}\text{MSE}(f(\hat{y}), f(y))$, where the $\frac{1}{10}$ factor is a rough balancing term found by running one epoch and observing relative magnitudes. Target features use `torch.no_grad()`. After 20 epochs, the kid has pupils, backgrounds gain texture, and the koala looks less blurry — a clear visual improvement, though no single metric captures it well. Jason Antic is cited as still relying heavily on eyeballing for image restoration evaluation.

Transfer learning is then applied: the pretrained classifier's weights are loaded into the UNet's downsampling path via `state_dict()` / `load_state_dict()`, then frozen with `requires_grad_(False)`. Training just the up path for 1 epoch gives loss 0.255 — already better than the previous 20-epoch run. After unfreezing and training 20 more epochs, loss drops to 0.198 with visibly better results.

Finally, cross connections are introduced. The downsampling activations serve double duty (classification features and skip connections), which is suboptimal. A res block is inserted on each skip path before addition, giving the model dedicated capacity to transform activations for the upsampling task. Loss improves from 0.198 to 0.189, and the koala finally shows a hint of an eye.

The lesson closes with a rich set of suggested exercises applying the same UNet + perceptual loss framework to different image-to-image tasks.

---

**Lesson Challenges**

- Build a UNet for image segmentation
- Style transfer: train a UNet to produce Van Gogh–style outputs (consider removing top-level skip connections)
- Colorization: grayscale input → color output
- In-painting: delete image centers and train the model to fill them; extend to panorama generation
- JPEG artifact removal: compress images heavily, train to restore
- Drawing to painting: edge detection on paintings as input
- Watermark removal: overlay text/watermarks with PIL, train to remove (applicable to radiology PII)
- Better super resolution on full ImageNet with larger images and longer training

**Potential Research Directions**

- Why does the cosine schedule with accidental $(-0.5, +0.5)$ scaling outperform Karras's unit-variance scaling? Is there an optimal pixel range for diffusion?
- Why are pre-activation ResNets worse on shallow models/smaller datasets? Systematic study across depth and augmentation regimes
- Optimal loss weighting between MSE and perceptual loss — can it be learned or scheduled?
- Better metrics for super resolution quality beyond MSE and FID
- Cross connections in UNets — exploring where and how many cross-conv blocks help most
- TrivialAugment vs. batch-level augmentation: understanding loss surface dynamics when entire batches receive extreme augmentation

**Homework**

- Implement the suggested exercises (segmentation, style transfer, colorization, in-painting, JPEG artifact removal, watermark removal, etc.)
- Experiment with pre-activation ResNets on different datasets and depths
- Try cross connections in your own UNet variants

**Things Jeremy Says You Should Do**

- Build datasets manually — just implement `__len__` and `__getitem__`; you don't need to inherit from anything
- Use padding + random crop instead of `RandomResizedCrop` for low-resolution images
- Check that megaflops are roughly constant across layers to ensure even compute distribution
- Use TrivialAugment per-item, not per-batch, to avoid loss spikes
- Zero-initialize the final layers of a UNet so it starts as an identity function
- Use perceptual loss instead of pure MSE for image generation/restoration tasks
- Use transfer learning (freeze → train head → unfreeze → fine-tune) even for UNets
- Add cross connections to give skip paths their own processing capacity
- Look at your images — there's no substitute for visual inspection in image generation

**Resources**

- [Lesson 23 course page](https://course.fast.ai/Lessons/lesson23.html)
- [Lesson 23 discussion thread](https://forums.fast.ai/t/lesson-23-official-topic/103965)
- [TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation](https://arxiv.org/pdf/2103.10158.pdf)
- [Identity Mappings in Deep Residual Networks (He et al.)](https://arxiv.org/abs/1603.05027)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [Papers with Code — Tiny ImageNet Classification](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1)
- Tiny ImageNet dataset (Stanford servers)
- PyTorch `torchvision.transforms.TrivialAugmentWide`
