The lesson opens by framing its twin goals: building a convolutional autoencoder and, when the difficulties of doing so become apparent, beginning to construct a deep learning framework that will carry through the rest of the course. Code is now being imported from `miniai`, a small library built with nbdev — notebooks with `default_exp` and `export` directives, `nbdev_export()` to generate modules, configuration in `settings.ini`, installed locally with `pip install -e .`.

Convolutions are introduced as a way of telling the network about the spatial structure of its input. Nearby pixels tend to be similar, differences across dimensions carry meaning, and the same pattern appearing in different locations should be recognized as the same thing. A CNN encodes these priors directly into its architecture, requiring fewer parameters and less compute than an MLP. The trade-off is flexibility — MLPs and Transformers can potentially discover things CNNs cannot, given enough data and compute. One-dimensional convolutions are noted as useful for language tasks.

The mechanics are shown through Matt Kleinsmith's visual walkthrough: a 2×2 kernel slides across a 3×3 image, producing a 2×2 output where each cell is the dot product of the kernel with the corresponding window. A top-edge kernel `[[-1,-1,-1],[0,0,0],[1,1,1]]` is applied to MNIST digits — where the top of a region is dark and the bottom is bright, a large positive value appears; the reverse gives a large negative. Pandas conditional formatting visualizes actual pixel values and the resulting activations. A left-edge kernel (the transpose) highlights vertical edges. The Zeiler and Fergus layer-1 visualizations from Lesson 1 are recalled: early conv layers learn edges and gradients, and stacking convolutions with nonlinear activations builds up to curves, corners, and complex features.

The naive Python implementation is extremely slow, so the lesson introduces **im2col** — the trick Yangqing Jia used in Caffe to convert convolutions into matrix multiplications by unrolling each input patch into a row. PyTorch's `F.unfold` does this natively: for a 3×3 kernel on a 28×28 image it produces a 9×676 matrix, and a matrix multiply with the flattened kernel yields the convolution output — roughly 400× faster than the Python loop. `F.conv2d` achieves similar speed. The unfold trick is flagged as especially useful for non-standard convolution configurations.

**Padding** is introduced to recover the lost border pixels: for a kernel of size *k*, padding of *k//2* preserves spatial dimensions (odd kernels only — even-sized kernels are rarely used). **Stride** controls how far the kernel moves each step; stride-2 convolutions halve spatial dimensions, which is the standard building block for classification and autoencoder architectures.

A simple CNN is built as a Sequential of stride-2 convolutions: 28×28×1 → 14×14×4 → 7×7×8 → 4×4×16 → 2×2×32 → 1×1×10 → Flatten. Comments track the grid size at each layer. The final layer omits the activation and outputs 10 channels (one per class). Data is reshaped, moved to the appropriate device (MPS or CUDA) via a `to_device` helper, and trained with a custom collate function. The CNN achieves comparable accuracy to the MLP but with only ~5,000 parameters versus ~40,000 — a concrete demonstration of the power of encoding structural priors. The axis convention NCHW (batch, channel, height, width) is noted, contrasted with TensorFlow's NHWC. A student question clarifies that this particular CNN only works for inputs that reduce exactly to 1×1.

**Receptive fields** are explored in Excel. A `conv-example.xlsx` spreadsheet implements convolutions using `SUMPRODUCT` with array formulas. Two convolutional layers with two filters each are shown, followed by max-pooling (taking the max of each 2×2 region) to halve dimensions. Excel's "Trace Precedents" visually reveals the receptive field of a given unit — the region of the input that can influence it. The receptive field doubles with each layer, and central pixels contribute more than peripheral ones. **Global average pooling** is introduced as the modern alternative to a dense classifier layer — simply averaging all activations in the final feature map.

The lesson then switches to **Fashion MNIST** (loaded via Hugging Face) and attempts to build a convolutional autoencoder. The encoder uses stride-2 convolutions to compress 28×28×1 down to 4×4×8 (8× compression); the decoder uses **nearest-neighbor upsampling** followed by stride-1 convolutions to expand back. The input is zero-padded to 32×32 (since 28 doesn't halve cleanly three times) and cropped back afterward with negative padding. A sigmoid forces outputs to [0,1], and MSE loss compares the reconstruction to the input itself. The results are disappointing — reconstructions are blobby approximations at best, despite careful learning rate tuning and switching to Adam.

Two practical problems are surfaced. First, training is painfully slow because the Hugging Face dataset stores images as individual PNGs — `htop` shows one CPU at 100% while the GPU sits idle at 1%. Adding `num_workers` to the DataLoader immediately breaks because the collate function puts tensors on the GPU, which is incompatible with multi-process loading. Second, classification accuracy is only 87%, well below the 92–96% range on paperswithcode. Both problems demand rewriting the training loop — motivating the construction of a proper framework.

A **Learner** class is introduced. `DataLoaders` wraps train and validation loaders with a `from_dd` classmethod. The Learner stores model, DataLoaders, loss function, learning rate, and optimizer function via `fc.store_attr()`. Its `fit` method loops epochs calling `one_epoch`, which loops batches calling `one_batch` (forward pass, loss, backward, optimizer step). With `to_device` removed from collate, `num_workers` now works and all CPUs light up.

To make the Learner flexible, a **Metric** base class is created with `add()` to accumulate mini-batch results and a `.value` property for the weighted average. `Accuracy` subclasses it, overriding `calc()`. Then a **callback system** is added: the `@with_cbs` decorator wraps key methods so that `before_*` and `after_*` hooks are called on each registered callback, with `CancelFitException` for graceful early stopping. `callback()` iterates through callbacks sorted by order, using `getattr` with `identity` as the default no-op. A `DeviceCB` callback handles moving the model to the device, a `MetricsCB` tracks and logs metrics, and a `ProgressCB` adds a progress bar with a live loss plot.

The lesson closes with a frank discussion of cognitive load. If decorators, `getattr`, properties, subclassing, or the debugger are unfamiliar, the framework code will feel overwhelming — not because the framework is complex, but because too many new Python concepts are competing for attention. The remedy is to practice each piece in isolation first. These are powerful, general software engineering patterns that will serve well beyond deep learning.

---

**Lesson Challenges**

- Implement im2col from scratch in PyTorch using tensor manipulation (repeats, tiles, reshapes)
- Create a PyTorch version of the NumPy im2col implementation linked in the notebook

**Potential Research Directions**

- Variational Autoencoders (VAEs) for better reconstruction quality
- Transposed convolutions vs nearest-neighbor upsampling — trade-offs and checkerboard artifacts
- Alternative autoencoder loss functions beyond MSE
- The 2006 paper on unrolled convolutions as matrix multiplications ("High Performance Convolutional Neural Networks for Document Processing" — Chellapilla, Puri, Simard)
- Matt Kleinsmith's ["CNNs from Different Viewpoints"](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)
- Zeiler and Fergus network visualizations
- Cognitive load theory applied to technical education

**Homework**

- Practice the Python foundations: try/except, decorators, `getattr`, properties, subclassing, and using `pdb`/`breakpoint()`
- Open the notebooks and create simplified versions of the key components
- Experiment with the Excel convolution workbook (`conv-example.xlsx`) to build intuition for receptive fields
- Get comfortable with the Learner, Metric, and callback patterns before the next lesson

**Things Jeremy Says You Should Do**

- Make sure you can pronounce all the Greek letters — they come up constantly
- Add comments noting the grid size after each convolutional layer in your architectures
- Use `o.numel()` instead of `np.product(o.shape)` to count parameters
- Practice creating really simple versions of things in the notebook to build understanding
- Get comfortable with Python foundations (try/except, decorators, getattr, debugger, properties, subclassing) to reduce cognitive load when learning the framework
- Use `breakpoint()` and step through code with the debugger

**Resources**

- [Lesson 15 official topic (forums)](https://forums.fast.ai/t/lesson-15-official-topic/102322)
- [Excel workbooks (conv-example.xlsx)](https://github.com/fastai/course22/tree/master/xl)
- [Notebook: 05_convolutions.ipynb](https://github.com/fastai/course22p2/blob/master/nbs/05_convolutions.ipynb)
- [Notebook: 06_autoencoders.ipynb](https://github.com/fastai/course22p2/blob/master/nbs/06_autoencoders.ipynb)
- [Notebook: 07_learner.ipynb](https://github.com/fastai/course22p2/blob/master/nbs/07_learner.ipynb)
- [Matt Kleinsmith — "CNNs from Different Viewpoints"](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)
- [NumPy im2col implementation](https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py)
- [Chellapilla, Puri, Simard (2006) — "High Performance Convolutional Neural Networks for Document Processing"](https://hal.inria.fr/inria-00112631)
- [Papers With Code — Fashion MNIST](https://paperswithcode.com/sota/image-classification-on-fashion-mnist)
- [miniai library (course repo)](https://github.com/fastai/course22p2)
- [Caffe deep learning library](https://caffe.berkeleyvision.org/)
