# Lesson 12: Mean Shift Clustering — Summary

The lesson opens with a detour into the **CLIP Interrogator**, a Hugging Face Spaces Gradio app that has been generating buzz online. Jeremy uploads his own photo and gets back a prompt full of oddities — "extremely long forehead," "tectonics," "without eyebrows." Many people on Twitter believe this tool returns the CLIP prompt that would regenerate the original image. Jeremy seizes this misconception as a teaching moment. He draws a diagram: an image goes through the CLIP image encoder, producing a small embedding vector. Could you invert this — send someone the embedding and have them reconstruct the photo? No. Not every function has an inverse. A function that maps a 512×512×3 image down to a short vector throws information away irreversibly, just as `def f(x): return 0` cannot be undone. What you *can* do is run a **diffusion process**: start from noise plus the embedding and iteratively denoise, producing something that *might* have generated that embedding. This is an approximation, not an inversion. Since CLIP was trained so that image embeddings and text embeddings of matching captions land near each other, text-conditioned diffusion works the same way. These are all instances of **inverse problems** — Stable Diffusion approximates their solution. The CLIP Interrogator, under the hood, just mixes hardcoded lists of artists, mediums, and movements with BLIP captions; it is not inverting anything.

The lesson then picks up where Lesson 11 left off: **matrix multiplication**. The broadcasting version had already achieved a 5,000× speedup over pure Python loops. Now Jeremy introduces **Einstein summation** (`torch.einsum`), a compact notation where repeated index letters between inputs mean element-wise multiplication along those axes, and omitting a letter from the output means summation along that axis. The full intermediate product `'ik,kj->ikj'` on two matrices of shapes 5×784 and 784×10 produces a 5×784×10 tensor; dropping the `k` to get `'ik,kj->ij'` collapses the 784 dimension by summation, yielding the 5×10 matrix product. Einsum runs in about 15 ms — far faster than the 600 ms broadcasting version. PyTorch's built-in `@` operator is about the same speed; einsum is sometimes clearer. Jeremy notes a question about APL: Einstein notation actually *predates* APL; Ken Iverson was heavily influenced by tensor analysis, and the idea of implicit loops without indices became central to APL, J, NumPy, PyTorch, and TensorFlow alike.

Next comes **GPU acceleration**. Jeremy writes a pure-Python "kernel" — a function that computes a single element of the result matrix given its coordinates, a bounds check, and an inner loop. A `launch_kernel` helper loops over all grid positions calling the kernel. This simulates how a GPU works (many kernels running in parallel), except sequentially. To actually run on the GPU, Jeremy uses **Numba's CUDA JIT**: the `@cuda.jit` decorator compiles nearly identical Python code into GPU machine code. The only change is replacing the explicit grid argument with `cuda.grid(2)`. Tensors are copied to the GPU with `cuda.to_device`, the kernel is launched with special bracket syntax specifying blocks and threads per block, and results come back with `copy_to_host()`. The Numba CUDA matmul runs in 3.6 ms. But PyTorch's own GPU matmul (`x_train.cuda() @ weights.cuda()`) does it in **458 microseconds** — a total speedup of roughly **5 million times** over the original pure Python version. This is why GPU computation matters: not a 20% improvement, but orders-of-magnitude transformation.

The heart of the lesson is **mean shift clustering**. Unlike the supervised learning done previously, clustering is unsupervised — there is no dependent variable. The goal is to discover groups of similar items. Mean shift has advantages over k-means: it does not require specifying the number of clusters in advance (only a bandwidth parameter, which can be chosen automatically), and it handles arbitrarily shaped clusters.

Jeremy builds the algorithm from scratch. First, he creates **synthetic data**: six random 2D centroids, then 750 samples around each using `MultivariateNormal` with a diagonal covariance matrix (standard deviation 5 in both X and Y). The plot shows six colored clusters with their true centers marked as X's.

The algorithm works by gravitational attraction: for each point, calculate the distance to every other point, weight those distances through a **Gaussian kernel** (the normal distribution curve, parameterized by bandwidth σ), and replace the point with the **weighted average** of all points. Nearby points exert strong pull; distant points contribute almost nothing. Iterating this process causes points within the same cluster to collapse together onto their shared center.

Jeremy defines the Gaussian kernel in one line, then shows how to quickly plot any function with `linspace` and `plt.plot`. He introduces `functools.partial` for creating partially applied functions — `partial(gaussian, bw=2.5)` — and notes that a simple **triangular weighting** (a linear ramp with `clamp_min`) produces nearly identical clustering results.

The distance computation uses broadcasting: `x - X` where x is shape (2,) and X is (1500, 2) works because the trailing dimensions match and the missing leading dimension gets a unit axis. Jeremy pauses to explain **norms**: the L2 norm (Euclidean distance, Pythagorean theorem) and L1 norm (Manhattan distance), and how RMSE is L2 loss while MAE is L1 loss. He squares the differences, sums over the coordinate axis, and takes the square root.

Weights come from the Gaussian kernel applied to distances. The **weighted average** is computed as the sum of (weight × data) divided by the sum of weights. Jeremy emphasizes the pedagogical approach: build everything step by step, checking shapes and values at each stage, then merge the cells into a function. The `one_update` function loops through all 1500 points; `meanshift` clones the data, iterates five times, and returns. Runtime: 600 ms. The result perfectly recovers the six cluster centers. Triangular weighting produces identical results.

To visualize the convergence, Jeremy creates a **Matplotlib animation** using `FuncAnimation`. The `do_one` callback updates data, clears the axis, and replots each frame. The animation is displayed inline with `HTML(ani.to_jshtml())`.

The looped version is slow because calling GPU code 1500 times from Python incurs enormous overhead. The solution: **batch the computation**. Instead of processing one point at a time, process a mini-batch of `bs` points simultaneously. The distance calculation for a batch requires careful broadcasting: X becomes (1, 1500, 2) and the mini-batch x becomes (bs, 1, 2), producing a (bs, 1500, 2) difference tensor, which is squared and summed to get (bs, 1500) distances. The Gaussian kernel broadcasts over anything. For the weighted sum, Jeremy shows that `weight[..., None] * X[None]` works, but then recognizes that the product-then-sum pattern is exactly `einsum('ij,jk->ik', weight, X)` — which is just **matrix multiplication** (`weight @ X`). This is a satisfying discovery: the weighted average reduces to matmul.

The batched GPU version runs in **1 millisecond** — compared to 400 ms on CPU. Jeremy notes that people write academic papers about GPU mean shift, yet here it is done straightforwardly in PyTorch.

The lesson closes with a brief **calculus refresher**, borrowing 3Blue1Brown's car-distance example. Distance as a function of time gives a straight line; the slope (rise over run) gives velocity — a change of dimension from meters to meters-per-second. For curves, the slope at a point is found by taking the ratio of tiny changes: (f(t+d) − f(t))/d as d gets very small. Jeremy advocates the **calculus of infinitesimals** perspective: treat dy and dx as very small real numbers. The chain rule dy/dx = (dy/du)(du/dx) is then just cancellation. He reassures the viewer: PyTorch will compute all derivatives automatically, so the only rule you truly need is the chain rule, which will be covered next lesson. He recommends 3Blue1Brown's "Essence of Calculus" series for anyone who needs a refresher.

---

## Lesson Challenges

- Rewrite the Euclidean distance calculation using `torch.einsum()` (you won't eliminate `x - X` or the sqrt, but the multiply-then-sum can be a single einsum)

## Potential Research Directions

- Invent a new mean shift variant using LSH or k-d trees for fast approximate nearest neighbors to avoid quadratic time complexity
- Publish a paper describing a novel fast mean shift algorithm (Jeremy's "super super bonus")
- Explore GPU-accelerated implementations of other clustering algorithms (DBSCAN, k-means, spectral clustering, LSH)
- Inverse problems and how diffusion models approximate their solutions

## Homework

- GPU-accelerate one of these clustering algorithms: DBSCAN, k-means, spectral clustering, or LSH
- Super bonus: invent a new mean shift algorithm using LSH or k-d trees for fast nearest neighbors
- Super super bonus: publish a paper describing it
- Watch [3Blue1Brown's "Essence of Calculus"](https://www.youtube.com/watch?v=WUvTyaaNkzM) series before the next lesson if you're not comfortable with derivatives
- Study NumPy broadcasting rules thoroughly if you haven't already

## Things Jeremy Says You Should Do

- Practice tensor manipulation (broadcasting, einsum, matrix ops) until it becomes second nature — it's like learning your times tables
- Build algorithms by first creating synthetic data with known behavior
- Develop code step by step: write it out piece by piece checking shapes and values, then merge into a function
- Use `functools.partial` for partial function application — it's a very important tool
- Have a utility for quickly plotting any function (`linspace` + `plt.plot`)
- Don't scroll through imports to find what something is — just type it and press Shift+Enter; use `?` for help and `??` for source
- Set up keyboard shortcuts: "run all cells above" and "run all cells below"
- Watch 3Blue1Brown's calculus series if you need a refresher
- Think of dy/dx as actual small numbers (infinitesimals) — it's rigorous and intuitive

## Resources

- [Lesson 12 discussion thread](https://forums.fast.ai/t/lesson-12-official-topic/101702)
- [CLIP Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator) (Hugging Face Spaces)
- [Essence of Calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM) — 3Blue1Brown
- [Lesson 12 video](https://youtu.be/_xIzPbCgutY)
- [Course page](https://course.fast.ai/Lessons/lesson12.html)
- Numba CUDA documentation
- `torch.einsum` documentation
- `functools.partial` (Python standard library)
- `matplotlib.animation.FuncAnimation`
- `torch.distributions.multivariate_normal.MultivariateNormal`
