The lesson opens on familiar ground — the simple, hardcoded Learner we've seen before, with its `fit`, `one_epoch`, and `one_batch` methods, its hardcoded accuracy and loss calculations, and its single learning rate. It works, but it's inflexible. The goal now is to take this step by step toward something much more powerful, starting with an intermediate version: the Basic Callbacks Learner.

The Basic Callbacks Learner looks structurally similar to the original, but threads in a callback system. At key moments — before fitting, before and after each epoch, before and after each batch — the Learner calls `self.callback(method_name)`, which in turn calls `run_cbs`. This function iterates through a list of callback objects, sorted by their `order` attribute, and invokes the named method on each one (if it exists). A base `Callback` class defines `order = 0`. A tiny `CompletionCB` is demonstrated first: it sets a counter to zero before fit, increments it each batch, and prints the total after fit. Jeremy manually walks through `getattr` on the callback to show there's no magic — just attribute lookup and method calls. He urges viewers not to run code blindly but to experiment with it, create simple examples, and build understanding interactively.

Three custom exception classes — `CancelFitException`, `CancelEpochException`, and `CancelBatchException` — enable a powerful pattern: exceptions as control flow. Each `try`/`except` wraps a before/after callback pair. Any callback can raise one of these exceptions to cleanly abort the current batch, epoch, or entire training run. A `SingleBatchCB` demonstrates this: raising `CancelFitException` after one batch stops training entirely. The `order` attribute controls which callbacks fire first, which matters when cancellation interacts with completion reporting.

A `Metric` class is built to track weighted averages across mini-batches. `Accuracy` subclasses it with a `calc` method comparing predictions to targets. The base class stores per-batch values and counts, and its `value` property computes the weighted mean — demonstrated with a manual loss example showing how batch sizes affect the average. The `@property` decorator is noted: it lets you access a computed result without parentheses.

A `DeviceCB` callback handles GPU placement: `before_fit` moves the model to the default device, `before_batch` moves each batch. This works because the Learner stores everything on `self`, making it modifiable by callbacks. Jeremy notes this avoids the DataLoader multi-process conflicts that plagued earlier approaches.

The `MetricsCB` callback ties it together. It accepts named metric objects (like `accuracy=MultiClassAccuracy()`) plus an automatic `loss` metric using weighted mean. Before each epoch it resets all metrics; after each batch it updates them with CPU-detached predictions and targets; after each epoch it computes and logs them. A `to_cpu` helper handles detaching tensors and moving them off GPU, handling dicts, lists, and tuples. The `_log` method is deliberately separated so subclasses can redirect output to progress bars, Weights & Biases, or anything else.

The Flexible Learner then compresses the same logic into a single screen of code using `@contextmanager`. The `callback_ctx` method wraps the before/after/exception pattern: everything before `yield` runs first, then the body executes at `yield`, then cleanup runs after. The `with self.callback_ctx('fit'):` statement replaces the verbose try/except blocks, and the exception class is looked up dynamically from `globals()` using `f'Cancel{nm.title()}Exception'`. This refactoring eliminates triple-duplicated code and makes adding new callback points trivial.

The most striking design choice: `one_batch` calls `self.predict()`, `self.get_loss()`, `self.backward()`, `self.step()`, and `self.zero_grad()` — but none of these methods exist on the Learner. Instead, `__getattr__` intercepts these five names and routes them to `self.callback(name)`. A `TrainCB` callback defines all five, accessing the Learner via `self.learn`. This means the training logic itself is pluggable — you could swap in a HuggingFace-compatible version or anything else without touching the Learner.

A `ProgressCB` using **fastprogress** wraps the epoch range in a `master_bar` and the DataLoader in a progress bar, optionally plotting the training loss in real time. It hooks into the `MetricsCB` by replacing its `_log` method. The result: a clean, updating display of metrics and a live loss plot.

Jeremy then shows that the same five methods can alternatively be defined by subclassing the Learner directly — `__getattr__` is never called because the methods now exist. He uses this to demonstrate a `MomentumLearner` where `zero_grad` is replaced with gradient scaling: instead of zeroing gradients, multiply them by 0.85. Since PyTorch accumulates gradients, this creates an exponentially weighted moving average of gradients — effectively momentum — stored directly in the `.grad` tensors themselves, avoiding the usual overhead of a separate momentum buffer. The marble-on-a-contour-map analogy illustrates why this reduces zig-zagging and speeds convergence. Results improve from ~0.7 to ~0.8 accuracy at the same learning rate.

The **learning rate finder** callback implements Leslie Smith's idea: gradually increase the learning rate each batch (by a multiplicative factor, e.g., 1.3×), track the loss, and stop when it exceeds 3× the minimum observed loss. The resulting plot of loss vs. learning rate (on a log scale) reveals the steepest descent region — the ideal learning rate neighborhood. A variant using PyTorch's `ExponentialLR` scheduler shows that schedulers are just thin wrappers around manual learning rate updates — nothing magical. A suggestion to rename the `plot` method to `after_fit` leads to the observation that `CancelFitException` skips `after_fit`, motivating the use of a `finally` block to ensure cleanup always runs.

The lesson then shifts to **notebook 10** and the critical question: how do we know what's happening inside a model during training? Most practitioners randomly try things when training fails. Instead, we're going to look inside. A `set_seed` function sets all three RNGs (PyTorch, NumPy, Python) and enables deterministic algorithms for reproducibility.

Training a CNN on Fashion-MNIST with a high learning rate of 0.6 shows the loss rising, crashing, and ultimately failing. To understand why, a custom `SequentialModel` records the mean and standard deviation of each layer's activations at every batch. Plotting these reveals a characteristic pathology: activations start near zero, spike up exponentially, crash, spike higher, crash lower — a devastating cycle. When the standard deviation is near zero, all activations are essentially the same value (close to zero), meaning the layer is doing no useful work. Jeremy emphasizes: if you haven't looked at these plots, you don't know whether your model is training well. You want means near zero and standard deviations near one throughout training.

**PyTorch hooks** provide the same diagnostic capability without modifying the model. A forward hook registered via `m.register_forward_hook(fn)` receives the module, its input, and its output each time the layer executes. Using `partial` to pass the layer index, we collect the same activation statistics from a standard `nn.Sequential`. Jeremy clarifies: hooks and callbacks are the same concept — code called for you at specified moments. PyTorch just uses the name "hooks."

The `Hook` class wraps this pattern: the constructor registers the hook, `remove` and `__del__` clean it up. A `Hooks` class (plural) inherits from `list`, implements `__enter__`/`__exit__` for use as a context manager, and enables clean one-liner usage: `with Hooks(model, append_stats) as hooks: fit(model)`. Jeremy demonstrates Python dunder methods (`__delitem__`, `__del__`, `__enter__`, `__exit__`) with simple `DummyCtxMgr` and `DummyList` examples, and strongly recommends viewers create their own dummy versions of any unfamiliar Python feature.

The climax is the **"colorful dimension" histogram plot**. Instead of just tracking means and standard deviations, we record a 50-bin histogram of absolute activation values at each batch. Each histogram is compressed into a single column of pixels — bright yellow for high counts, dark blue for low, darkest for zero — and all columns are stacked left to right across batches. The result is a single image per layer that reveals the full distribution of activations over time. Taking `log1p` of counts improves visibility. The pathological pattern is immediately visible: a thick bright bar at the bottom (most activations near zero) with occasional spikes. A complementary plot shows the fraction of activations in the bottom two bins — effectively dead units. For the final layer, nearly all activations are dead from early on. Jeremy states plainly: if you see the rising-crash pattern early in training, stop and restart, because recovery is unlikely.

The lesson concludes by noting that we now have all the key pieces — a flexible training framework and tools to see inside models. From here, we start building back up toward reliable, fast training and ultimately high-quality generative models. Next lesson: **initialization**, which requires comfort with standard deviations.

---

**Lesson Challenges**

- Build a custom `SequentialModel` that records activation means and standard deviations, then plot them
- Create a `SingleBatchCB` that cancels training after one batch, and experiment with the `order` attribute
- Implement the learning rate finder callback and interpret the resulting plot

**Potential Research Directions**

- Memory-efficient momentum via gradient scaling instead of separate momentum buffers
- Better activation histogram diagnostics — automated detection of pathological training patterns
- [Cyclical Learning Rates for Training Neural Networks — Leslie Smith](https://arxiv.org/abs/1506.01186)
- [A Disciplined Approach to Neural Network Hyper-parameters — Leslie Smith](https://arxiv.org/abs/1803.09820)
- [Methods for Automating Learning Rate Finders — Zach Mueller](https://www.novetta.com/2021/03/learning-rate/)
- Using exceptions as control flow in training frameworks — tradeoffs and alternatives

**Homework**

- For every Python feature used in the lesson that you're not 100% comfortable with (context managers, `__getattr__`, `__enter__`/`__exit__`, `__delitem__`, `@property`, `partial`, inheriting from `list`), create a simple dummy version from scratch that fully explores how it works
- If you're already comfortable with the Python pieces, do the same for the PyTorch pieces (hooks, `register_forward_hook`, schedulers)
- Review standard deviations and related statistics in preparation for the next lesson on initialization

**Things Jeremy Says You Should Do**

- Don't run code willy-nilly — experiment with it, understand it, create simple examples
- If Jeremy hasn't provided a simple example to make something easy to understand, create one yourself
- Don't just use what someone else has already created — build your own understanding
- Always look at activation mean/std plots and histograms before concluding a model is training well
- If you see the rising-crash pattern early in training, stop and restart — the model will probably never recover
- Train as fast as possible: higher learning rates find more generalizable weights and reduce overfitting

**Resources**

- [Lesson 16 Official Topic (Forums)](https://forums.fast.ai/t/lesson-16-official-topic/102472)
- Notebook: `09_learner.ipynb`
- Notebook: `10_activations.ipynb`
- Library: [torcheval](https://pytorch.org/torcheval/) (official PyTorch metrics)
- Library: [fastprogress](https://github.com/fastai/fastprogress) (progress bars)
- [Cyclical Learning Rates for Training Neural Networks — Leslie Smith](https://arxiv.org/abs/1506.01186)
- [A Disciplined Approach to Neural Network Hyper-parameters — Leslie Smith](https://arxiv.org/abs/1803.09820)
- [Methods for Automating Learning Rate Finders — Zach Mueller](https://www.novetta.com/2021/03/learning-rate/)
- [Lesson 16 Course Page](https://course.fast.ai/Lessons/lesson16.html)
