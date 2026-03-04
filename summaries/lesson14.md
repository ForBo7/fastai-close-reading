The lesson opens with a review of backpropagation from Lesson 13, grounding the math in code. Jeremy writes out the chain rule on screen: if the loss is the composition of a loss function and a neural network, $L = l(N(w, x))$, then the derivative of the loss with respect to the weights is $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial N} \cdot \frac{\partial N}{\partial w}$. He maps this directly to the `lin_grad` function from last time — `out.g` is $\frac{\partial L}{\partial N}$, and the rest is matrix multiplication with transposes. The point is that every line of backward-pass code corresponds to a term in the chain rule. He recommends Kaushik Sinha's write-up (linked in the Lesson 13 resources), and for anyone who hasn't studied calculus, 3Blue1Brown's "Essence of Calculus" and Khan Academy — a few hours of study and it all clicks.

With backpropagation reviewed, the lesson pivots to the real theme: refactoring. The training loop from last time works — 96% accuracy on MNIST — but it's clunky. Jeremy introduces `torch.nn.Module`, showing that when you assign an `nn.Linear` as an attribute, the module automatically tracks it as a child and its tensors as parameters. He then builds an MLP class inheriting from `nn.Module` with two linear layers and a ReLU, and demonstrates that `model.parameters()` yields all four parameter tensors (two weight matrices, two bias vectors) without any manual bookkeeping. The training loop shrinks: just iterate through `model.parameters()`, subtract the gradient times learning rate, and call `model.zero_grad()`.

Then comes the real magic — rebuilding `nn.Module` from scratch. A custom `MyModule` class uses `__setattr__`, the Python dunder method called every time you set an attribute. If the attribute name doesn't start with an underscore, the value gets stashed in a `_modules` dictionary. The `parameters()` method iterates through `_modules.values()` and yields each module's parameters. Jeremy introduces `yield from` as a one-line replacement for the yield-in-a-loop pattern. Having demystified `nn.Module`, he shows `nn.ModuleList` for registering lists of layers, builds a `SequentialModel` using it (with `functools.reduce` as an alternative to the explicit forward loop), and then swaps it out for PyTorch's `nn.Sequential`.

The optimizer is next. An `Optimizer` class stores parameters and a learning rate; `step()` updates each parameter in-place, `zero_grad()` zeroes all gradients. The training loop now just calls `opt.step()` and `opt.zero_grad()`. Since PyTorch already provides `torch.optim.SGD`, the from-scratch version is retired.

Datasets and DataLoaders follow the same build-then-replace pattern. A `Dataset` class with `__init__`, `__len__`, and `__getitem__` wraps x and y tensors so you can index them together. A `DataLoader` wraps a Dataset with a batch size and yields slices in `__iter__`. A `Sampler` produces indices (optionally shuffled), and a `BatchSampler` chunks those indices into batches. The collate function — central to PyTorch data loading — takes a list of `(x, y)` tuples, transposes them with `zip()`, and stacks them with `torch.stack()`. Jeremy emphasizes how important `zip()` is and how collation is used heavily in both PyTorch and Hugging Face. He also introduces `fastcore.store_attr()` as a one-liner replacement for repetitive `self.x = x` assignments.

He briefly explains Python's `multiprocessing` and how PyTorch's DataLoader uses `torch.multiprocessing` (a reimplementation that works with tensors) and `Pool.map()` to parallelize `__getitem__` calls. A validation loop is added, doubling batch size since no gradients are needed, achieving 97% accuracy on the full validation set.

The lesson then shifts to Hugging Face Datasets, loading Fashion-MNIST — a drop-in MNIST replacement with clothing categories. `load_dataset_builder()` provides metadata; `load_dataset()` downloads and caches data. Unlike PyTorch Datasets that return tuples, Hugging Face returns dictionaries. A transform function (applied via `with_transform()`) converts PIL images to flattened tensors on-the-fly. Jeremy builds an `inplace` decorator — a function that wraps another function to automatically return its first argument — and shows that the `@inplace` decorator syntax is identical to calling the wrapper manually. He then introduces `operator.itemgetter` as a function-returning-function for extracting dictionary keys, and `default_collate` for stacking dictionaries of tensors. These combine into `collate_dict()`, bridging Hugging Face's dict-based data with PyTorch's tuple-based expectations.

Image plotting gets its own treatment: `show_image()` wraps `plt.imshow()` with axis cleanup, CPU/NumPy conversion, and `@delegates` for automatic kwarg documentation. `get_grid()` and `show_images()` build grids of labeled images. Everything gets exported via nbdev's `#|export` directive into a `miniai` library — the from-scratch framework that will grow throughout the rest of the course.

The lesson's final major section introduces callbacks. Starting with a simple `slow_calculation` that accepts an optional `cb` parameter, Jeremy shows callbacks as plain functions, lambdas, `functools.partial` applications, callable classes, and finally objects with named methods like `before_calc` and `after_calc`. This progression from simple function callbacks to method-based callback objects sets up the Learner framework coming in the next lesson.

Throughout, Jeremy weaves in Python foundations: `*args` and `**kwargs` (both in definitions and calls), f-string format specifiers, generators, and a tour of eleven key dunder methods (`__repr__`, `__init__`, `__add__`, `__getitem__`, `__setattr__`, `__getattr__`, `__call__`, `__len__`, `__iter__`, `__enter__`/`__exit__`). He shows `__getattr__` as a hook that fires only for undefined attributes — a pattern used heavily in fastai and Hugging Face. He closes by encouraging everyone to study unfamiliar parts, use `pdb.set_trace()` to step through code, set up keyboard shortcuts for "Run All Above," and ask for help on the forums.

---

- **Lesson Challenges**
  - Rebuild `nn.Module` from scratch using `__setattr__`
  - Implement a custom SGD optimizer
  - Build Dataset, DataLoader, Sampler, BatchSampler, and collate function from scratch
  - Create `collate_dict` to bridge Hugging Face dict-style data with PyTorch tuple-style
  - Build an `inplace` decorator

- **Potential Research Directions**
  - Custom collation strategies for non-standard data (graphs, variable-length sequences, nested dicts)
  - Extending the callback pattern toward a full Learner framework with flexible event hooks
  - Multiprocessing vs multithreading trade-offs in data loading for different workloads (image augmentation, tokenization, etc.)
  - Designing domain-specific Dataset/DataLoader abstractions beyond image classification

- **Homework**
  - Use `import pdb; pdb.set_trace()` to step through the inner workings of functions like `default_collate`, the DataLoader, and the training loop
  - Study Python's [Data Model documentation](https://docs.python.org/3/reference/datamodel.html) to understand the eleven dunder methods discussed
  - Review 3Blue1Brown's ["Essence of Calculus"](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) and Khan Academy if the chain rule and derivatives are unfamiliar
  - Review Kaushik Sinha's backpropagation explanation (linked in the Lesson 13 resources)

- **Things Jeremy Says You Should Do**
  - Set up keyboard shortcuts for frequently used actions like "Run All Above" (e.g., QA for above, QB for below)
  - Use `pdb.set_trace()` liberally — put breakpoints and step through, especially inside inner functions
  - Always make sure your starting code works before refactoring; keep the steps visible so readers can follow the reasoning
  - Look up Python f-string format specifiers if you haven't used them
  - Spend a few hours studying any unfamiliar parts — nothing is inherently harder, it's just a matter of background
  - Ask on the forums if you get lost — people are really keen to help

- **Resources**
  - [Lesson 14 Official Topic (Forums)](https://forums.fast.ai/t/lesson-14-official-topic/102018)
  - [Lesson 14 Video](https://youtu.be/veqj0DsZSXU)
  - [Course page](https://course.fast.ai/Lessons/lesson14.html)
  - Notebooks: `05_datasets` and `05a_foundations` from the [course22p2 repo](https://github.com/fastai/course22p2)
  - [Kaushik Sinha's backpropagation explanation](https://forums.fast.ai/t/lesson-13-official-topic/101990) (linked in Lesson 13 resources)
  - [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
  - [Khan Academy — Calculus](https://www.khanacademy.org/math/calculus-1)
  - [Python Data Model documentation](https://docs.python.org/3/reference/datamodel.html)
  - [Hugging Face Datasets library](https://huggingface.co/docs/datasets/)
  - [Fashion-MNIST on Hugging Face](https://huggingface.co/datasets/fashion_mnist)
  - [nbdev](https://nbdev.fast.ai/)
  - [fastcore](https://fastcore.fast.ai/)
  - [PyTorch nn.Module documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
