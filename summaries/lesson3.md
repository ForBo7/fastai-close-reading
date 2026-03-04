Lesson three shifts gear from the practical, deployment-focused first two lessons into the mathematical foundations of deep learning. Jeremy opens with a quick survey showing the class pace is about right, then points viewers to Lesson Zero — based heavily on Radek's book *Meta Learning* — which covers the science of learning itself and how to get the most out of fast.ai. He outlines his recommended study workflow: watch the whole lecture once without stopping, then rewatch with pauses while running the notebook. The "clean" versions of the book chapters (headings only, no outputs) are highlighted as an excellent self-test tool. Study groups — in-person or virtual — are strongly encouraged.

A showcase of student work follows: a Marvel character detector, a rock-paper-scissors game, an Elon detector, a temperature predictor from aerial photos, a cloud detector by a real meteorologist, a dinosaur classifier, a choose-your-own-adventure driven by facial expressions, a music genre classifier, a Microsoft Power App, an art movement classifier, and a redaction detector. Jeremy's tone is one of genuine delight — these are the kinds of creative projects the course aims to inspire.

Jeremy then demonstrates his pet breed classifier on HuggingFace Spaces (37 breeds, not just cat vs. dog), and introduces Paperspace Gradient Notebooks as the recommended platform — real persistent storage, JupyterLab, free GPUs, and an affordable paid tier. He walks through the two-piece workflow: training produces a `model.pkl`, deployment feeds it inputs. The training notebook uses `ImageDataLoaders`, `show_batch`, and a ResNet34, achieving ~7% error.

The lesson pivots to model selection. Using a notebook benchmarking 500+ architectures from the PyTorch Image Models (`timm`) library, Jeremy shows a scatterplot of speed vs. accuracy on ImageNet. ResNet18 is small and fast (ideal for prototyping); ResNet34 is solid but no longer state-of-the-art. The ConvNeXT family stands out — swapping to `convnext_tiny_in22k` drops the error rate from 7.2% to 5.5% with only a modest speed penalty. The `timm.list_models` function returns architecture names as strings, which can be passed directly to `vision_learner`. Jeremy stresses: trying better architectures should be the *very last* step of a project, not the first.

The `predict` method returns 37 probabilities, one per breed. The ordering is stored in the `vocab` attribute of the DataLoaders. The `model.pkl` file is a Learner containing both preprocessing steps and the trained model. Drilling into the model with `get_submodule` reveals layers upon layers — each containing parameters: just numbers. Where do they come from?

To answer this, Jeremy opens the "How does a neural net really work?" Kaggle notebook. The core idea: machine learning fits functions to data. Starting with a simple quadratic $f(x) = ax^2 + bx + c$, he shows how Python's `partial` function fixes coefficients to create specific quadratics. Using Jupyter's `@interact` decorator, he creates sliders for $a$, $b$, and $c$, and demonstrates fitting the curve to noisy data by hand — adjusting one slider at a time, checking visually whether the fit improves. This manual process is then formalized with a **loss function** (mean squared error), which gives a single number measuring how wrong the predictions are.

The automation comes through **gradient descent**. PyTorch's `requires_grad_()` flags parameters for automatic differentiation; calling `.backward()` on the loss populates `.grad` attributes showing how much each parameter should change. The update rule is simple: subtract the gradient times a small number (the **learning rate**). Wrapped in `torch.no_grad()` to avoid tracking this arithmetic, this loop — calculate loss, backward, update — is the beating heart of all neural network training.

Jeremy draws on a whiteboard to explain why we can't just use the raw gradient values: the loss surface is curved, so a steep slope doesn't mean you should jump far. Too large a learning rate causes divergence (oscillating past the minimum); too small means painfully slow convergence. The learning rate is a **hyperparameter** — a parameter used to find parameters.

Then comes the key insight: quadratics aren't flexible enough. Enter the **Rectified Linear Unit** (ReLU): $\max(0, mx + b)$. A single ReLU is just a line clipped at zero. But add two ReLUs with different slopes and offsets, and you get a bent line with a kink. Add more and you get more kinks. Add enough, and you can approximate *any* function to arbitrary precision, in any number of dimensions. Combined with gradient descent to find the parameters, this is — genuinely — the entirety of deep learning. Jeremy invokes the "how to draw an owl" meme: step one, draw two circles; step two, draw the rest of the owl. In deep learning, there truly is nothing between these two steps.

The lesson culminates in building a complete model in an Excel spreadsheet using the Kaggle Titanic dataset. Jeremy walks through data preparation: converting categorical variables (sex, embarkation port, passenger class) into binary dummy variables ($n-1$ columns for $n$ categories), normalizing continuous features (dividing by the max, using log for skewed fare data), and adding a column of ones for the bias term. A linear regression is built with SUMPRODUCT and optimized with Excel's Solver (a built-in gradient descent optimizer), achieving a loss of ~0.10. Then he upgrades to a neural network by adding a second set of coefficients, applying ReLU (`IF(cell<0, 0, cell)`), and summing the results — achieving a loss of ~0.08. A final version replaces SUMPRODUCT with MMULT (matrix multiplication), showing that matrix multiplication is simply the compact notation for "multiply things together and add them up" — the critical operation in all of deep learning, executed on GPUs via specialized tensor cores.

The lesson closes with a preview of Lesson 4: natural language processing using HuggingFace Transformers (chosen deliberately for its lower-level API), working on a Kaggle patent phrase-matching competition. Jeremy notes that underrepresented languages are an *opportunity*, not a barrier, and previews the important topics of validation sets and metrics.

---

**Lesson Challenges**
- Reproduce the quadratic-fitting and ReLU demonstrations from the ["How does a neural net really work?"](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work) notebook
- Recreate the Titanic neural network in a spreadsheet (Excel, Numbers, or Google Sheets) — or from scratch in Python
- Try swapping architectures using `timm.list_models` and compare results on your own dataset

**Potential Research Directions**
- The ConvNeXT architecture family and why it revitalized pure convolutional approaches after the Transformer wave
- The Universal Approximation Theorem — the formal proof that sums of ReLUs can approximate any continuous function
- Semi-supervised learning techniques for leveraging unlabeled data
- Learning rate scheduling and automatic learning rate finding (covered later in the course)
- Why log-transforming skewed features (like fare) improves model performance

**Homework**
- Look at the ["Getting Started with NLP for Absolute Beginners"](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) notebook and the U.S. Patent Phrase Matching competition data before Lesson 4
- Work through Chapters 1–2 of the book as notebooks; try reproducing results using the "clean" versions, then repeat with a different dataset
- Watch Lesson Zero if you haven't already

**Things Jeremy Says You Should Do**
- Watch each lecture all the way through once, then rewatch with pauses while running the notebook
- Use the "clean" notebook versions to test your understanding — predict each cell's output before running it
- Join or create a study group (forums or Discord)
- Search the forum and check FAQs before asking questions
- Train your first model on day one of any project — don't wait for perfect data
- Use ResNet18 for prototyping; try fancier architectures only as the very last step
- Spend your time on data augmentation, data cleaning, and external data — not architecture hunting
- Watch Lesson Zero, especially the parts on the science of learning

**Resources**
- [Course page for Lesson 3](https://course.fast.ai/Lessons/lesson3.html)
- [Lesson Zero video](https://www.youtube.com/watch?v=gGxe2mN3kAg)
- [Chapter 4 of the fastbook](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb)
- [HuggingFace Spaces Pets repository](https://huggingface.co/spaces/jph00/pets/tree/main)
- [Which image models are best? (Kaggle notebook)](https://www.kaggle.com/code/jhoward/which-image-models-are-best/)
- [How does a neural net really work? (Kaggle notebook)](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work)
- [Getting started with NLP for absolute beginners (Kaggle notebook)](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners)
- [Titanic spreadsheet (course repository)](https://github.com/fastai/course22)
- [Titanic competition data (Kaggle)](https://www.kaggle.com/competitions/titanic/)
- [Chapter 4 questionnaire solutions](https://forums.fast.ai/t/fastbook-chapter-4-questionnaire-solutions-wiki/67253)
- [Matrix multiplication visualization](https://matrixmultiplication.xyz)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)
- [Know Your Pet](https://gettoknowyourpet.com/)
- Radek's book: *Meta Learning*
