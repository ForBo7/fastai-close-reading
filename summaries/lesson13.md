# Lesson 13: Backpropagation & MLP

The lesson opens with Jeremy celebrating student successes from the previous week — notably a fast mean shift algorithm using random sampling of data points, which he flags as a genuinely powerful technique (random sampling and random projections are two excellent ways of speeding up algorithms). He also highlights strong Einstein summation implementations and DiffEdit work, encouraging MOOC viewers to share their own results on the forums.

The lesson then turns to notebook three in the course22p2 repo: implementing the forward and backward passes of a simple multi-layer perceptron from scratch. But first, a quick conceptual review. Jeremy draws a single-pixel example on a whiteboard: one pixel value as x, probability of being a "3" as y, fitted with a simple line. The problem is obvious — a line can't capture a curve. And adding more lines doesn't help, because the sum of lines is still a line. The trick is *rectified* lines: take a line and clamp everything below zero to zero, creating two segments. Adding the original line to this rectified version changes the slope on one side. Add another rectified line further out and the slope changes again. With enough of these rectified lines summed together, you can approximate any arbitrary shape. This is the geometric intuition behind ReLU. For multiple pixels, the same idea extends to rectified planes in higher dimensions.

With that foundation in place, the MLP is built. The MNIST dataset gives 50,000 images of 784 pixels each, with 10 possible digit classes. Jeremy chooses 50 hidden activations (`nh = 50`). A matrix multiply of the inputs (50,000 × 784) by a weight matrix (784 × 50) creates 50 linear combinations per image. These go through ReLU (clamp at zero), then through a second linear layer (50 × 1) to produce a single output per image. The single output is a deliberate simplification — rather than 10 class probabilities with cross-entropy, Jeremy uses MSE against the raw digit label (acknowledging this is a terrible loss function, since it treats 9 as "further" from 2 than 4 is). The weights start as random values; biases start at zero.

A broadcasting pitfall surfaces immediately when computing MSE. The result tensor is (10,000 × 1) and the target is (10,000). Subtracting these produces a 10,000 × 10,000 matrix — 100 million points instead of 10,000. Jeremy walks through the broadcasting rules right-to-left to show exactly why, then demonstrates the fix: either index with `[:, 0]` or call `.squeeze()` to remove the trailing unit dimension.

The lesson shifts to gradients. Jeremy draws the slope-as-rise-over-run picture, connects it to speed (distance over time), then quickly arrives at derivatives and infinitesimals. The key question: if the loss is a function of the weights, what is ∂loss/∂w₀? If increasing a weight decreases the loss, increase it; if it increases the loss, decrease it. Multiply by a learning rate and subtract — that's SGD. But with multiple inputs and outputs, derivatives become matrices: a Jacobian, where each cell says "if I nudge this input, how much does this output change." Jeremy references the paper he co-authored with Terence Parr, *The Matrix Calculus You Need For Deep Learning*, as the go-to resource for understanding this.

The chain rule is the core mechanism. Jeremy uses an interactive GeoGebra animation of connected wheels: if the x-wheel is twice the circumference of the u-wheel, then u spins twice per rotation of x (du/dx = 2). Connect u to a y-wheel that's half its size, and y spins twice per u rotation (dy/du = 2). The compound effect: dy/dx = 4. This is exactly what happens when computing gradients through stacked layers — you multiply the derivatives at each stage.

For the actual backward pass code, Jeremy writes gradient functions for MSE (derivative of diff² is 2·diff, divided by n for the mean), for a linear layer (input gradients are the weights times the output gradient; weight gradients are input transposed times output gradient; bias gradients are the output gradient summed), and for ReLU (gradient is 1 where input > 0, else 0, times the output gradient). He then demonstrates the Python debugger (`import pdb; pdb.set_trace()`) as the way to interactively explore these calculations inside functions — printing shapes, stepping through lines, inspecting intermediate values. He strongly advocates learning `pdb` as one of the most powerful tools for improving your coding.

While debugging, Jeremy discovers that the unsqueeze-multiply-sum pattern in the linear gradient can be expressed as an Einstein summation (`einsum('ij,ik->jk', inp, out.g)`), which further simplifies to `inp.T @ out.g`. The computed gradients are verified against PyTorch's autograd — they match.

The code is then refactored in stages. First, each operation becomes a class with `__call__` (so instances act like functions) that stores its inputs and outputs during the forward pass and uses them during backward. Then a base `Module` class extracts the repeated boilerplate (storing inputs, calling forward, returning output). The Relu, Lin, and Mse classes become dramatically simpler — each just implements `forward` and `bwd`. The Model class holds a list of layers, runs them in sequence for forward, and reverses through them for backward. This is backpropagation: just the chain rule, applied layer by layer in reverse, with a computational trick of caching intermediate values.

Having reimplemented everything, Jeremy invokes the course rule: once you've built it from scratch, you're allowed to use the library version. PyTorch's `nn.Module` works identically — you define `forward` and PyTorch handles backward automatically because it knows the derivatives of all its operations and how to chain them.

The lesson then improves the loss function. Using the Part 1 Excel entropy workbook as reference, Jeremy explains softmax (exponentiate each output, divide by the sum), then log softmax. Key log/exponent identities are reviewed: log(a/b) = log(a) − log(b), and log(eˣ) = x. These simplify log softmax to x − log(Σeˣ). But computing Σeˣ directly is numerically dangerous — large exponents destroy floating point precision. The **log-sum-exp trick** fixes this: subtract the maximum value before exponentiating, then add it back after the log. This is a *complexification* that makes the floating point unit's life much easier. PyTorch provides `torch.logsumexp()`.

For cross-entropy, since targets are integers (not one-hot), the loss reduces to simply indexing into the log softmax predictions at the correct class position and negating. Jeremy demonstrates this with fancy indexing: `sm_pred[range(n), target]`. PyTorch calls this `nll_loss`, and the combination of log softmax + nll_loss is `F.cross_entropy()`.

Finally, a training loop. Batch size 64, learning rate 0.5, three epochs. For each batch: compute predictions, loss, backward pass, then update weights (w -= lr × w.grad) and zero gradients. Starting from ~10% accuracy (random weights), the model reaches **97% accuracy** in just three epochs.

---

## Lesson Challenges

- Fastest mean shift algorithm competition (ongoing from previous lesson) — can anyone beat random sampling?

## Potential Research Directions

- Random sampling and random projections as general acceleration techniques for algorithms
- Custom autograd functions in PyTorch: trading memory for compute by caching intermediate calculations (e.g., storing `diff` in MSE to avoid recomputation)
- Numerical stability techniques beyond log-sum-exp
- The relationship between Einstein summation notation and gradient computations

## Homework

- Recreate log_softmax(), nll_loss(), and cross_entropy() from scratch and verify against PyTorch's values
- Recreate the matrix multiply from scratch
- Recreate the forward and backward passes for linear layers and ReLU
- Recreate the Module/Layer class pattern with `.forward()` and `.backward()`
- Go through notebook 3: Kernel → Restart and Clear Output, then predict shapes at each step before running
- Share results on the fast.ai forums

## Things Jeremy Says You Should Do

- Learn to use the Python debugger (`pdb`) — it's one of the most powerful things you can do to improve your coding
- Pause the video and go back through each step of the cross-entropy derivation; think not just *what* it's doing but *why*
- Type in lots of different values yourself to build intuition
- Try to recreate everything without peeking as much as possible
- At the very least, go through the notebooks with cleared outputs and try to guess shapes before running cells
- If you don't know high school calculus, check out Khan Academy
- Share your homework results on the fast.ai forums

## Resources

- **Notebook**: [Notebook 3 — Forward and backward passes](https://github.com/fastai/course22p2/tree/master/nbs) (course22p2 repo)
- **Paper**: [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) — Terence Parr & Jeremy Howard
- **Interactive**: [The Intuitive Notion of the Chain Rule (GeoGebra)](https://webspace.ship.edu/msrenault/geogebracalculus/derivative_intuitive_chain_rule.html)
- **Excel**: [Part 1 Excel workbooks (entropy example)](https://github.com/fastai/course22/tree/master/xl)
- **Blog**: [Simple Neural Net Backward Pass](https://nasheqlbrm.github.io/blog/posts/2021-11-13-backward-pass.html)
- **Library**: [SymPy](https://www.sympy.org/) — symbolic differentiation in Python
- **Library**: [PyTorch `nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and [`torch.nn.functional`](https://pytorch.org/docs/stable/nn.functional.html)
- **Learning**: [Khan Academy — Calculus](https://www.khanacademy.org/math/calculus-1)
- **Forum**: [Lesson 13 Official Topic](https://forums.fast.ai/t/lesson-13-official-topic/101876)
- **Forum**: [Calculus Help Topic](https://forums.fast.ai/t/calculus-help-topic/102020)
