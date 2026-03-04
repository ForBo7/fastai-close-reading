The lesson opens with Jeremy showcasing student work from the forum. John Robinson created a striking video by interpolating between seasonal prompts for an oak tree — winter blizzard through spring, summer, fall, and back — using the previous interpolation's final image as the starting image for the next. The result is a remarkably stable, beautiful transition between seasons, and it turns out there was a bug in John's code that accidentally produced the effect, sparking interest in reverse-engineering what made it work so well.

Sebastian (@sebderhy) identified a problem with classifier-free guidance: the update formula u + g·(t − u) can overshoot because the guided vector becomes much longer than the original unconditional prediction. His fix rescales the result so its norm matches the unconditional prediction's norm. Jeremy walks through the visual improvements — better texture on an astronaut's boot, real stars in the sky, a horse that finally has four legs — and explains how rescaling the update direction versus rescaling the whole prediction each have distinct effects. Combining both yields the best results. Ben Poole from Google Brain later pointed out this may already exist in a text-to-speech diffusion paper, raising the question of whether Sebastian independently rediscovered a known technique.

Rekil Prashanth then introduces a cosine schedule for the guidance scale, decreasing it from 7.5 toward zero over the inference steps. The intuition: once the model has the right direction, let it do its thing. The improvement in fine detail is dramatic — a Yorkshire Terrier's eye goes from flat black to lifelike, and fur becomes sharply textured. Jeremy also highlights Alex's lesson notes as a model for how to study: listing all techniques, libraries, and links encountered, trying the lesson on a new dataset (Fashion MNIST), and jotting down tips that might otherwise be forgotten.

The lesson then pivots to reading the DiffEdit paper together. Jeremy demonstrates his preferred paper-reading workflow: saving papers from arXiv directly into Zotero via the browser connector, organizing them in shared group libraries. He walks through the abstract, introduction, and related work, emphasizing that your goal is not to understand every word but to grasp the core idea well enough to implement it. DiffEdit performs semantic image editing — changing a horse to a zebra, a bowl of fruits to a bowl of pears — by automatically generating a mask. The key insight of the paper's three steps: (1) denoise a noised image twice, once conditioned on the reference text and once on the query text, then derive a mask from the difference in noise predictions; (2) encode the input image with DDIM to the desired noise level; (3) decode conditioned on the query, pasting original pixel values in unmasked regions at each step.

Jeremy uses the paper's background section to teach how to read mathematical notation. He recommends learning the Greek alphabet so you can actually pronounce symbols like ε (epsilon) and θ (theta). For unknown notation, he shows two techniques: MathPix (or its free alternative pix2tex) to convert screen-selected math into searchable LaTeX, and downloading the paper's LaTeX source from arXiv to find symbol definitions directly. He decodes the DDPM loss function step by step: 𝔼 is the expected value operator, ‖·‖₂² is the squared L2 norm (just sum of squares), ε is random noise, and ε_θ is a neural network that predicts noise. The whole equation is simply mean squared error — an enormous amount of notation for a very simple idea. He explains expected value with coin-flip and dice examples, and notes that the background section of a paper is a reminder of things you should already know, not a tutorial.

After the break (where Diego reminds everyone about Detexify for drawing and identifying LaTeX symbols), Jeremy moves to building matrix multiplication from scratch. Starting with a 5×784 mini-batch of MNIST digits and a 784×10 weight matrix, he writes the naive triple-nested Python loop and times it at ~450ms — painfully slow. He then introduces Numba: adding @njit to the innermost dot-product loop compiles it to machine code, achieving a ~2000× speedup to about 268 microseconds.

Next comes element-wise operations, introduced through APL. Jeremy fires up TryAPL and demonstrates how APL's concise notation — where true/false are 0/1, and the mean function is just four characters (+/÷≢) — reduces cognitive overhead when exploring mathematical ideas. He covers the Frobenius norm (square each element, sum, take the square root), showing it's trivial in both PyTorch and APL. He replaces the innermost loop with an element-wise multiply-and-sum (a dot product), bringing the time down to ~661 microseconds.

The real payoff comes with broadcasting. Jeremy explains how NumPy/PyTorch compare shapes element-wise from the trailing dimensions, expanding any dimension of size 1 to match. He builds up carefully: broadcasting a scalar across a vector, a vector across a matrix using None/unsqueeze to add unit axes, and then the outer product via c[None] * c[:, None]. He shows how expand_as uses zero-stride storage so no memory is copied. The broadcasting rules are simple: dimensions are compatible if equal or if one is 1.

Armed with broadcasting, Jeremy eliminates the column loop from matrix multiplication. Instead of iterating over columns of B, he takes each row of A, adds a unit axis, multiplies by the entire B matrix, and sums — all in one expression. The result: ~70 microseconds for the mini-batch, and 656ms for the entire 50,000-image dataset. From 450ms for 5 images to 656ms for 50,000 — a transformative speedup that makes training simple models practical. Jeremy closes by emphasizing that broadcasting is the single most critical foundational operation in all deep learning and machine learning code.

---

**Lesson Challenges**

- Implement Step 1 of DiffEdit: automatically generate a segmentation mask by contrasting noise predictions from two different text prompts using the Lesson 9 notebook code
- Reverse-engineer John Robinson's "buggy" interpolation code to understand what it accidentally did right

**Potential Research Directions**

- Whether Sebastian's guidance rescaling is identical to the technique in the text-to-speech diffusion paper Ben Poole referenced
- Cosine-scheduled guidance scale and its interaction with different samplers
- Combining guidance rescaling with scheduled guidance decay
- Improving Detexify's symbol recognition accuracy (Jeremy suggests this as a project)

**Homework**

- Try implementing DiffEdit's mask generation (Step 1) — the code from the Lesson 9 notebook contains everything needed
- Practice broadcasting extensively — it is the most critical foundational operation in deep learning code
- Learn the Greek alphabet for reading papers
- Try Alex's study approach: list all techniques/libraries encountered, try the lesson on a new dataset, and write down tips

**Things Jeremy Says You Should Do**

- Learn the Greek alphabet so you can read equations aloud
- Use Zotero (free, better than Mendeley) with the Chrome connector for paper management
- Read paper abstracts and results first; skip sections that aren't useful; don't read top-to-bottom
- Don't expect to understand background sections from scratch — they're reminders of things you learn elsewhere
- Don't follow every citation — you'll never finish
- Do check the appendices of papers — they often contain extra examples and failure cases
- Use `set_print_options` (PyTorch) or equivalent (NumPy) to widen output display beyond 80 columns
- Build code step-by-step in Jupyter, then merge cells into a function once working
- Consider learning APL — many people say it taught them more about programming than anything else
- Use pix2tex (free, open source) instead of MathPix for LaTeX OCR
- Download paper source from arXiv to search for symbol definitions in LaTeX

**Resources**

- [Lesson 11 discussion thread](https://forums.fast.ai/t/lesson-11-official-topic/101508)
- [DiffEdit paper (arXiv)](https://arxiv.org/abs/2210.11427)
- [Greek alphabet (Wikipedia)](https://en.wikipedia.org/wiki/Greek_alphabet)
- [All-in-one mathematics cheat sheet (PDF)](https://ourway.keybase.pub/mathematics_cheat_sheet.pdf)
- [Glossary of mathematical symbols (Wikipedia)](https://en.wikipedia.org/wiki/Glossary_of_mathematical_symbols#Other_brackets)
- [pix2tex / LaTeX-OCR (GitHub)](https://github.com/lukas-blecher/LaTeX-OCR)
- [MathPix](https://mathpix.com/)
- [Greek Letters for Deep Learning — Anki deck](https://ankiweb.net/shared/info/2118139507)
- [Detexify](https://detexify.kirelabs.org/classify.html)
- [Zotero](https://www.zotero.org/)
- [TryAPL](https://tryapl.org/)
- [Yorick language](https://github.com/LLNL/yorick) — origin of NumPy-style broadcasting
- [Numba](https://numba.pydata.org/)
- [NumPy broadcasting documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- Fashion MNIST dataset
- Lesson 9 notebook (contains code needed for DiffEdit implementation)
