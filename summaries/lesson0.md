Jeremy Howard opens this entirely optional "Lesson Zero" by explaining its purpose: too many students reach the end of the course only to realize how they *should* have approached it from the beginning. He intends to spare you that regret. The course — *Practical Deep Learning for Coders* — has a companion book co-authored with Sylvain Gugger, available for purchase on Amazon or entirely free as Jupyter Notebooks in the "fastbook" GitHub repo. Each lesson maps roughly to a chapter. The course covers the first half of the book; Part Two covers the rest plus new material.

The first and most emphatic instruction is to **finish the damn course**. Jeremy shows YouTube analytics proving many students drop off, and urges you to schedule specific days for watching and for assignments, and to tell someone you're going to finish — use social pressure. Then he raises the stakes: don't just finish the course, **finish a project**. He introduces Christine McLeavey, a fast.ai alumna now at OpenAI, whose deep-learning music generation project (which the BBC orchestra eventually performed) grew directly from Jeremy's advice to polish one great project rather than accumulate half-finished ones. The project need not be world-changing — it could be a cousin-recognizer for your fiancée's family reunions, or the "Hot Dog or Not Hot Dog" app from Silicon Valley that got millions of downloads. Or it could solve medicine.

Jeremy then hammers the theme of **tenacity** — the only consistent trait among every fast.ai student who became a world-class practitioner. Being tenacious doesn't mean ignoring life's obstacles; it means returning to the work after the obstacle passes, even if that takes a year. He illustrates with Radek Osmulski, author of *Meta Learning*, a non-degree-holding former corporate worker who couldn't code, failed repeatedly at learning deep learning, but kept trying until he won a Kaggle competition, joined a top medical AI startup, and now works at a nonprofit translating animal language.

A major anti-pattern is then dissected: **endless preparation**. Many would-be practitioners spiral from calculus to real analysis to set theory, never actually training a model. The fast.ai philosophy is the opposite — you train a model in Week One. Theory is introduced as needed, in context. Radek and James Clear are cited: claiming you need to "learn more" before starting is just a crutch. Jeremy assures you that the actual linear algebra required for virtually all deep learning is matrix multiplication — multiplying things together and adding them up.

The four-step method for doing a fast.ai lesson is laid out: (1) watch the lecture, (2) run the notebook and experiment with it — change things, break things, feed your brain input-output patterns, (3) reproduce the results from scratch in a blank notebook, typing the code yourself, (4) repeat parts of the lesson with a different dataset. Most students go through the entire course two or three times before completing all four steps for every lesson.

Jeremy then walks through the practical setup. **Notebook servers** like Google Colab and Gradient are the easiest — click and you're running Jupyter. **Full Linux servers** (AWS EC2, Google Cloud, Jarvis Labs) require more setup but mirror real-world workflows. He demonstrates Colab live: enabling the GPU runtime, running the first cell that installs everything via `pip`, connecting to Google Drive, and training a cat-vs-dog classifier that reaches 99% accuracy in 54 seconds. He also shows the "clean" versions of each notebook (in the `clean/` folder) — identical code with all prose and outputs stripped — designed for active recall: before running each cell, predict what it does and what it will output. At the bottom of each chapter, a questionnaire tests your recall further.

Radek's "four-legged table" of practical skills is highlighted: coding concepts, a good editor, git/GitHub, and SSH/Linux. MIT's *The Missing Semester of Your CS Education* is recommended for filling these gaps. Jeremy emphasizes **sharing your work** — blogging, tweeting, posting to the forums. Nobody is better positioned than you to write for the audience of who you were six months ago. Fast Pages (now succeeded by Quarto) makes blogging from Jupyter Notebooks trivial. He shows how Aman Arora turned one of Jeremy's talks into a blog post that reached audiences the original video never did, including the CEO of Australia's CSIRO Data61.

The lesson closes with deeper strategic advice. Machine learning's distinguishing feature is **generalization** — training on one set of data and performing well on unseen data. A good validation set is therefore critical, and Rachel Thomas's blog post on the topic is highlighted. ML code is notoriously hard to debug because errors are silent — your model trains, but is subtly wrong. The antidote is to always start with a **simple baseline** (even just the average), then iterate incrementally. Jeremy recounts Silicon Valley founders whose elaborate systems turned out to be worse than a simple average they never bothered to check. Kaggle competitions are recommended as ideal projects: they force the full end-to-end pipeline, they have leaderboards for instant feedback, and daily incremental improvement over weeks teaches more than you'd expect. Finally, everything you build — blog posts, forum contributions, GitHub repos, Kaggle entries — becomes the **portfolio** that gets you a job, especially at startups and emerging AI teams where credentials matter less than demonstrated capability.

---

**Lesson Challenges**

- Train the cat-vs-dog classifier from the Chapter 1 notebook on Colab (or equivalent platform)
- Work through the "clean" version of the notebook, predicting outputs before running each cell
- Complete the Chapter 1 questionnaire without looking at the book

**Potential Research Directions**

- How validation set design affects model reliability across domains (cf. Rachel Thomas's blog post)
- Meta-learning and learning-to-learn strategies (cf. Radek Osmulski's *Meta Learning*)
- Transfer learning effectiveness with very small datasets (<50 items)
- The relationship between incremental baseline iteration and final model performance in Kaggle competitions

**Homework**

- Set up a GPU-enabled environment (Colab, Kaggle Notebooks, Paperspace Gradient, or a full Linux server)
- Run the Chapter 1 notebook end to end
- Reproduce the Chapter 1 results from scratch in a blank notebook
- Train a model on your own dataset and share it on the [fast.ai forums](https://forums.fast.ai/t/share-your-work-here/96015)
- Start a blog (using Fast Pages / Quarto / similar)
- Join a Kaggle competition

**Things Jeremy Says You Should Do (Lesson 0)**

- Finish the damn course — schedule it, tell someone, commit
- Finish a project — one polished project is better than ten half-finished ones
- Be tenacious — keep going after setbacks, even if it takes a year to resume
- Stop endlessly preparing — don't spiral into prerequisite rabbit holes; start training models immediately
- Write code every day — read and write as much deep learning code as possible
- Share your work — tweet, blog, post to forums; write for the you of six months ago
- Use the "clean" notebooks and chapter questionnaires for active recall
- Start with a simple baseline for every project
- Repeat lessons with different datasets
- Join Kaggle early, enter competitions with the intent to win, and make small daily improvements
- Build a portfolio — your blog posts, GitHub, forum contributions, and Kaggle entries are what will get you a job
- Follow ML practitioners on Twitter to immerse yourself in the community
- Learn practical tools: git, SSH, Linux, a good editor (the "four-legged table")

**Resources**

- [Practical Deep Learning for Coders — Course Homepage](https://course.fast.ai)
- [Lesson 0 Video: "How to fast.ai"](https://youtu.be/gGxe2mN3kAg)
- [fastbook GitHub Repo (free book as notebooks)](https://github.com/fastai/fastbook)
- [Deep Learning for Coders with fastai and PyTorch (Amazon)](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527)
- [fast.ai Forums](https://forums.fast.ai)
- ["Share Your Work Here" Forum Thread](https://forums.fast.ai/t/share-your-work-here/96015)
- [Kaggle Notebooks (free GPU)](https://www.kaggle.com/docs/notebooks)
- [Paperspace Gradient](https://gradient.run/notebooks)
- [Google Colab](https://colab.research.google.com)
- [Harvard CS50 (intro CS course)](https://cs50.harvard.edu)
- [The Missing Semester of Your CS Education (MIT)](https://missing.csail.mit.edu)
- [How (and why) to create a good validation set — Rachel Thomas](https://www.fast.ai/2017/11/13/validation-sets/)
- *Meta Learning: How To Learn Deep Learning And Thrive In The Digital World* — Radek Osmulski
- [Fast Pages (blogging platform)](https://github.com/fastai/fastpages)
- [fast.ai Library Documentation](https://docs.fast.ai)
- [PyTorch](https://pytorch.org)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Gradio](https://gradio.app)

---

**Things Jeremy Says to Do — All Lessons** *(from [this forum post](https://forums.fast.ai/t/things-jeremy-says-to-do/36682) by Robert Bracco)*

*Lesson 1*

1. Don't try to stop and understand everything
2. Learn Jupyter keyboard shortcuts — 4-5 each day
3. Run the code. Don't go deep on theory. Play with the code, see what goes in and out
4. Pick one project. Do it really well. Make it fantastic
5. Run the notebook, then get your own dataset and run it
6. If you have lots of categories, use `interp.most_confused(min_val=n)` instead of confusion matrix

*Lesson 2*

1. If forum posts are overwhelming, click "summarize this topic"
2. Follow official server install/setup instructions
3. It's okay to feel intimidated — just pick one piece and dig into it
4. If you're stuck, keep going
5. If unsure which learning rate is best from plot, try both and see
6. Use CPU for inference in production (except at massive scale)
7. Don't spend too much time gathering data — get a small amount first, see how it goes
8. Watch Rachel's talk "There's no such thing as not a math person"

*Lesson 3*

1. Cite and thank dataset creators
2. Try to solve a problem using multi-label classification, image regression, image segmentation, etc.
3. Always use the same stats the model was trained with
4. Try progressive resizing (64→128→256) — it works great

*Lesson 4*

1. For NLP, use all your text (including unlabeled validation set) to train the language model
2. Try both random forest and neural net — may as well try both
3. Use ML terms (parameters, layers, activations, etc.) accurately

*Lesson 5*

1. The answer to "Should I try X?" is to try X and see
2. Create your own `nn.Linear` class from scratch — write more things from scratch and debug them
3. Take Lesson 2 SGD and add momentum to it

*Lesson 6*

1. Data augmentation for different domains is a huge research opportunity
2. Go through the convolution kernel and heatmap notebook carefully — run and change the code, think about tensor shapes and "why?"

*Lesson 7*

1. Don't be intimidated — it's meant to give you ideas to keep busy
2. Go back and rewatch the videos — you'll understand things you didn't before
3. Write code and put it on GitHub — it doesn't have to be great
4. Start reading papers — skip derivations/theorems/lemmas, read the "why" and results. Write summaries for "you of 6 months ago"
5. Get together with others — study groups, meetups, build things. Finish something, then make it better
6. Just code all the time. Rebuild the notebooks from scratch without cheating

*Lesson 8*

1. The cutting edge of DL is about engineering, not papers
2. Experiment lots in your domain. Write stuff down for "the you of six months ago"
3. If you don't understand something from Part 1, go back and rewatch — don't just blast forward
4. Overfit → Reduce overfitting → There is no step 3. Overfitting means validation error getting *worse*, not just training loss < validation loss
5. Learn to pronounce Greek letters — you can't read what you can't pronounce
6. Get very used to: PyTorch tensors, `.view()`, matrix multiplications, `c[i,j] += a[i,k] * b[k,j]`
7. Take the most mind-bending broadcast and convince yourself why it works (Excel or paper)
8. Apply simple broadcasting rules — don't try to keep it in your head
9. Always normalize validation and training sets the same way
10. Read papers from competition winners
11. Read section 2.2 of the ResNet paper
12. Put comments in your code so the next person knows what you're doing
13. If you don't remember the chain rule, do Khan Academy's tutorial

*Lesson 9*

1. Don't assume libraries are correct — dig into them yourself
2. Don't set a random seed — see variation in your model
3. Learn about Python coroutines
4. Schedule everything: dropout, data augmentation, weight decay, learning rate, momentum

*Lesson 10*

1. It's okay if you're not keeping up — don't feel you need to understand everything in a week
2. When lost, go back to where it was easy and find the gap
3. Know these dunder methods: `__getitem__`, `__getattr__`, `__setattr__`, `__del__`, `__init__`, `__new__`, `__enter__`, `__exit__`, `__len__`, `__repr__`, `__str__`
4. Be good at browsing source code — learn: jump to symbol, jump to tag, jump to library tags, go back, search, outlining/folding
5. Mean absolute deviation is underused — use it more (it's less sensitive to outliers than std dev)
6. Replacing squares with absolute values often works better
7. Never look at an equation without also typing it in Python, calculating values, and plotting it
8. Toy problem: how accurate can you make MNIST with only the layers built so far?
9. Epsilon is a fantastic hyperparameter — use it to train better
10. Create interesting toy problems — try best single-epoch accuracy with any normalization

*Lesson 11*

1. Learn to create small, workable, useful datasets — come up with toy problems in your domain
2. Learn about `compose` in programming
3. Use telemetry to view activations of different layers — combine theory and practice
4. Make Adam's epsilon ~0.1 (between 1e-3 and 1e-1), not 1e-7
5. Look at / listen to your augmented data — check for information loss
6. Think about *when* in the pipeline to do augmentation (bytes vs floats)
7. For non-image augmentation: what changes wouldn't alter the label but still produce reasonable examples?

*Lesson 12*

1. Be careful with automated formatting — unconventional formatting can aid understanding
2. Read the Mixup paper
3. Make code equations match the paper as closely as possible
4. Don't wait for data cleanup to start modeling — start now
5. Read the Bag of Tricks paper — think about *why* each tweak was made. Build architectures thoughtfully
6. Never freeze batchnorm layers when doing partial-layer fine-tuning
7. Debug DL by not making mistakes: make code so simple it can't have bugs, check every intermediate result, keep a journal
8. Leave data as raw as possible when preprocessing for neural nets
9. Learn a new programming language — each one makes you a better developer

*Lesson 13*

1. Use Python-in-Swift to fill gaps, but build native solutions as soon as possible
2. If something isn't how you want it, change it
3. Study interesting code patterns after the lesson
4. You don't have to use every advanced feature

*Lesson 14*

1. Pick a piece that's interesting in your domain and explore it over 12-24 months — start small (a notebook, convert a library, write a blog post)
