The lesson opens with a provocation. An xkcd comic from late 2015 made a joke about how recognizing whether a photo contains a bird was so impossibly hard it would require "a research team and five years." Jeremy sets out to demolish that joke in real time. Using Python, DuckDuckGo image search, and the fastai library, he downloads 200 bird photos and 200 forest photos, builds and trains a deep learning model on his presentation laptop — while simultaneously streaming video — and in under 30 seconds produces a classifier that identifies a bird with probability 1.0000. Something that was the punchline of a joke in 2015 is now a two-minute exercise.

Before diving into how that worked, Jeremy surveys what deep learning can do now. DALL·E 2 generates elaborate artwork from nothing but text prompts — a friend typed Twitter bios like "happy Sisyphus" and "bookbear" and got back surreal, creative illustrations. MidJourney produces vibrant images of "a female scientist writing code" or "a psychedelic pink elephant." When actual artists use these tools, spending months tuning their own deep learning software, the results are breathtaking. It's not just images: Google's Pathways Language Model (PaLM) can answer arbitrary English questions with chain-of-thought reasoning and even explain jokes involving technical puns about TPU pods and whale pods. Deep learning models are doing things that seemed beyond the horizon of what computers could accomplish in a human lifetime.

Jeremy then explains his teaching philosophy. Drawing on education researchers Paul Lockhart (*A Mathematician's Lament*) and David Perkins (*Making Learning Whole*), he teaches top-down: start with a complete working model, then gradually peel back the layers to understand foundations. The traditional bottom-up approach — years of linear algebra before touching anything interesting — is not how most people learn effectively. It's how we teach sports: show the whole game first, let people play, then build skills. Students with heavy technical backgrounds may find this uncomfortable; students without will find it natural.

After establishing his credentials — 30 years in ML, top Kaggle competitor, co-founder of fast.ai, inventor of ULMFiT, author of *Deep Learning for Coders* — Jeremy turns to the conceptual heart of the lesson: why deep learning works where older approaches didn't. In 2012, image recognition required teams of domain experts hand-crafting thousands of features ("relationships between epithelial nuclear neighbors") before feeding them into a simple model. Neural networks skip all of that. Visualizations from Zeiler and Fergus show what a trained network learns on its own: Layer 1 discovers edges and color gradients. Layer 2 finds corners, curves, and circles. Layer 3 detects repeating geometric patterns, flower edges, even text. Deeper layers build increasingly sophisticated detectors. Nobody coded these — they emerged from training. This is why the field is called "deep" learning: stacking layers of learned features produces representational power no hand-engineering could match.

Image classifiers aren't limited to images. Students have converted sound waveforms into spectrograms and classified them (beating state-of-the-art results), turned time series into pictures, and even visualized mouse-movement patterns as images for fraud detection. With creativity, the same foundational technique extends remarkably far. And none of this requires vast data or expensive hardware — the bird classifier used 200 images on a laptop. The myth that deep learning demands enormous datasets is largely propagated by companies selling compute. Transfer learning, where a model pre-trained on a large dataset is fine-tuned on a small one, is a key reason small datasets work so well.

The lesson covers PyTorch and fastai. PyTorch has overtaken TensorFlow decisively in research, and what researchers adopt is a leading indicator for industry. However, PyTorch alone requires substantial boilerplate. fastai, built on top of PyTorch, dramatically reduces the code needed while embedding best practices. The AdamW optimizer implementation, for instance, shrinks from a full screen of PyTorch code down to a small highlighted block in fastai. Less code means fewer bugs, easier maintenance, and better defaults.

Jeremy then walks through the bird classifier code step by step. Everything runs in Jupyter notebooks — the same environment used for the slides, the book, the fastai library itself, and even fastai's test suite. On cloud platforms like Kaggle, notebooks become the world's most powerful calculator. The code starts by downloading one bird and one forest photo to visually verify them, then batch-downloads 200 of each, resizes them to 400 pixels max, and removes any broken images.

The **DataBlock API** is introduced as the central abstraction for getting data into a model. Five things vary between projects: (1) what kind of input (images), (2) what kind of output (categories), (3) how to get the items (a function listing image files), (4) how to create a validation set (random 20% split), and (5) how to label items (parent folder name). An `item_transforms` step resizes all images to 192×192 pixels. From the DataBlock, **DataLoaders** are created — these feed batches of data to the GPU during training. `show_batch()` displays a grid of labeled samples for visual inspection.

The **Learner** combines a model and its data. Using `vision_learner` with a pre-trained ResNet18 from ImageNet (over a million images, 1,000+ categories), fastai downloads those weights so training starts from a knowledgeable base rather than random noise. The `fine_tune()` method carefully adjusts these pre-trained weights for the new task. After a few seconds of training, the model achieves 100% accuracy. Calling `learn.predict()` on the original bird photo returns `("bird", 0, tensor([1.0000, 0.0000]))`.

Beyond classification, Jeremy demonstrates **segmentation** (coloring every pixel of road scenes by category — cars, fences, buildings — trained in about 20 seconds), **tabular analysis** (predicting income brackets from demographic data, using `TabularDataLoaders` with categorical and continuous columns), and **collaborative filtering** (predicting movie ratings from user-movie pairs, the basis of recommendation systems like Spotify). Each uses the same pattern: load data, create a learner, train, inspect results.

The lesson takes a step back to describe the machine learning training loop as Arthur Samuel conceived it. A normal program takes inputs and produces results. A model adds **weights** (parameters) alongside inputs. A neural network is astonishingly simple: multiply inputs by weights, add them up, replace negatives with zeros, repeat across layers. This is provably an infinitely flexible function — it can approximate any computable function given sufficient data and training time. Training starts with random weights, calculates a **loss** measuring how wrong the results are, then updates the weights to reduce that loss, iterating until the model becomes good. Once trained, the weights are fixed and the model becomes just another callable function: `inputs → model → results`, indistinguishable from a regular program. The Mark I Perceptron from 1957 already embodied this idea; what changed is GPUs, SSDs, and available data.

Jeremy closes with homework and encouragement. The most important thing is to experiment — run the Kaggle notebooks, change things, try classifying something besides birds and forests, push yourself a little but finish something before the next lesson. Read chapter one of the book. Share your work in the fast.ai forums' "Share Your Work Here" thread, where past students have posted over a thousand projects — many of which became startups, papers, and career breakthroughs.

---

**Lesson Challenges**

- Build a classifier for something other than birds vs. forests — try three or four categories
- Experiment with the [Is it a bird?](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data) notebook on Kaggle: change inputs, try different searches, see what happens
- Complete the quiz questions at the end of chapter 1

**Potential Research Directions**

- Converting non-image data (sound, time series, mouse movements) into images for classification
- Transfer learning with very small datasets — how few examples can you get away with?
- Deep learning for creative art generation (DALL·E 2, MidJourney, custom tools)
- Ethical implications of deep learning capabilities (see [ethics.fast.ai](https://ethics.fast.ai))
- Zeiler & Fergus feature visualization — what do deeper layers learn for different domains?
- Fraud detection via behavioral pattern visualization

**Homework**

- Read [chapter 1](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb) of the fastai book
- Run the Kaggle notebooks and experiment with changes
- Build your own classifier project and share it in the [forums](https://forums.fast.ai/c/p1v5/54)
- Answer the chapter 1 questionnaire ([solutions](https://forums.fast.ai/t/fastbook-chapter-1-questionnaire-solutions-wiki/65647))

**Things Jeremy Says You Should Do**

- Experiment — run notebooks, change things, try your own data
- Always start notebooks with `!pip install -Uq fastai` to avoid version issues
- Visually inspect your data at every step (`show_batch()`, view individual images)
- Push yourself a little, but make sure you finish something before the next lesson
- Read the book — it presents the same material differently, which aids learning
- Share your work in the forums — it leads to feedback, jobs, and community
- Check out [ethics.fast.ai](https://ethics.fast.ai) for the data ethics course
- Check out Radek Osmulski's book [Meta Learning](https://radekosmulski.gumroad.com/l/learn_deep_learning) for tips on getting into the field

**Resources**

- Course page: [course.fast.ai/Lessons/lesson1.html](https://course.fast.ai/Lessons/lesson1.html)
- Kaggle notebook: [Is it a bird? Creating a model from your own data](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data)
- Kaggle notebook: [Jupyter Notebook 101](https://www.kaggle.com/code/jhoward/jupyter-notebook-101)
- Book chapter 1: [fastbook/01_intro.ipynb](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb)
- Book (published): [Deep Learning for Coders with fastai and PyTorch](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch-ebook-dp-B08C2KM7NR/dp/B08C2KM7NR)
- Book (free notebooks): [github.com/fastai/fastbook](https://github.com/fastai/fastbook)
- All lesson notebooks: [github.com/fastai/course22](https://github.com/fastai/course22)
- Chapter 1 questionnaire solutions: [forums.fast.ai](https://forums.fast.ai/t/fastbook-chapter-1-questionnaire-solutions-wiki/65647)
- fastai paper: [Fastai: A Layered API for Deep Learning](https://www.mdpi.com/2078-2489/11/2/108) ([arxiv](https://arxiv.org/abs/2002.04688))
- [timm](https://timm.fast.ai) — PyTorch Image Models library
- fast.ai teaching philosophy: [Providing a Good Education in Deep Learning](https://www.fast.ai/2016/10/08/teaching-philosophy/)
- Data ethics course: [ethics.fast.ai](https://ethics.fast.ai)
- *A Mathematician's Lament* by Paul Lockhart: [PDF](https://www.maa.org/external_archive/devlin/LockhartsLament.pdf)
- *Making Learning Whole* by David Perkins: [Harvard](http://www.pz.harvard.edu/resources/making-learning-whole-how-seven-principles-of-teaching-can-transform-education)
- *Meta Learning* by Radek Osmulski: [Gumroad](https://radekosmulski.gumroad.com/l/learn_deep_learning)
- DALL·E 2 Twitter bio illustrations: [tweet](https://twitter.com/nickcammarata/status/1511861061988892675)
- Jupyter RISE (notebook presentations): [docs](https://rise.readthedocs.io/en/stable/)
- nbdev: [nbdev.fast.ai](https://nbdev.fast.ai/)
- fast.ai forums: [forums.fast.ai](https://forums.fast.ai/c/p1v5/54)
- Help threads: [Setup](https://forums.fast.ai/t/help-setup/95289) · [Datasets & Gradio](https://forums.fast.ai/t/help-creating-a-dataset-and-using-gradio-spaces/96281) · [Colab/Kaggle](https://forums.fast.ai/t/help-using-colab-or-kaggle/96280) · [Python/git/bash](https://forums.fast.ai/t/help-python-git-bash-etc/96282) · [SGD & NN foundations](https://forums.fast.ai/t/help-sgd-and-neural-net-foundations/96286) · [fastai/PyTorch/numpy basics](https://forums.fast.ai/t/help-basics-of-fastai-pytorch-numpy-etc/96285) · [Other beginner questions](https://forums.fast.ai/t/help-beginner-questions-that-dont-fit-elsewhere/96284)
- xkcd 1425: [xkcd.com/1425](https://xkcd.com/1425/)
- PixSpy (image pixel inspector): referenced in lesson
