Jeremy opens from his home study — the university room got booked — and he's visibly excited. This lesson covers genuinely new ground, he says, stuff that hasn't been in courses like this before. The energy is infectious.

He starts by reminding you about the companion book (free on GitHub or Colab), recommending you read the quiz at the end of each chapter *before* watching the video, then answer it *after*. He highlights Radek's **aiquizzes.com** for spaced-repetition practice with brand new questions. The forums get a shoutout too — every lesson has an official topic, and the "Summarize this topic" button filters to the most-upvoted replies, saving you from information overload.

Then comes the student showcase. A damaged car classifier, a beard detector (a one-letter typo away from "bird"), a blog post on FastPages, and a full production food classifier web app from Suvash. Jeremy's message is clear: you'll be able to do all of this too — and specifically, *today*.

The production pipeline begins with **Chapter 2 of the book**. Jeremy walks through the workflow: find a problem, gather data, then clean it. But he detours into Jupyter productivity tips first — install **Jupyter Notebook Extensions** for a Table of Contents sidebar and **Collapsible Headings** with keyboard navigation (left/right arrows to jump between and collapse sections).

For gathering images, he swaps Bing for **DuckDuckGo** (no API key needed) and demonstrates the Python help system: `??` for source code, `?` for brief info, and `doc()` for full documentation with links. Then comes the counterintuitive lesson that defines much of the lecture: **train a model before you clean your data**. The model's mistakes reveal your data's problems far more efficiently than manual inspection.

He takes a sidebar through **resizing methods** — squish (distorted aspect ratios), crop (loses edges), pad with zeros (black bars), and the preferred **RandomResizedCrop** which grabs different random portions each epoch, effectively generating infinite slightly-different views. This leads naturally into **data augmentation** via `aug_transforms`, which warps, rotates, recolors, and stretches images in real time during training — no copies stored, just on-the-fly transformations. For anything beyond five or ten epochs, you'll want both.

After training, Jeremy introduces the **confusion matrix** — which category errors is the model making? — and **`plot_top_losses`**, which surfaces the images where the model is most wrong or most uncertain. A high loss means either confidently wrong or correctly classified but with low confidence. This diagnostic power feeds directly into the **ImageClassifierCleaner**, a fastai widget that shows images ordered by loss so you can quickly relabel or delete the problematic ones. The insight: always train first, *then* clean — let the model guide your data curation.

A practical note on **GPU vs CPU RAM**: GPUs can't swap to disk, so close unused notebooks to free VRAM. Jeremy also recommends watching the entire video once through without coding, then going back to follow along — that way you always know what's coming.

The lesson then pivots to deployment. Jeremy introduces **HuggingFace Spaces** with **Gradio** as the target platform (free, public, easy). He walks through creating a Space, cloning it via Git, and writing a minimal `app.py`. For Windows users, he demonstrates installing **WSL** (a one-command process) and **Windows Terminal**, then opening VS Code from the terminal. The minimal Gradio app — three lines: import, function, interface — gets pushed to HuggingFace and is live within a minute.

For the real classifier, Jeremy trains a dog/cat model on Kaggle, exports it with `learn.export('model.pkl')` — the *only* step requiring a GPU — downloads the pickle file, and builds the Gradio interface locally. The `classify_image` function calls `learn.predict()`, converts probabilities to plain floats (Gradio doesn't handle PyTorch tensors), and returns a dictionary. There's a charming anecdote about his six-year-old daughter searching "what is a dog mixed with a cat called" and testing the hybrid image.

He shows how to export notebook cells to a script using nbdev's `#|export` markers and `notebook2script`, then pushes everything to HuggingFace. The deployed app classifies dogs, cats, and the mysterious dog-cat hybrid (the model says cat).

On training duration: train until it's good enough, until patience or compute runs out, or until the error rate starts getting worse. Simple as that.

The final act goes beyond prototyping. Jeremy reveals Gradio's **"View the API"** button — any Space exposes a REST endpoint. He builds **tinypets**, a pure JavaScript web app (fits on one screen of HTML) that calls the HuggingFace API for pet breed classification. A community member's **"Get to know your pet"** site combines an image classifier with an NLP model — the kind of composition that's impossible with a canned interface. For hosting, **GitHub Pages** (free, via Jekyll) serves static HTML, and **FastPages** simplifies the setup.

Jeremy closes by teasing the next lesson: natural language processing, a look under the hood at how models work, stochastic gradient descent, and maybe a bit of calculus.

---

**Lesson Challenges**
- Build your own image classifier using your own dataset
- Deploy it as a working web application on HuggingFace Spaces
- Create a JavaScript front-end that calls your model's API

**Potential Research Directions**
- How different resizing strategies (squish, crop, pad, RandomResizedCrop) affect model accuracy across domains
- Model-guided data cleaning workflows — when does training-first outperform manual curation?
- Data augmentation strategies for non-image modalities (text, tabular, audio)
- Combining multiple models (image + NLP) in production apps, as in the "Get to know your pet" example
- Lightweight deployment architectures: when does a static JS frontend + hosted API outperform server-side rendering?

**Homework**
- Read [Chapter 2](https://github.com/fastai/fastbook/blob/master/02_production.ipynb) of the book and complete the end-of-chapter questionnaire
- Train and deploy your own classifier to HuggingFace Spaces with Gradio
- Try the quizzes on [aiquizzes.com](https://aiquizzes.com)

**Things Jeremy Says You Should Do**
- Read the book alongside the course — it covers things from different angles
- Read the quiz before watching the video, then answer it after
- Use the "Summarize this topic" button on the forums to cut through noise
- Always train a model *before* cleaning your data
- Use `RandomResizedCrop` and `aug_transforms` for training beyond ~5–10 epochs
- Close and halt unused Jupyter notebooks to free GPU memory
- Watch the entire video without touching the keyboard first, then go back and follow along
- Use mamba (not conda) for installing Python environments — it's much faster
- Make your HuggingFace Spaces public to build your portfolio
- Don't use the system Python — install your own via mambaforge/fastsetup

**Resources**
- Course page: [course.fast.ai/Lessons/lesson2.html](https://course.fast.ai/Lessons/lesson2.html)
- Book Chapter 2: [fastbook/02_production.ipynb](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)
- Chapter 2 questionnaire solutions: [forums.fast.ai](https://forums.fast.ai/t/fastbook-chapter-2-questionnaire-solutions-wiki/66392)
- Saving a basic fastai model: [Kaggle](https://www.kaggle.com/jhoward/saving-a-basic-fastai-model) / [Colab](https://colab.research.google.com/drive/1M-mzhZdFQ2XWBSbLCuKzrmLsm0aLEYxQ?usp=sharing)
- Gradio + HF Spaces tutorial by Tanishq Abraham: [tmabraham.github.io](https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial)
- HuggingFace Spaces: [huggingface.co/spaces](https://huggingface.co/spaces)
- fastsetup: [github.com/fastai/fastsetup](https://github.com/fastai/fastsetup)
- WSL install: [docs.microsoft.com](https://docs.microsoft.com/en-us/windows/wsl/install)
- Windows Terminal: [Microsoft Store](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701)
- tinypets: [GitHub](https://github.com/fastai/tinypets) / [site](https://fastai.github.io/tinypets/)
- tinypets fork: [GitHub](https://github.com/jph00/tinypets) / [site](https://jph00.github.io/tinypets/)
- aiquizzes.com: [aiquizzes.com](https://aiquizzes.com)
- forums.fast.ai: [forums.fast.ai](https://forums.fast.ai)
- Jupyter Notebook Extensions: `pip install jupyter_contrib_nbextensions`