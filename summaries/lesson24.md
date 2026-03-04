The lesson opens with notebook "26 Diffusion UNet," building an unconditional diffusion UNet from scratch. The convolution used throughout is the **pre-activation convolution** — normalization and activation happen first, with the actual convolution at the end. The `UnetResBlock` is a standard preact ResNet block but always stride 1; downsampling is handled separately inside the **DownBlock** via a stride-2 convolution appended after the res blocks. This differs from the usual average-pooling approach — it's inherited from the original DDPM codebase, and Jono notes we shouldn't assume it's optimal, pointing to the general scarcity of ablation studies in the diffusion literature.

The lesson revisits UNet skip connections, recalling the super-resolution UNet from earlier where downsampling activations were added back during upsampling. Here, instead of adding, the activations are **concatenated** — more common in modern UNets. To make saving activations automatic, Jeremy introduces the **SaveModule mixin pattern**: a class whose `forward` calls `super().forward()` and stores the result in `.saved`. Through Python's **multiple inheritance** (method resolution order), `SavedResBlock(SaveModule, UnetResBlock)` produces a res block that automatically remembers its output, with literally zero implementation of its own — just `pass`. The mixin pattern, Jeremy notes, appears throughout the Python standard library.

A **DownBlock** is then a `Sequential` of saved res blocks with an optional saved stride-2 conv for downsampling. An **UpBlock** mirrors this but uses nearest-neighbor upsampling (duplicating pixels into a 2×2 grid) followed by a stride-1 conv, and crucially, it **concatenates** saved activations from the downsampling path via `.pop()`. The complete `UNet2DModel` chains an initial conv (3→224 channels), four down blocks (224→448→672→896), a mid block, four up blocks in reverse, and a final conv back to 3 channels. The up blocks have `num_layers + 1` res blocks because even the downsampling conv's output is saved, yielding one extra cross-connection.

Training works but lacks two critical pieces: **time embedding** and **attention**. For time embedding, the lesson develops **sinusoidal embeddings** step by step. The embedding dimension is split in half (e.g., 16 → 8), an exponent vector is computed from `−log(max_period)` scaled linearly, and the outer product of time steps with these exponents gives raw values. Taking the **sine and cosine** and concatenating them yields the full embedding — each column looks different from its neighbors while adjacent columns remain similar. Jeremy and Jono observe that the standard `max_period=10000` (inherited from NLP) wastes embedding space for diffusion's much smaller sigma range; using 1000 or even 10 produces richer embeddings. This is flagged as a promising area to experiment with.

The `EmbResBlock` integrates time by projecting the embedding to twice the number of filters, splitting via `torch.chunk` into **scale** and **shift**, and applying them to the activations between the two convolutions. SiLU (Swish) — defined as x·σ(x) — is used as the activation throughout. A second Python technique for saving activations is also shown: the `saved` function, which wraps a module's forward method using `functools.wraps` to store the result, avoiding the need for multiple inheritance.

The lesson then moves to **self-attention**. The motivation: convolutions have limited receptive fields, so distant but relevant pixels (like a rabbit's two ears) can't easily inform each other. Attention computes a weighted average of all pixels regardless of distance. For implementation, the 2D feature map is **flattened** to a 1D sequence (batch × sequence × channels). For pixel xᵢ, attention adds a weighted sum of all other pixels, where weights come from a **dot-product similarity matrix**: Q × Kᵀ (both being linear projections of the same input), scaled by 1/√nf, followed by softmax, then multiplied by V (a third projection). The result is added back residually. Jeremy shares a practical trick: instead of three separate Q/K/V projections, use one linear layer to 3×nᵢ and `chunk` it — faster and more concise.

**Multi-head attention** extends this by splitting channels into groups (heads), each computing independent attention weights. The implementation requires just two extra lines: reshape to fold heads into the batch dimension before attention, then reshape back after. The lesson introduces **einops** `rearrange` for this — `'n s (h d) -> (n h) s d'` — which Jeremy says he liked within 10 minutes of first using it. Jono explains why multiple heads matter: softmax concentrates weight on the highest-scoring position, so a single head forces all channels to attend to the same place, while multiple heads let different channel groups attend to different spatial locations.

Attention is only added after a certain depth in the UNet because the attention weight matrix is (H·W)² — at full resolution this would explode memory. Typically attention starts at 16×16; stable diffusion starts at 32×32 since its latents begin at 64×64. Jeremy candidly notes he hasn't personally observed attention improving FID in his experiments, though it might matter more at larger scales.

With attention and an MLP (linear → GELU → layer norm → linear), the lesson shows this is exactly a **Transformer block**: normalize → attention → add → normalize → MLP → add. The mid block is replaced with sequential Transformer blocks, verified to match PyTorch's `nn.TransformerEncoder`. **Vision Transformers** (ViTs) are discussed briefly — they work poorly on ImageNet alone because pure attention needs far more data to learn what convolutions provide as inductive bias, but pre-trained on larger datasets they surpass convnets. A caveat: positional embeddings are fixed-size, so a ViT trained on 224×224 can't directly handle 128×128 images.

The lesson concludes with a **conditional model**. The only changes: `nn.Embedding` maps the 10 Fashion MNIST class labels to vectors the same size as the time embedding, and these are simply **added** to the time embedding before being projected. At sampling time, passing a class ID (e.g., 0 for T-shirt) produces images of that class — no changes needed to the DDIM sampler itself. This completes everything in stable diffusion except CLIP (text conditioning) and the VAE (latent space), which are promised for the next lesson.

---

**Lesson Challenges**

- Replace `.reshape` and `.transpose` calls with equivalent einops `rearrange` expressions
- Experiment with different `max_period` values for sinusoidal embeddings (e.g., 10, 100, 1000 vs. the default 10000)
- Compare adding vs. concatenating skip connections in the UNet
- Try different attention start points and observe the effect on memory and quality

**Potential Research Directions**

- Ablation studies on UNet downsampling (stride-2 conv vs. average pooling vs. other methods)
- Optimal `max_period` for sinusoidal embeddings in diffusion models (the 10000 default is a historical accident from NLP)
- 2D-native attention mechanisms vs. the current "hacky" 1D flattening approach used in stable diffusion
- Whether attention actually improves diffusion on small-scale tasks, and at what image/model size it begins to matter
- Alternative positional/time embeddings (learned vs. sinusoidal, different frequency schedules)

**Homework**

- Complete the unconditional and conditional diffusion UNet notebooks
- Train and sample from both models on Fashion MNIST
- Implement the `saved` function approach as an alternative to the mixin pattern
- Prepare for the next lesson on latent diffusion and VAEs

**Things Jeremy Says You Should Do**

- Experiment with sinusoidal embedding parameters — the default 10000 max period wastes space and "seems like it would be a lot richer to use embeddings with a suitable max period"
- Try einops `rearrange` — "within 10 minutes I liked it much better"
- Look into ablation studies before assuming any architectural choice is optimal
- Practice the mixin pattern and the `saved` function pattern as useful Python techniques

**Resources**

- [Lesson 24 video](https://youtu.be/DH5bp6zTPB4)
- [Lesson 24 official topic (forums)](https://forums.fast.ai/t/lesson-24-official-topic/104358)
- [Lesson 24 course page](https://course.fast.ai/Lessons/lesson24.html)
- Notebook: "26 Diffusion UNet" (unconditional diffusion from scratch)
- Notebook: "26a Diffusion Attention" (self-attention and multi-head attention)
- Notebook: "26b Diffusion with Conditioning" (conditional model)
- [einops library](https://github.com/arogozhnikov/einops) — `rearrange` for tensor reshaping
- [PyTorch SiLU docs](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
- [diffusers `AttentionBlock`](https://github.com/huggingface/diffusers) — reference attention implementation
- PyTorch `nn.MultiheadAttention` and `nn.TransformerEncoder`
- [ViT paper — "An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929)
- Karras et al. notebook (notebook 23, referenced for the noisify/schedule approach)
- DDPM and Improved DDPM papers (referenced for UNet architecture lineage)
