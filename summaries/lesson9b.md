Waseem, an Entrepreneur in Residence at fast.ai currently visiting headquarters in Australia, walks through the math behind diffusion models together with Tanishq, a PhD student at UC Davis who also works at Stability AI. Jeremy is present throughout, interjecting with connections to earlier course material. Waseem has no formal math background and approaches this as someone who found the derivation beautiful and wants to help others parse the notation and concepts.

The talk opens with the very first piece of math from the 2015 Sohl-Dickstein et al. paper: q(x⁰), the data distribution. The notation is unpacked carefully — x is the input variable, the superscript 0 implies a sequence (x¹, x², …), and q is a probability density function. The choice of q rather than p hints that p will appear later for the thing we actually want to model, with q playing the supporting role.

Tanishq grounds q(x⁰) concretely: if we're working with images, x⁰ is just an image — say an MNIST digit. q acts as a "magic API" (the same metaphor from Lesson 9) that takes in an image and returns its probability of looking like a real digit. You can't write down what q actually is, but you imagine it exists. PDFs rarely appear in code directly, yet they let you treat random quantities as ordinary functions with centuries of math behind them, and they eventually reduce to simple equations.

The forward transition q(x^t | x^{t-1}) is introduced next — a conditional PDF describing how one image becomes a slightly noisier version. It's defined as a Gaussian: N(x^t; √(1−β_t) x^{t-1}, β_t I). The semicolon separates the variable from the parameters (mean and covariance). The normal distribution is chosen because it's described by just two parameters (mean and covariance) and has thin tails, so behavior outside a small region can be ignored.

Jeremy explains the covariance matrix by connecting it to CLIP from an earlier lesson: if you take the dot product of the same set of vectors against themselves, the diagonal gives you variance and the off-diagonal entries tell you how pairs of pixels co-vary. Setting all off-diagonal entries to zero assumes pixels are independent — unrealistic for real images, but the assumption we make here. The identity matrix I has ones on the diagonal and zeros elsewhere; multiplying by β_t gives each pixel variance β_t with no cross-pixel relationships.

The formula's behavior at the extremes is then examined. When β = 0, the mean is exactly the previous image and the variance is zero — you get the same image back. When β = 1, the mean is zero and the variance is I — pure noise. Between these extremes, you get a mixture of signal and noise, controlled by β.

Chaining many such small steps produces the forward diffusion process: a sequence from a clean image to pure noise. This is formally a Markov process with Gaussian transitions — "Markov" meaning each step depends only on the previous one, "Gaussian" meaning the transitions are normal distributions. Sampling is straightforward: draw from N(0, 1), multiply by the standard deviation, add the mean.

The crucial insight, discovered around the 1950s, is that the reverse process — going from noise back toward data — has the same functional form: p(x^{t-1} | x^t) = N(x^{t-1}; □, △), where □ and △ are unknown parameters to learn. q describes the forward process, p describes the reverse. The question becomes: how do we find those unknowns?

The answer involves the likelihood function, but computing it exactly requires intractable high-dimensional integrals over thousands of steps. Taking the log helps — the log is monotonically increasing (so the optimum doesn't change), converts products to sums, and cancels out the exponentials in Gaussians. Even the log likelihood can't be optimized directly, but there exists a tractable surrogate called the ELBO (Evidence Lower Bound). Optimizing the ELBO is almost as good as optimizing the true likelihood, and because everything is Gaussian, the KL Divergence between forward and reverse distributions can be computed analytically.

The 2020 DDPM paper (Denoising Diffusion Probabilistic Model) simplifies further with two assumptions: the reverse variance is a fixed constant (not learned), and the forward step sizes are also constants. This reduces the entire problem to predicting the noise — train a neural network that takes a noisy image and predicts what part is noise, using plain MSE loss. Despite these simplifications, DDPM produces much better images.

Tanishq then connects this to the score function perspective from Lesson 9. The score function is the gradient of the log likelihood, ∇_x log p(x). Theorems from the 1950s show that for Gaussian noise, denoising is equivalent to learning the score function. So the probabilistic framework (predict noise via ELBO) and the score-based framework (learn the gradient of the log likelihood) arrive at the same place — predicting noise *is* learning the score function. This unifies two bodies of literature spanning decades.

The talk closes with Waseem reflecting on the beauty of these cross-field mathematical connections, and Jeremy reassuring viewers that understanding all this math is not required — the course will cover what's needed gradually over many lessons.

---

- **Lesson Challenges** — None explicitly stated.

- **Potential Research Directions**
  - The original 2015 Sohl-Dickstein et al. paper and how its approach differs from modern implementations
  - The 1950s results on reversibility of Markov processes with Gaussian transitions
  - Connections between the probabilistic (DDPM/ELBO) and score-based (score matching) perspectives
  - Learning the variance in the reverse process (relaxing the DDPM constant-variance assumption)
  - Non-Gaussian transition kernels and whether the same framework extends

- **Homework** — None explicitly assigned.

- **Things Jeremy Says You Should Do**
  - You don't need to understand all the math in this video — it will be covered gradually across the course
  - Check out the main course lesson (Lesson 9)
  - Watch Johno's video for a deeper dive into code and concepts

- **Resources**
  - [Sohl-Dickstein et al. 2015 — "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"](https://arxiv.org/abs/1503.03585)
  - [Ho et al. 2020 — "Denoising Diffusion Probabilistic Models" (DDPM)](https://arxiv.org/abs/2006.11239)
  - Lesson 9 of fast.ai Part 2 (the "magic API" and score function approach)
  - Johno's video (referenced but not linked)
