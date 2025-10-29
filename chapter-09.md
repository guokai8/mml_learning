# Chapter 9: Generative Models for Multimodal Data

---

**Previous**: [Chapter 8: Transformer Architecture](chapter-08.md) | **Next**: [Chapter 10: Seminal Models and Architectures](chapter-10.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand autoregressive generation fundamentals
- Understand diffusion models and their mechanics
- Implement text-conditional image generation
- Compare different generative approaches
- Apply generative models to multimodal tasks
- Handle training challenges in generative models

## 9.1 Autoregressive Generation

### Core Concept

**Definition:**
```
Generate sequences one token at a time
Each token probability conditioned on previous tokens

P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)

Each factor: one conditional probability to learn
Multiply together: joint probability of sequence
```

**Why "autoregressive"?**
```
Auto = self
Regressive = using past values to predict future

Like autoregression in statistics:
  y_t = α + β*y_{t-1} + error

Here:
  x_t ~ Distribution(previous tokens)
  Each token generated using previous tokens
```

**Example - Text generation:**

```
Task: Generate sentence about cats

Step 0: Start with [START] token

Step 1: Predict first word
  Input: [START]
  Model outputs: P(word | [START])
  Distribution: {the: 0.3, a: 0.2, ..., cat: 0.05}
  Sample: "The" (or use greedy: highest probability)

Step 2: Predict second word
  Input: [START] The
  Model outputs: P(word | [START], The)
  Distribution: {cat: 0.4, dog: 0.1, ...}
  Sample: "cat"

Step 3: Predict third word
  Input: [START] The cat
  Model outputs: P(word | [START], The, cat)
  Distribution: {is: 0.5, sat: 0.2, ...}
  Sample: "is"

Continue until [END] token or maximum length

Result: "The cat is sleeping peacefully on the couch"
```

### Decoding Strategies

**Strategy 1: Greedy Decoding**

```
At each step, choose highest probability token

Algorithm:
  for t in 1 to max_length:
    logits = model(previous_tokens)
    next_token = argmax(logits)
    previous_tokens.append(next_token)

Advantages:
  ✓ Fast (single forward pass per step)
  ✓ Deterministic (same output every time)
  ✓ Simple to implement

Disadvantages:
  ✗ Can get stuck in local optima
  ✗ May produce suboptimal sequences
  ✗ "Does not" → "Does" (highest prob) → "not" never chosen
  ✗ No diversity (always same output)

When to use:
  - When consistency matters more than quality
  - Real-time applications where speed critical
  - Baseline comparisons
```

**Strategy 2: Beam Search**

```
Keep track of K best hypotheses
Expand each by one token
Prune to K best

Example with K=3:

Step 1:
  Hypotheses: ["The", "A", "One"]
  Scores: [0.3, 0.2, 0.15]

Step 2 (expand each by one token):
  From "The":
    "The cat" (0.3 × 0.4 = 0.12)
    "The dog" (0.3 × 0.1 = 0.03)
    "The bird" (0.3 × 0.08 = 0.024)

  From "A":
    "A cat" (0.2 × 0.35 = 0.07)
    "A dog" (0.2 × 0.15 = 0.03)
    "A bird" (0.2 × 0.10 = 0.02)

  From "One":
    "One cat" (0.15 × 0.3 = 0.045)
    ...

Step 3 (keep top 3):
  Best: "The cat" (0.12)
  Second: "The dog" (0.03)
  Third: "A cat" (0.07) or "One cat" (0.045)

Continue...

Algorithm:
  hypotheses = [[start_token]]
  scores = [0]

  for t in 1 to max_length:
    candidates = []

    for each hypothesis h in hypotheses:
      logits = model(h)
      for next_token in vocab:
        score = scores[h] + log(logits[next_token])
        candidates.append((h + [next_token], score))

    # Keep best K
    hypotheses, scores = topK(candidates, K)

    # Stop if all ended
    if all ended: break

  return hypotheses[0]  # Best hypothesis

Advantages:
  ✓ Better quality than greedy
  ✓ Still relatively fast
  ✓ Finds better global optimum

Disadvantages:
  ✗ Slower than greedy (K hypotheses tracked)
  ✗ Still deterministic
  ✗ No diversity

When to use:
  - Standard for machine translation
  - When quality important but speed constrained
  - Most common in practice
```

**Strategy 3: Sampling (Temperature-Based)**

```
Instead of greedy, sample from distribution

Algorithm:
  for t in 1 to max_length:
    logits = model(previous_tokens)
    logits = logits / temperature
    probabilities = softmax(logits)
    next_token = sample(probabilities)
    previous_tokens.append(next_token)

Temperature effect:

temperature = 0.1 (cold - sharp):
  Softmax becomes one-hot-like
  Mostly sample highest probability
  Like greedy but with small randomness
  Output: Deterministic

temperature = 1.0 (normal):
  Standard softmax
  Sample according to distribution
  Balanced randomness
  Output: Somewhat random

temperature = 2.0 (hot - smooth):
  Softmax becomes nearly uniform
  All tokens equally likely
  Very random generation
  Output: Very random, often nonsensical

Example:
  Logits: [2.0, 1.0, 0.5]

  Temperature 0.1:
    After scaling: [20, 10, 5]
    After softmax: [0.99, 0.01, 0.0]
    Sample distribution: Mostly first token

  Temperature 1.0:
    After scaling: [2.0, 1.0, 0.5]
    After softmax: [0.66, 0.24, 0.09]
    Sample distribution: Balanced

  Temperature 2.0:
    After scaling: [1.0, 0.5, 0.25]
    After softmax: [0.54, 0.30, 0.15]
    Sample distribution: More uniform

Advantages:
  ✓ Diverse outputs
  ✓ Can be creative
  ✓ Different each time

Disadvantages:
  ✗ Can produce nonsense
  ✗ Quality depends on temperature tuning
  ✗ Slower (need many samples to evaluate)

When to use:
  - Creative tasks (poetry, stories)
  - When diversity valued
  - User-facing applications (less repetitive)
```

**Strategy 4: Top-K Sampling**

```
Only sample from K most probable tokens

Algorithm:
  for t in 1 to max_length:
    logits = model(previous_tokens)

    # Get top K logits
    topk_logits, topk_indices = topk(logits, K)

    # Compute probabilities from only these K
    probabilities = softmax(topk_logits)

    # Sample from this restricted distribution
    next_token_idx = sample(probabilities)
    next_token = topk_indices[next_token_idx]

    previous_tokens.append(next_token)

Example with K=5:

Logits: [5, 4, 3, 1, 0.5, 0.2, 0.1, ...]
Top 5: [5, 4, 3, 1, 0.5]
Softmax of top 5: [0.4, 0.3, 0.2, 0.08, 0.02]

Sample from these 5 tokens only
Never sample from tail tokens
```

**Strategy 5: Top-P (Nucleus) Sampling**

```
Sample from smallest set of tokens with cumulative probability > p

Algorithm:
  for t in 1 to max_length:
    logits = model(previous_tokens)
    probabilities = softmax(logits)

    # Sort by probability descending
    sorted_probs = sort(probabilities, descending=True)

    # Find cutoff
    cumsum = cumsum(sorted_probs)
    cutoff_idx = first index where cumsum > p

    # Keep tokens up to cutoff
    mask = cumsum <= p

    # Renormalize and sample
    filtered_probs = probabilities * mask
    filtered_probs = filtered_probs / sum(filtered_probs)

    next_token = sample(filtered_probs)
    previous_tokens.append(next_token)

Example with p=0.9:

Probabilities: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
Cumsum: [0.5, 0.8, 0.9, 0.95, 0.98, 1.0]

Keep tokens where cumsum <= 0.9:
  [0.5, 0.3, 0.1] with cumsum [0.5, 0.8, 0.9]

Sample from these three tokens
Never sample from last three (low probability)
```

### Training Autoregressive Models

**Training objective:**

```
Goal: Maximize probability of correct sequence

For sequence [w₁, w₂, w₃, w₄]:

Loss = -log P(w₁, w₂, w₃, w₄)
     = -log [P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × P(w₄|w₁,w₂,w₃)]
     = -[log P(w₁) + log P(w₂|w₁) + log P(w₃|w₁,w₂) + log P(w₄|w₁,w₂,w₃)]

Each term: Cross-entropy loss for predicting next token

Total loss = Sum of cross-entropy losses for each position

Gradient flows to each position
All trained simultaneously (efficient!)
```

**Teacher forcing:**

```
During training:
  Use true tokens for context (not predicted tokens)

Without teacher forcing:
  Step 1: Predict w₂ from w₁ (could be wrong)
  Step 2: Predict w₃ from (w₁, [predicted w₂]) (error accumulates)
  Step 3: Predict w₄ from (w₁, [predicted w₂], [predicted w₃]) (more errors)

Result: Model learns on error distribution
        Model overfits to teacher forcing
        At test time, predicted tokens are different!

With teacher forcing:
  Step 1: Predict w₂ from w₁ (true)
  Step 2: Predict w₃ from (w₁, w₂) (true)
  Step 3: Predict w₄ from (w₁, w₂, w₃) (true)

Result: Clean training signal
        But distribution mismatch at test time!

Solution: Scheduled sampling
  Start with teacher forcing
  Gradually use predicted tokens during training
  Mix of training and test distribution
```

**Implementation:**

```python
def train_autoregressive(model, sequences, optimizer, device):
    """Train autoregressive model with teacher forcing"""
    model.train()
    total_loss = 0

    for sequence in sequences:
        sequence = sequence.to(device)  # (seq_len,)

        # Input: all but last token
        input_ids = sequence[:-1]  # (seq_len-1,)

        # Target: all but first token
        target_ids = sequence[1:]  # (seq_len-1,)

        # Forward pass
        logits = model(input_ids)  # (seq_len-1, vocab_size)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(sequences)
```

## 9.2 Diffusion Models

### Core Idea

**The diffusion process (forward):**

```
Start with clean image
Add noise gradually
After many steps: Pure noise

Image → slightly noisy → more noisy → ... → pure noise

Reverse process (learning):
Pure noise → slightly less noisy → ... → clean image

If we learn reverse process:
  Can generate images from noise!
  noise → network → slightly clean → network → ... → image
```

**Why this works:**

```
Traditional approach:
  Learn complex distribution directly
  High-dimensional, multi-modal distribution
  Hard!

Diffusion approach:
  Learn simple steps: noise → slightly cleaner
  Each step: Small denoising
  Accumulate small steps: noise → image
  Each step easier to learn!

Analogy:
  Hard: Draw perfect portrait in one step
  Easy: Start with sketch, refine step-by-step
       Each refinement small improvement
       Final result: Beautiful portrait
```

### Forward Process (Diffusion)

**Markov chain:**

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

Interpretation:
  Take previous x_{t-1}
  Scale by √(1-β_t) (slightly shrink)
  Add Gaussian noise with variance β_t
  Result: x_t

β_t is variance schedule
  Usually small: 0.0001 to 0.02
  Controls how much noise added

  Small β_t: Small change (smooth)
  Large β_t: Big change (abrupt)
```

**Closed form solution:**

```
Instead of T sequential steps, compute directly:

q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1-ᾱ_t) I)

where ᾱ_t = ∏_{s=1}^t (1-β_s)

Benefit:
  Sample x_t directly from x_0 and noise
  Don't need to compute all intermediate steps
  Fast training!

Formula:
  x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε

  where ε ~ N(0, I) is Gaussian noise

Properties:
  At t=0: ᾱ_0 = 1
    x_0 = 1 * x_0 + 0 * ε = x_0 (clean image)

  At t=T: ᾱ_T ≈ 0
    x_T ≈ 0 * x_0 + 1 * ε = ε (pure noise)

  Intermediate: ᾱ_t ∈ (0, 1)
    Mix of original and noise
```

**Visualization:**

```
Clean image ────→ Slight noise ────→ More noise ────→ Pure noise
  x_0                x_100              x_500            x_1000

ᾱ_t = 1.0          ᾱ_t ≈ 0.9          ᾱ_t ≈ 0.3         ᾱ_t ≈ 0.001

[Clear cat]  →  [Slightly fuzzy]  →  [Grainy]  →  [Random pixels]
```

### Reverse Process (Denoising)

**Learning the reverse:**

```
Forward: q(x_t | x_{t-1})  [given by math]
Reverse: p_θ(x_{t-1} | x_t)  [learn with network!]

Network predicts:
  Given noisy image x_t
  Predict slightly less noisy image x_{t-1}

Training:
  Use forward process to create noisy versions
  Train network to denoise
  Loss: How close is predicted to true x_{t-1}
```

**Equivalent formulation - Noise prediction:**

```
Instead of predicting x_{t-1}, predict noise:

x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε

Rearrange:
  ε = (x_t - √(ᾱ_t) x_0) / √(1-ᾱ_t)

Network learns: ε_θ(x_t, t)
  Given: x_t (noisy image) and t (timestep)
  Predict: ε (noise that was added)

Then:
  x_{t-1} = (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √(1-β_t)

Benefit:
  Network predicts smaller values (noise)
  Easier to learn than predicting full image
  More stable training
```

**Training loss:**

```
For each training image x_0:
  1. Sample random timestep t
  2. Sample random noise ε ~ N(0, I)
  3. Create noisy version: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
  4. Predict noise: ε_pred = ε_θ(x_t, t)
  5. Loss: ||ε_pred - ε||²

Intuition:
  Network learns to predict noise
  For any timestep
  For any noise level
  From corresponding noisy image
```

### Sampling (Generation)

**Iterative denoising:**

```
Start: x_T ~ N(0, I)  (pure random noise)

For t from T down to 1:
  ε_pred = ε_θ(x_t, t)  (network predicts noise)

  x_{t-1} = (x_t - √(1-ᾱ_t) ε_pred) / √(1-β_t)

  Add small noise (for stochasticity):
  x_{t-1} = x_{t-1} + √(β_t) z
  where z ~ N(0, I)

Result: x_0 is generated image
```

**Why this works:**

```
Step 1: x_1000 = pure noise
Step 2: Apply denoising step → x_999 (slightly cleaner)
Step 3: Apply denoising step → x_998 (more refined)
...
Step 1000: Apply denoising step → x_0 (clean image!)

Each step removes some noise
1000 small improvements → coherent image
```

**Scaling - How many steps?**

```
More steps = better quality but slower

T = 50:   Fast, okay quality
T = 100:  Standard, good quality
T = 1000: Very good quality, slow

In practice:
  Train with T = 1000 (for learning)
  Can sample with smaller T (faster, slightly worse)
  DDIM: Sample in 50 steps instead of 1000
```

### Conditional Diffusion

**Adding text conditioning:**

```
Standard diffusion:
  ε_θ(x_t, t) predicts noise
  Only input: noisy image, timestep
  Output: unconditioned noise prediction

Text-conditioned:
  ε_θ(x_t, t, c) predicts noise
  Inputs: noisy image, timestep, text embedding c
  Output: text-aware noise prediction

Training:
  1. Sample image x_0 and text description c
  2. Encode text: c = text_encoder(c)  (768D)
  3. Sample timestep t and noise ε
  4. Noisy image: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
  5. Network prediction: ε_pred = ε_θ(x_t, t, c)
  6. Loss: ||ε_pred - ε||²

Effect:
  Network learns text-image alignment
  During denoising, follows text guidance
  Generated image matches description
```

**Cross-attention for conditioning:**

```
Network architecture:

Input x_t:
  ├─ CNN layers (process noisy image)
  │  └─ Feature maps
  │       ├─ Self-attention (refine image understanding)
  │       │
  │       └─ Cross-attention to text
  │           Query: image features
  │           Key/Value: text embeddings
  │           ↓ Result: Image attends to relevant text

Text embedding c:
  └─ Project to key/value space
```

**Classifier-free guidance:**

```
Problem: Guidance strength vs diversity trade-off

Solution: Predict both conditioned and unconditioned

During training:
  Some batches: Predict with text (conditioned)
  Some batches: Predict without text (unconditioned)

  Network learns both paths

During sampling:
  Compute both predictions:
    ε_cond = ε_θ(x_t, t, c)      (with text)
    ε_uncond = ε_θ(x_t, t, None) (without text)

  Interpolate with guidance scale w:
    ε_final = ε_uncond + w * (ε_cond - ε_uncond)

Interpretation:
  w=0: Ignore text, purely random
  w=1: Normal, follow text
  w=7: Strong guidance, adhere closely to text
  w=15: Extreme guidance, saturated colors, distorted

Trade-off:
  w=1:  High diversity, moderate text adherence
  w=7:  Good balance
  w=15: Low diversity, extreme text adherence

Sweet spot: Usually w ∈ [7, 15]
```

### Stable Diffusion Architecture

**Full pipeline:**

```
Text prompt: "A red cat on a chair"
    ↓
Text encoder (CLIP):
  "A red cat on a chair" → 77×768 embeddings
    ↓
Diffusion model:
  Input: noise (H×W×4) from VAE latent space
         timestep t
         text embeddings (77×768)

  Processing:
    ① ResNet blocks (noise processing)
    ② Self-attention (within image)
    ③ Cross-attention to text
    ④ Repeat 12 times

  Output: Predicted noise
    ↓
Denoising loop (1000 steps):
  For each step:
    ① Input current noisy latent
    ② Network predicts noise
    ③ Denoise: x_{t-1} = denoise(x_t, prediction)
    ④ Next step
    ↓
Latent space representation of clean image
    ↓
VAE decoder:
  4D latent → 512×512×3 RGB image
    ↓
Image: Red cat on chair!
```

**Why VAE compression?**

```
Diffusion on high-res images:
  512×512×3 = 786,432 dimensions
  Computationally infeasible!

Solution: VAE compression
  512×512×3 image → 64×64×4 latent
  ~100× compression!

  Latent captures semantic information
  Pixels details discarded

Benefit:
  ① Faster computation
  ② Diffusion on semantics, not pixels
  ③ Better scaling
```

## 9.3 Text-Conditional Image Generation

### Dataset Requirements

**For training text-to-image models:**

```
Billions of image-caption pairs needed:

LAION dataset: 5.8 billion pairs
  Collected from web
  Uncurated, noisy
  Large diversity
  ↓ Used for Stable Diffusion

Conceptual Captions: 3.3M pairs
  More curated than LAION
  Better quality
  Smaller

For fine-tuning: 10K-100K pairs often sufficient
For training from scratch: Billions needed
```

**Data quality considerations:**

```
Good pairs:
  Image of red car
  Caption: "A shiny red sports car"

Bad pairs (but exist in web data):
  Image of red car
  Caption: "Why cars are important"
  (Not descriptive of image)

Impact:
  Model learns incorrect alignments
  Generates wrong things from descriptions

Solution:
  Filter low-quality pairs
  Use robust training (contrastive pre-training helps)
  Ensure at least 80% correct pairs
```

### Training Process

**Step 1: Pre-training (Image-Text Alignment)**

```
Before training diffusion, learn text-image alignment

Method: CLIP-style contrastive learning

Dataset: 400M+ image-caption pairs
Loss: Make matched pairs similar in embedding space

Result:
  Text encoder learns to encode descriptions meaningfully
  Image features align with text
  Diffusion can then learn from well-aligned signal
```

**Step 2: Diffusion Model Training**

```
Start: Noisy latent z_t
Timestep: t (1 to 1000)
Condition: Text embedding c

Network learns:
  Given z_t and c, predict noise

Loss function:
  L = ||ε - ε_θ(z_t, t, c)||²

Training:
  Batch size: 256-4096 (huge!)
  Learning rate: 1e-4
  Optimizer: Adam or AdamW
  Duration: Days to weeks on large GPU clusters

  Example:
    4 clusters, 8 GPUs each
    32 V100 GPUs total
    Training for 2 weeks
    Cost: ~$100K in compute
```

**Step 3: Fine-tuning (Optional)**

```
Pre-trained model trained on billions of pairs
General knowledge of image generation

Fine-tune on specific domain:

Domain: Medical imaging
  1. Take pre-trained Stable Diffusion
  2. Add new layers for medical images
  3. Train on 10K medical image-description pairs
  4. 1-2 days training on single GPU
  5. Result: Medical image generation model

Other domains:
  - Anime art
  - Product design
  - Fashion
  - Architecture
```

### Inference Tricks

**Latent space optimization:**

```
Instead of denoising from random noise,
optimize noise latent directly

Process:
  1. Encode target image to latent z
  2. Add timestep t noise: z_t = noise_t(z)
  3. Denoise from z_t
  4. Result: Image similar to target but modified per text

Use case: Inpainting (fill in regions)
```

**Negative prompts:**

```
Text prompt: "A beautiful cat"
Negative prompt: "ugly, blurry, deformed"

Effect:
  Network learns what NOT to generate
  Classifier-free guidance applied to both

  ε_final = ε_uncond + w * (ε_cond - ε_uncond)
            - w_neg * (ε_neg - ε_uncond)

Benefit:
  More control over generation
  Avoid common artifacts
```

**Multi-step refinement:**

```
Step 1: Generate image with text
  Prompt: "A cat"
  Result: Generic cat

Step 2: Inpaint to add details
  Prompt: "A red cat"
  Mask: Cat region
  Result: Red cat

Step 3: Upscale
  Use super-resolution model
  Result: High-res red cat

Benefits:
  ① Progressive refinement
  ② More control
  ③ Better results than single step
```

## 9.4 Practical Generative Systems

### Building Text-to-Image System

```python
import torch
from diffusers import StableDiffusionPipeline

class TextToImageGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-2"):
        # Load pre-trained model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")

    def generate(self, prompt, num_images=1, guidance_scale=7.5,
                 steps=50, seed=None):
        """
        Generate images from text prompt

        Args:
            prompt: Text description
            num_images: Number of images to generate
            guidance_scale: How much to follow prompt (7.5 is default)
            steps: Number of denoising steps (more = better quality but slower)
            seed: Random seed for reproducibility

        Returns:
            images: List of PIL Images
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Generate
        output = self.pipe(
            prompt=prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )

        return output.images

    def generate_with_negative(self, prompt, negative_prompt="",
                               guidance_scale=7.5, steps=50):
        """Generate with negative prompt to avoid artifacts"""
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )
        return output.images

    def inpaint(self, image, mask, prompt, guidance_scale=7.5, steps=50):
        """
        Inpaint: modify specific regions of image

        Args:
            image: PIL Image to modify
            mask: Binary mask (white = inpaint region)
            prompt: Text description of what to generate

        Returns:
            Modified image
        """
        from diffusers import StableDiffusionInpaintPipeline

        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpaint",
            torch_dtype=torch.float16
        )
        inpaint_pipe = inpaint_pipe.to("cuda")

        output = inpaint_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )

        return output.images[0]

# Usage
generator = TextToImageGenerator()

# Simple generation
images = generator.generate("A beautiful sunset over mountains")

# With negative prompt to improve quality
images = generator.generate(
    prompt="A realistic portrait of a woman",
    negative_prompt="ugly, blurry, deformed",
    guidance_scale=10.0,
    steps=50
)

# Save
images[0].save("generated_image.png")
```

### Building Image-Text Model for Understanding

```python
class ImageCaptioningModel(nn.Module):
    """Generate captions from images"""

    def __init__(self, image_encoder, text_decoder, embedding_dim=256):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_decoder = text_decoder
        self.projection = nn.Linear(2048, embedding_dim)

    def forward(self, images, text_ids=None):
        """
        Args:
            images: Batch of

-----

> continue

images
            text_ids: Optional, for training

        Returns:
            logits or loss
        """
        # Encode images
        image_features = self.image_encoder(images)  # (batch, 2048)
        image_embeddings = self.projection(image_features)  # (batch, 256)

        if text_ids is None:
            # Inference mode
            return image_embeddings
        else:
            # Training mode
            logits = self.text_decoder(
                image_embeddings=image_embeddings,
                input_ids=text_ids
            )
            return logits

    def generate_caption(self, image, max_length=50, temperature=0.7):
        """Generate caption for image"""
        self.eval()

        with torch.no_grad():
            # Encode image
            image_features = self.image_encoder(image.unsqueeze(0))
            image_embeddings = self.projection(image_features)

            # Start with [CLS] token
            caption_ids = [tokenizer.cls_token_id]

            # Generate tokens
            for _ in range(max_length):
                # Predict next token
                logits = self.text_decoder.predict_next(
                    image_embeddings,
                    torch.tensor([caption_ids]).to(device)
                )
                logits = logits[:, -1, :] / temperature

                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                caption_ids.append(next_token.item())

                # Stop on [SEP] token
                if next_token.item() == tokenizer.sep_token_id:
                    break

        # Decode to text
        caption = tokenizer.decode(caption_ids)
        return caption

def train_captioning_model(model, train_loader, optimizer, device, epochs=10):
    """Train image captioning model"""
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            caption_ids = batch['caption_ids'].to(device)

            # Forward pass
            logits = model(images, caption_ids)

            # Reshape for loss
            logits = logits.view(-1, vocab_size)
            targets = caption_ids[:, 1:].contiguous().view(-1)

            # Compute loss
            loss = criterion(logits, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```

### Handling Generation Failures

**Problem 1: Mode collapse (generating same thing)**

```
Symptoms:
  All outputs identical or very similar
  Low diversity

Causes:
  Temperature too low
  Batch size too small
  Insufficient diversity in training data

Solutions:
  Increase temperature (0.7 → 0.9)
  Use top-p sampling (not greedy)
  Increase batch size
  Data augmentation

Code:
  # Low temperature (bad)
  next_token = argmax(logits)

  # Better: Use temperature
  logits = logits / temperature
  probs = softmax(logits)
  next_token = sample(probs)

  # Best: Top-p sampling
  probs = softmax(logits)
  probs = top_p_filter(probs, p=0.9)
  next_token = sample(probs)
```

**Problem 2: Generating nonsense**

```
Symptoms:
  Output doesn't match prompt
  Incoherent sequences
  Missing objects from description

Causes:
  Insufficient text conditioning strength
  Poor text encoder
  Text alignment not learned well

Solutions:
  Increase guidance scale (7 → 10 or 15)
  Pre-train text encoder more (CLIP)
  Use stronger conditioning

Example - Diffusion models:
  # Weak guidance
  guidance_scale = 1.0
  Result: ~50% follow prompt

  # Standard guidance
  guidance_scale = 7.5
  Result: ~80% follow prompt

  # Strong guidance
  guidance_scale = 15.0
  Result: ~95% follow prompt, but less diversity
```

**Problem 3: Slow generation**

```
Symptoms:
  Takes minutes per image
  Not practical for deployment

Causes:
  Too many denoising steps (1000 default)
  Inefficient implementation
  No GPU acceleration

Solutions:
  Reduce inference steps (1000 → 50)
  Use distilled model (faster but lower quality)
  Use DDIM sampler (faster convergence)
  Batch generation (process multiple at once)

Performance trade-off:

  Steps    Quality    Time
  ─────────────────────────
   10      Poor       10ms
   20      Okay       50ms
   50      Good       200ms (Stable Diffusion standard)
  100      Very good  400ms
 1000      Best       4000ms (training standard)

For production: 50 steps usually sufficient
```

## 9.5 Comparing Generative Approaches

**Autoregressive vs Diffusion:**

```
                  Autoregressive    Diffusion
────────────────────────────────────────────────
Output quality    Good              Excellent
Training time     Moderate          Very long
Inference steps   100-1000          50-1000
Inference speed   Moderate          Slower
Diversity         High              Moderate
Training simplicity Easier          Harder
                  (language model)  (complex process)

When to use:
  Autoregressive: Text generation, fast inference needed
  Diffusion: High-quality images, time not critical
```

**Generating text vs images:**

```
TEXT GENERATION (Autoregressive):
  ✓ Fast inference (greedy decoding)
  ✓ Easy to understand (token by token)
  ✓ Works well with beam search
  ✗ Can repeat or get stuck

Use: Chatbots, summarization, translation

IMAGE GENERATION (Diffusion):
  ✓ High quality with text control
  ✓ Flexible (can do inpainting, editing)
  ✗ Slow (many denoising steps)

Use: Art, design, content creation
```

---

