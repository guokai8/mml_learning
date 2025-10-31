# Chapter 7: Contrastive Learning

---

**Previous**: [Chapter 6: Attention Mechanisms in Multimodal Systems](chapter-06.md) | **Next**: [Chapter 8: Transformer Architecture](chapter-08.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand contrastive learning principles and motivation
- Implement InfoNCE loss
- Understand CLIP's revolutionary approach
- Compare different contrastive methods
- Apply contrastive learning to your own problems

## 7.1 The Problem Contrastive Learning Solves

### Traditional Supervised Learning

**Standard approach:**

```
Training data: (input, label) pairs

Task: Image classification
  Input: Image
  Label: "cat" or "dog"

  Process:
  ① Pass image through network
  ② Output logits for each class
  ③ Cross-entropy loss compares to label
  ④ Backprop updates weights

Requirements:
  ✗ Requires labels for everything
  ✗ Labels are expensive (human annotation)
  ✗ Limited to labeled dataset size
  ✗ New task = new labeled data needed
```

**Bottleneck in practice:**

```
Problem: Most data is unlabeled

Example:
  ImageNet: 1.4M labeled images
  Internet: Billions of images daily

  Ratio: ~1 labeled per 1 million unlabeled!

Question: How to leverage the vast unlabeled data?

Traditional supervised learning: Can't use it!
Solution: Contrastive learning
```

### Self-Supervised Learning Intuition

**Key insight:**
```
Don't need explicit labels!
Create labels from data itself using natural structure
```

**Example - Image rotation prediction:**

```
Unlabeled image:
  [Photo of cat]

Create self-supervised task:
  Rotate image 90°

  Rotated image → Network → Predict rotation

Label is free! (We created it by rotation)

Training:
  ① Rotate image by random angle (0°, 90°, 180°, 270°)
  ② Network predicts angle
  ③ Loss: Cross-entropy between predicted and actual angle

Result:
  Network learns visual representations
  Without any human labels!

Benefit:
  Can train on billions of unlabeled images
  Representations useful for downstream tasks
  Transfer to real tasks with small labeled dataset
```

**Why this works:**

```
To predict rotation, network must understand:
  - What's the "up" direction? (spatial orientation)
  - What are objects and their structure? (semantics)
  - What's foreground vs background? (attention)

These are useful representations for other tasks!
```

### Contrastive Learning Idea

**Core concept:**

```
Supervised learning: "Is this input A or B or C?"
Contrastive learning: "Which B is similar to A?"

Example:
  Supervised:      "Is this a dog?" (Yes/No)
  Contrastive:     "Given this dog photo, which text matches best?
                    A) 'A dog running'
                    B) 'A cat sleeping'
                    C) 'A car parked'"

Contrastive doesn't need explicit labels
Just needs relative similarities!
```

**Why it's powerful:**

```
Advantage 1: No labels needed
  ✓ Use unlabeled data directly
  ✓ Billions of image-text pairs from web
  ✓ Much cheaper than labeling

Advantage 2: Richer signal
  Binary classification: Yes/No (1 bit)
  Contrastive: Ranking among many (log₂(N) bits)

  With N=1000 options:
  Ranking gives ~10 bits of information
  vs 1 bit for binary

Advantage 3: Metric learning
  Directly optimize for similarity
  Better representations for retrieval
  Natural distance metrics
```

## 7.2 InfoNCE Loss - The Foundation

### Understanding the Loss

**Name breakdown:**
- **Info** = Information theory
- **NCE** = Noise Contrastive Estimation

**Goal:**
```
Make positive pairs similar
Make negative pairs dissimilar

Positive pair: (cat image, "cat" text)
Negative pair: (cat image, "dog" text)
```

### Mathematical Formulation

**Formula:**

```
L = -log [ exp(sim(q,k+)/τ) / (exp(sim(q,k+)/τ) + Σⱼ exp(sim(q,k⁻ⱼ)/τ)) ]

Breakdown:

q = query (e.g., image)
k+ = positive key (e.g., matching text)
k⁻ⱼ = negative keys (non-matching texts)
τ = temperature (controls sharpness)
sim = similarity function (cosine, dot product)
```

**Step-by-step explanation:**

```
Step 1: Compute similarities
  sim(query, positive) = dot product
  sim(query, negative₁) = dot product
  sim(query, negative₂) = dot product
  ...

  Result: Scores (could be any value)

Step 2: Scale by temperature
  Score / τ

  Temperature effect:
    τ small (0.01): Scores become extreme
    τ normal (0.1): Moderate scaling
    τ large (1.0): Minimal scaling

  Why temperature?
    Prevents softmax from being too sharp
    Allows gradient flow during training

Step 3: Exponential
  exp(score / τ)

  Result: All positive (e^x > 0 for all x)

  Effect:
    Larger scores → larger exponents
    Softmax then emphasizes them

Step 4: Softmax (normalize)
  exp(positive) / (exp(positive) + Σ exp(negatives))

  Result: Probability in [0, 1]

  Interpretation:
    Probability that positive is highest ranked
    Perfect: Probability = 1.0
    Random: Probability = 1/(1+num_negatives)

Step 5: Negative log
  -log(probability)

  If probability = 1.0: loss = 0 (perfect!)
  If probability = 0.1: loss = -log(0.1) = 2.3 (bad)
  If probability = 0.5: loss = -log(0.5) = 0.69 (medium)
```

### Numerical Example

**Setup:**

```
Query: Image of red cat
Positive: Text "a red cat"
Negatives:
  - "a blue dog"
  - "a green parrot"
  - "a car"

Similarities (before temperature):
  sim(query, positive) = 0.8    (high, should be!)
  sim(query, neg1) = 0.2        (low, good)
  sim(query, neg2) = 0.15       (low, good)
  sim(query, neg3) = 0.1        (low, good)

Temperature τ = 0.1
```

**Computing loss:**

```
Step 1: Scale by temperature
  0.8 / 0.1 = 8.0
  0.2 / 0.1 = 2.0
  0.15 / 0.1 = 1.5
  0.1 / 0.1 = 1.0

Step 2: Exponentials
  e^8.0 ≈ 2981
  e^2.0 ≈ 7.4
  e^1.5 ≈ 4.5
  e^1.0 ≈ 2.7

Step 3: Softmax (probability)
  2981 / (2981 + 7.4 + 4.5 + 2.7)
  = 2981 / 2995.6
  ≈ 0.995   (99.5% probability positive is best!)

Step 4: Loss
  loss = -log(0.995) ≈ 0.005   (very small! Model doing great)
```

**What if model was bad:**

```
Similarities:
  sim(query, positive) = 0.1    (low! bad!)
  sim(query, neg1) = 0.5        (high! worse)
  sim(query, neg2) = 0.4
  sim(query, neg3) = 0.3

After temperature scaling (τ = 0.1):
  0.1 / 0.1 = 1.0     → e^1.0 ≈ 2.7
  0.5 / 0.1 = 5.0     → e^5.0 ≈ 148
  0.4 / 0.1 = 4.0     → e^4.0 ≈ 55
  0.3 / 0.1 = 3.0     → e^3.0 ≈ 20

Softmax:
  2.7 / (2.7 + 148 + 55 + 20)
  = 2.7 / 225.7
  ≈ 0.012   (1.2% probability - terrible!)

Loss:
  -log(0.012) ≈ 4.4   (very large! Forces update)
```

### Why This Works

**Mathematical properties:**

```
1. Bounded between 0 and log(1+N)
   where N = number of negatives

   N=10: Loss ∈ [0, log(11) ≈ 2.4]
   N=100: Loss ∈ [0, log(101) ≈ 4.6]

   Interpretable scale

2. Gradient is informative

   Perfect case (prob ≈ 1): gradient ≈ 0
   Good case (prob ≈ 0.9): gradient ≈ small
   Bad case (prob ≈ 0.1): gradient ≈ large

   Automatically focuses on hard cases

3. Invariant to scale

   If all similarities multiplied by constant K:
   exp(K*sim) has same relative ordering
   Softmax still works correctly

   Enables using unnormalized similarities
```

### Temperature Parameter

**Role of τ:**

```
Temperature controls softmax sharpness

τ = 0.01 (very cold):
  Softmax becomes nearly one-hot
  exp(5) = 148
  exp(4) = 55
  exp(3) = 20
  Ratio: 148/55 = 2.7x difference

  Large differences between outputs
  Large gradients
  Potential instability

τ = 0.1 (standard):
  Moderate softmax
  exp(0.5) = 1.65
  exp(0.4) = 1.49
  exp(0.3) = 1.35
  Ratio: 1.65/1.49 = 1.1x difference

  Balanced gradients
  Stable training
  Common choice

τ = 1.0 (very hot):
  Softmax becomes smooth
  exp(0.05) = 1.05
  exp(0.04) = 1.04
  exp(0.03) = 1.03
  Ratio: 1.05/1.04 ≈ 1.01x difference

  Small differences between outputs
  Small gradients
  Slow learning

τ = 10.0 (extremely hot):
  Softmax nearly uniform
  All classes almost equally likely
  Nearly no signal
  Training doesn't work
```

**Effect on learning:**

```
Optimal temperature depends on:
  - Number of negatives
  - Difficulty of task
  - Data quality

Typical range: τ ∈ [0.05, 0.2]

CLIP uses: τ ≈ 0.07 (learned during training)
```

## 7.3 CLIP - Contrastive Learning Success Story

### Context and Impact

**Problem statement (2020):**

```
Existing vision models:
  - Trained on ImageNet (1.4M images)
  - Limited to 1000 classes
  - Can't generalize to new concepts
  - Require supervised fine-tuning

Question:
  Can we use web data (unsupervised) for vision?
  Can we match NLP's success with massive unlabeled data?
```

**CLIP solution:**

```
Data: 400M image-caption pairs from web
Task: Learn from natural language supervision
Method: Contrastive learning on image-text pairs

Result: Revolutionary zero-shot transfer
```

### CLIP Architecture

**Components:**

```
Image encoder:           Text encoder:
  Vision Transformer      Transformer (BERT-like)
  Input: 224×224 image    Input: Text tokens
  Output: 512D vector     Output: 512D vector

            ↓                     ↓

    [Normalize to unit length]

            ↓                     ↓

    Similarity computation (dot product of normalized)

            ↓

    Contrastive loss
```

**Data collection:**

```
400 million image-caption pairs from internet

Sources:
  - Web pages with images and captions
  - Publicly available image databases
  - Social media posts with text
  - Stock photo sites with descriptions

Quality:
  - Uncurated and diverse
  - Contains noise and biases
  - Reflects web distribution
  - Natural language (not formal labels)
```

### Training Process

**Batch construction:**

```
Batch size: 32,768 (massive!)

Images: [img_1, img_2, ..., img_32k]
Captions: [caption_1, caption_2, ..., caption_32k]

Encode all:
  Image embeddings: 32k × 512
  Caption embeddings: 32k × 512

Compute similarity matrix (32k × 32k):
  sim[i,j] = image_i · caption_j

Goal:
  Diagonal elements high (matched pairs)
  Off-diagonal elements low (mismatched pairs)
```

**Loss computation:**

```
For each image:
  Compute InfoNCE loss
  Positive: matching caption
  Negatives: all other 32k-1 captions

For each caption:
  Compute InfoNCE loss
  Positive: matching image
  Negatives: all other 32k-1 images

Total loss = average of all losses

Optimization:
  Adam optimizer
  Learning rate: 5×10⁻⁴
  Training: ~2 weeks on large clusters
```

### Zero-Shot Transfer - Revolutionary Capability

**Traditional approach:**

```
New task: Classify images of birds (not in ImageNet)

Steps:
  1. Get labeled training data for birds
  2. Fine-tune ImageNet model
  3. Get predictions

Problem: Need labeled bird data!
Cost: Expensive annotation
```

**CLIP zero-shot approach:**

```
New task: Classify images of birds

No training needed!

Steps:
  1. Text prompts: "a photo of a bird"
                   "a photo of a person"
                   "a photo of a car"

  2. Encode each prompt with CLIP text encoder
     → 512D vectors

  3. For test image:
     - Encode with CLIP image encoder
     - Compute similarity to each prompt
     - Select highest similarity

  4. Done! Zero-shot classification

Example:
  Image similarity scores:
    "a photo of a bird": 0.92    ← Highest
    "a photo of a person": 0.15
    "a photo of a car": 0.08

  Prediction: Bird
```

**Why it works:**

```
CLIP trained on 400M diverse image-caption pairs
Learned that:
  - Images with birds cluster with "bird" text
  - Images with people cluster with "person" text
  - Images with cars cluster with "car" text

These mappings generalize to new images!

Transfer learning without fine-tuning:
  - No labeled data needed
  - No training required
  - Immediate deployment
```

### Benchmark Results

**Zero-shot transfer (ImageNet classification):**

```
Traditional supervised:
  ResNet-50: 76.1% accuracy

CLIP zero-shot:
  CLIP-ViT-L/14: 62.8% accuracy

Seems lower, BUT:
  - CLIP trained on NO labeled images
  - Just 400M raw internet data
  - Immediately applicable to any category
  - ResNet trained with 1.4M labeled ImageNet

Adjusted for training data:
  ResNet: 76.1% on specific dataset
  CLIP: 62.8% on ANY dataset (zero-shot)

  CLIP more generalizable!
```

**After fine-tuning on small labeled sets:**

```
ImageNet (1% labeled):
  CLIP: 76.3% accuracy

Comparison:
  - CLIP fine-tuned with 1% labels ≈ ResNet with 100% labels
  - 100× more data-efficient!
  - Shows power of pre-training
```

**Other domains:**

```
Transfer to new datasets:

STL10 (airplane, bird, car, etc.):
  CLIP: 92.9% zero-shot

Food101 (food classification):
  CLIP: 88.3% zero-shot

EuroSAT (satellite imagery):
  CLIP: 58.4% zero-shot

Works across diverse domains!
```

### Why CLIP is Revolutionary

**1. Scale:**
```
400M image-text pairs >> 1.4M ImageNet
Shows power of scale in representation learning
Unlabeled data is abundant!
```

**2. Natural supervision:**
```
Language is natural way to describe images
Not forced to 1000 classes like ImageNet
Flexible descriptors
Can specify any attribute
```

**3. Zero-shot transfer:**
```
No fine-tuning needed
Immediate deployment
No labeled data required
Generalizes across domains
```

**4. Open-ended prediction:**
```
Not limited to predefined classes
Can describe images with any text
"A cat wearing a hat"
"A red car on a mountain"
Any description works!
```

### Impact on Field

```
CLIP (April 2021) was watershed moment

Before CLIP:
  - Supervised learning paradigm dominant
  - Limited to ImageNet 1000 classes
  - Required labeled data for new tasks
  - Struggled on out-of-distribution data

After CLIP:
  - Contrastive learning became mainstream
  - Foundation model era began
  - Zero-shot transfer became practical
  - Industry adopted language-grounded vision

Inspired:
  - ALIGN (Google)
  - LiT (Google)
  - COCA (Meta)
  - Flamingo (DeepMind)
  - BLIP (Salesforce)
  - Many others...
```

## 7.4 Variants and Extensions of Contrastive Learning

### Method 1: SimCLR - Self-Supervised Vision

**Motivation:**
```
CLIP uses text for supervision
What if we only have unlabeled images?

Answer: Use image augmentations as "supervision"
```

**Core idea:**

```
Single image:
  [Original cat photo]

Create two augmented versions:
  [Rotated, cropped, color-adjusted]
  [Different rotation, crop, colors]

Treat as positive pair:
  Both should have similar representations
  (Same cat, different augmentations)

Negatives:
  Other images in batch

Loss: Make augmentations similar,
      other images dissimilar
```

**Process:**

```
1. Sample image x from dataset

2. Create two augmented versions:
   x_i = Aug(x)  (augmentation 1)
   x_j = Aug(x)  (augmentation 2)

   Different random augmentations!

3. Encode both through network f:
   h_i = f(x_i)
   h_j = f(x_j)

4. Project to embedding space:
   z_i = g(h_i)
   z_j = g(h_j)

5. Contrastive loss:
   sim(z_i, z_j) should be high
   sim(z_i, z_k) should be low (for k ≠ i,j)

6. Backprop updates f and g
```

**Key insights:**

```
Why this works:

Assumptions:
  1. Augmentations preserve content
  2. Different images are different

Implications:
  Model learns representations that:
  - Survive augmentations (robust features)
  - Differ between images (discriminative features)
  - Capture semantic content (not style)

Result:
  Representations useful for downstream tasks
  Without any labels!
```

**Augmentations used:**

```
Strong augmentations needed for self-supervised learning:

Random crop:
  (up to 85% crop)
  ↑ Forces learning of part representations

Color jittering:
  Brightness, contrast, saturation, hue
  ↑ Prevents learning from color only

Gaussian blur:
  Blurs fine details
  ↑ Forces learning of structure, not pixels

Random grayscale:
  Removes color information
  ↑ Forces learning of shape and texture

Gaussian noise:
  Adds random noise
  ↑ Makes features robust

Note: Extreme augmentations avoid (would destroy content)
  - Extreme rotation: Flips meaning
  - Extreme scaling: Makes object invisible
  - Extreme distortion: No longer recognizable
```

**Differences from CLIP:**

```
                SimCLR          CLIP
────────────────────────────────────
Supervision     Image augment   Text
Data            Unlabeled       Image-caption pairs
Requires        Images only     Images + text
Generalization  Moderate        Excellent
Task alignment  Generic vision  Language grounding
Transfer        Good            Excellent
Interpretable   No              Yes (language)

When to use:
  SimCLR: When you only have unlabeled images
  CLIP: When you have image-caption pairs
```

### Method 2: MoCo - Momentum Contrast

**Problem with SimCLR:**

```
SimCLR requires large batch size:
  - Small batch: Few negatives → weak learning signal
  - Large batch: Better negatives → better learning

  Batch size 4096 requires massive GPU memory
  And distributed training complexity
```

**MoCo solution:**

```
Use memory bank instead of current batch

Benefits:
  ✓ Can use smaller batch size
  ✓ Negatives more diverse (from different times)
  ✓ More efficient
```

**Architecture:**

```
Online encoder: f_q
  Learns from current batch
  Updated every step

Memory bank: Queue
  Stores recent representations
  Old representations pushed out as new added

Momentum encoder: f_k
  Slowly following online encoder
  f_k = α × f_k + (1-α) × f_q

  Typically α = 0.999
  Moves slowly (momentum!)

Process:

1. Current batch through online encoder
   → query embeddings q

2. Pop old representations from queue
   → memory negatives

3. Compute loss using:
   - query from online encoder (positive)
   - memory from momentum encoder (negatives)

4. Push new representations to queue

5. Update momentum encoder (slowly follows online)
```

**Why momentum encoder:**

```
Without it:
  Queue contains representations from old network
  Network keeps changing → representations inconsistent
  Training unstable

With momentum encoder:
  Queue contains representations from slow network
  Representations are consistent
  Training stable

Effect:
  Momentum = inertia
  Small updates accumulate
  Smooth trajectory
```

**Performance:**

```
ImageNet pre-training → transfer to other tasks

                Top-1 Accuracy
────────────────────────────────
Supervised      76.5% (ResNet-50)
SimCLR          69.3% (requires large batch)
MoCo v1         60.6% (with 65K negatives)
MoCo v2         71.3% (improved version)
MoCo v3         76.7% (vision transformer)

Note: Self-supervised eventually matched supervised!
      Shows power of approach
```

### Method 3: BYOL - Contrastive Without Negatives

**Surprising finding (Grill et al., 2020):**

```
Do we even need negative examples?

Traditional contrastive:
  Make positives similar
  Make negatives dissimilar

BYOL:
  Only make positives similar
  No explicit negatives!

Question: How does this work?

Answer: Still has implicit negatives
        (Through model architecture and learning dynamics)
```

**Architecture:**

```
Online network:
  Encoder f + Projector g
  Input: image → output: representation
  Updated every step

Target network:
  Copy of online network
  Parameter updates: EMA (exponential moving average)
  target_param = α × target_param + (1-α) × online_param

Predictor h:
  Additional MLP on top of online network
  NOT on target network (asymmetry!)

Loss:
  For two augmentations of same image:
  loss = ||h(online(aug1)) - target(aug2)||²

  Make online and target predictions close
  Using MSE loss (not contrastive!)

  Also symmetrically:
  loss += ||h(online(aug2)) - target(aug1)||²
```

**Why this works (still debated!):**

```
Possible explanations:

1. Implicit negatives through optimization
   - Mini-batch gradient descent creates diversity
   - Network can't collapse to constant
   - Similar to negative mining

2. Momentum encoder provides stability
   - Target network changes slowly
   - Creates effective "negatives" through difference

3. Predictor prevents mode collapse
   - Without predictor: Would learn trivial solution
   - With predictor: Breaks symmetry
   - Forces meaningful learning

Empirical results:
  BYOL works surprisingly well!
  Without explicit negatives!
  Counterintuitive but effective
```

**Advantages:**

```
✓ Doesn't need negative pairs
✓ Don't need image-text pairs (image-only sufficient)
✓ Works with small batches
✓ Stable training
✓ Strong performance (competitive with SimCLR)
```

**Disadvantages:**

```
✗ Why it works still not fully understood
✗ Less interpretable
✗ More complex architecture
✗ Harder to debug when it fails
```

## 7.5 Practical Guide to Contrastive Learning

### Implementing Contrastive Learning

**Basic template:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearningModel(nn.Module):
    def __init__(self, encoder, projection_dim=256):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Linear(encoder.output_dim, projection_dim)

    def forward(self, x):
        # Encode
        h = self.encoder(x)

        # Project
        z = self.projector(h)

        # Normalize
        z = F.normalize(z, p=2, dim=1)

        return z

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss
        z_i, z_j: (batch_size, embedding_dim) tensors
        """
        batch_size = z_i.shape[0]

        # Concatenate: positive pairs are diagonal
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch, dim)

        # Similarity matrix
        similarity = torch.mm(z, z.t()) / self.temperature

        # Create labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels, labels])

        # Positive pairs at positions (i, batch+i) and (batch+i, i)
        # Compute loss: each sample should match its pair

        # Loss for all positions
        loss = F.cross_entropy(similarity, labels)

        return loss

# Training loop
def train_contrastive(model, data_loader, optimizer, device, epochs=100):
    criterion = ContrastiveLoss(temperature=0.07)

    for epoch in range(epochs):
        total_loss = 0

        for images in data_loader:
            # Get two augmented versions
            x_i = augment(images)
            x_j = augment(images)

            x_i = x_i.to(device)
            x_j = x_j.to(device)

            # Forward pass
            z_i = model(x_i)
            z_j = model(x_j)

            # Compute loss
            loss = criterion(z_i, z_j)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.3f}")
```

### Choosing Hyperparameters

**Temperature:**

```
Range: [0.05, 0.2]

Diagnostic:
  Training loss plateaus at high value?
    → Temperature too low (sharp, unstable)
    → Increase τ

  Training loss decreases but very slowly?
    → Temperature too high (smooth, weak signal)
    → Decrease τ

Rule of thumb:
  Start with τ = 0.1
  Adjust based on loss curve
```

**Batch size:**

```
Larger batch = more negatives = better signal

Typical choices:
  Small GPU: 256-512
  Medium GPU: 1024-2048
  Large GPU: 4096+
  Multi-GPU: 32K+ (like CLIP)

Trade-off:
  Larger batch: Better learning, slower per epoch
  Smaller batch: Worse learning, faster per epoch
```

**Projection dimension:**

```
Embedding dimension (before projection): 1024-2048 (from encoder)
Projection dimension: 128-512

Common choices:
  256D (standard)
  128D (more compression)
  512D (less compression)

Effect:
  Smaller: Faster computation, less memory
  Larger: More expressive, risk of overfitting
```

**Number of negatives:**

```
Within batch:
  Batch size 256 → 255 negatives per sample

Memory bank (MoCo):
  Queue size 65536 → 65535 negatives

More negatives → better learning signal
But more computation
Typical: 255-65K negatives
```

### Evaluating Contrastive Models

**Method 1: Linear evaluation protocol**

```
1. Train contrastive model on unlabeled data
   → Get representations

2. Freeze encoder
   → Don't update weights

3. Train linear classifier on representations
   → Small labeled dataset

4. Evaluate on test set

Metric: Accuracy of linear classifier
Insight: If representations good → linear classifier accurate

Example:
  CIFAR-10 (50K training images)
  Contrastive pre-training: All 50K unlabeled
  Linear eval: 5K labeled for training, 10K for testing

  Result: 96% accuracy
  Interpretation: Representations capture meaningful patterns
```

**Method 2: Transfer learning evaluation**

```
1. Train contrastive model on source dataset
2. Fine-tune on target task
3. Compare to:
   - Supervised baseline
   - Random initialization
   - Other pre-training methods

Metric: Downstream task accuracy
Insight: Better representations → better transfer
```

**Method 3: Downstream task performance**

```
Pre-training dataset: ImageNet (unlabeled contrastive)
Downstream tasks:
  1. ImageNet-100 classification (supervised fine-tune)
  2. CIFAR-10 classification
  3. STL10 classification
  4. Transfer to object detection
  5. Transfer to segmentation

Results show generalization across tasks
```

## 7.6 Troubleshooting Contrastive Learning

### Problem 1: Loss not decreasing

**Potential causes:**

```
① Temperature too low
   Effect: Softmax too sharp
   Solution: Increase τ (e.g., 0.1 → 0.2)

② Learning rate too small
   Effect: Updates too tiny
   Solution: Increase learning rate

③ Batch size too small
   Effect:

-----


```
   Effect: Weak learning signal
   Solution: Increase batch size if possible

④ Bad initialization
   Effect: Starting in bad local minimum
   Solution: Use proper weight initialization

⑤ Augmentations too weak
   Effect: Positive pairs too similar anyway
   Solution: Increase augmentation strength

⑥ Augmentations too strong
   Effect: Positive pairs become different objects
   Solution: Decrease augmentation strength
```

**Debugging steps:**

```python
# 1. Check loss values
print(f"Initial loss: {loss.item()}")
# Should decrease over time
# If increasing or constant: something wrong

# 2. Check similarity matrix
similarity = torch.mm(z, z.t())
print(f"Max similarity: {similarity.max():.3f}")
print(f"Min similarity: {similarity.min():.3f}")
# Should: Max ≈ 1, Min ≈ -1 for normalized vectors

# 3. Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.3f}")
# Should be reasonable values (not 0, not inf)

# 4. Check temperature effect
temperatures = [0.01, 0.05, 0.1, 0.2, 0.5]
for tau in temperatures:
    loss = compute_loss(embeddings, tau)
    print(f"τ={tau}: loss={loss:.3f}")
# Should have sweet spot, not too high/low everywhere
```

### Problem 2: Representation collapse

**What is it:**

```
Model learns to make all representations nearly identical

Example:
  All images → representation [0.5, 0.5, 0.5, ...]
  All images → representation [0.51, 0.49, 0.50, ...]

  Trivial solution: "All same = all similar"
  Loss can be artificially low!
  But representations useless for downstream tasks
```

**Symptoms:**

```
✓ Loss decreasing nicely
✗ Linear evaluation performance poor
✗ Representations clustered at single point
✗ Variance of representations near zero
```

**Causes and solutions:**

```
Cause 1: No negatives (only positives)
  Solution: Ensure you have negatives in batch

Cause 2: Batch too small
  Solution: Increase batch size

Cause 3: No regularization
  Solution: Add normalization (L2 normalization helps)

Cause 4: Poor augmentations
  Solution: Ensure augmentations are meaningful
  (Reproduce the issue with weak augmentations)
```

**Prevention:**

```python
# Monitor variance
def monitor_collapse(z):
    """Check if representations are collapsing"""
    # Variance across batch
    variance = torch.var(z, dim=0).mean()

    # Std across batch
    std = torch.std(z, dim=0).mean()

    print(f"Variance: {variance:.4f}")
    print(f"Std: {std:.4f}")

    if variance < 0.001:
        print("WARNING: Representations collapsing!")
        return False
    return True

# During training
for z_i, z_j in batches:
    if not monitor_collapse(z_i):
        # Take corrective action
        # Adjust learning rate, batch size, etc.
        pass
```

### Problem 3: Slow convergence

**Causes:**

```
① Learning rate too small
   → Gradients don't produce meaningful updates
   → Training takes forever

② Too few negatives
   → Weak learning signal
   → Takes many steps to learn

③ Bad data augmentation
   → Positive pairs too similar/different
   → Model confused about what to learn

④ Model too complex
   → Slow to train
   → Consider simpler architecture
```

**Solutions:**

```
1. Learning rate warmup
   Gradually increase LR from 0 to target
   Helps with stability

   Schedule:
   LR(t) = target_lr * min(1, t / warmup_steps)

2. Learning rate scheduling
   Reduce LR as training progresses
   Helps fine-tuning

   CosineAnnealingLR: Common choice

3. Increase batch size
   If hardware permits
   Each sample gets more negatives
   Stronger learning signal

4. Use momentum
   Keep moving average of gradients
   Smooths noisy gradient signal
```

## Key Takeaways

- **Contrastive learning** learns from similarity/dissimilarity without labels
- **InfoNCE loss** is the foundation: maximize positive similarity relative to negatives
- **CLIP** revolutionized the field with language-grounded vision at scale
- **Temperature** controls softmax sharpness and learning signal
- **Self-supervised variants** (SimCLR, MoCo, BYOL) enable learning from unlabeled data
- **Large batch size** provides more negatives and stronger signal
- **Hyperparameter tuning** (temperature, batch size, augmentation) is crucial
- **Representation collapse** is a real risk to monitor

## Exercises

**⭐ Beginner:**
1. Implement InfoNCE loss from scratch
2. Compute temperature effects on loss
3. Understand positive/negative pairs in a batch

**⭐⭐ Intermediate:**
4. Build image-text contrastive model on small dataset
5. Implement temperature scheduling
6. Compare different similarity metrics

**⭐⭐⭐ Advanced:**
7. Implement SimCLR with proper augmentations
8. Build MoCo with momentum encoder
9. Debug and fix representation collapse

---

