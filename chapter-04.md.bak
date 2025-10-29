# Chapter 4: Feature Alignment and Bridging Modalities

---

**Previous**: [Chapter 3: Feature Representation for Each Modality](chapter-03.md) | **Next**: [Chapter 5: Fusion Strategies](chapter-05.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand why alignment is necessary
- Implement shared embedding spaces
- Use cross-attention for fine-grained alignment
- Handle bidirectional alignment
- Solve alignment in practice

## 4.1 The Alignment Problem

### Why Alignment Matters

**The Core Challenge:**

```
Image features: 2048-dimensional vector (from ResNet)
Text features: 768-dimensional vector (from BERT)

Question: How similar are they?

Problem:
  ✗ Different dimensions (can't directly compare)
  ✗ Different scales (0-1 for text, -infinity to infinity for images)
  ✗ Different semantics (what does dimension 500 mean in each?)
  ✗ No natural similarity metric

We need: ALIGNMENT
Goal: Make image and text features "understand" each other
```

**Real-world consequence:**

```
Application: Image-text search
  User searches: "red cat"
  System has: 10 million images + descriptions

Without alignment:
  Can't compare image and text vectors
  Search impossible

With alignment:
  Image vectors and text vectors in same space
  Similarity computed easily
  Search works!
```

### Levels of Alignment

**Level 1: Coarse-grained (Document-level)**

```
Entire image ↔ Entire text description

Example:
  Image: [Full photo of cat on chair]
  Text: "A tabby cat relaxing on a wooden chair"

  Alignment: Image matches entire text

Use cases:
  - Image-text retrieval
  - Image classification with descriptions
  - Document understanding (image + caption)

Challenge: Image might have multiple objects
           Text mentions most important ones
```

**Level 2: Fine-grained (Region-level)**

```
Image regions ↔ Text phrases

Example:
  Image regions:
    Region 1: [Cat's head area]
    Region 2: [Chair seat area]
    Region 3: [Background]

  Text phrases:
    "tabby cat" ↔ Region 1
    "wooden chair" ↔ Region 2
    "cozy room" ↔ Region 3

Use cases:
  - Visual question answering (where are things?)
  - Dense image captioning
  - Object detection with descriptions
  - Grounding language in images

Challenge: Multiple valid region boundaries
           Phrases don't perfectly correspond to regions
```

**Level 3: Very Fine-grained (Pixel/Token-level)**

```
Image pixels ↔ Text tokens

Example:
  Video frame:
    [Pixels 100-200]: Red fur
    [Pixels 500-600]: Cat's eye
    [Pixels 800-900]: Chair texture

  Text tokens:
    "red" ↔ Red fur pixels
    "cat" ↔ Cat structure pixels
    "chair" ↔ Chair pixels

Use cases:
  - Semantic segmentation with text
  - Dense video captioning with timestamps
  - Pixel-level understanding with descriptions

Challenge: Extremely fine-grained
           Requires pixel-level annotations
           Computationally expensive
```

### Why Alignment is Hard

**Reason 1: One-to-many mappings**

```
Single image can have many valid descriptions:

Image: [Cat on bed]

Valid descriptions:
  ① "A cat is sleeping on a bed"
  ② "A cat on a bed"
  ③ "Feline on furniture"
  ④ "A cozy cat"
  ⑤ "Kitty resting"

All correct!
No single "ground truth" alignment

Challenge: How to learn from multiple valid targets?
Solution: Use soft targets or ranking-based losses
```

**Reason 2: Implicit pairing in training data**

```
Web data structure:

[Article with title: "Beautiful pets"]
│
├─ [Image 1]
├─ [Image 2]
├─ [Large paragraph mentioning pets]
├─ [Image 3]
└─ [Image 4]

Challenge:
  Which image goes with which sentence?
  Are all images described equally?

Solutions:
  - Assume images near text match it
  - Learn implicit pairings
  - Use weak supervision signals
```

**Reason 3: Semantic gaps**

```
Image and text express different aspects:

Image: "Tabby cat, orange color, on blue chair, sunny room"
Text: "A cat resting"

Text is abstract summary
Image is concrete visual

How to align?
  Need to map concrete visual features
  to abstract semantic concepts

This requires:
  ① Understanding visual features
  ② Understanding text semantics
  ③ Bridging the gap
```

**Reason 4: Missing or corrupted data**

```
Data quality issues:

Situation 1: Image and text don't match
  Image: [Car]
  Text: "Beautiful sunset"

  Alignment should recognize mismatch

Situation 2: Image is corrupted
  Image: [Blank/noise]
  Text: "A dog running"

  Should still align based on text

Situation 3: Text is poorly written
  Image: [Cat photo]
  Text: "teh kat iz vry smrt"

  Should understand despite bad spelling
```

## 4.2 Shared Embedding Space - The Standard Solution

### Core Concept

**Idea:**
```
Project both modalities to common space
where similarity can be computed

Image (2048D) --┐
               ├─→ Shared Space (256D)
Text (768D) ───┘

Now both in same space!
Can compute cosine similarity directly
```

### Implementation

**Step 1: Learn projection matrices**

```
For images:
  W_img ∈ ℝ^(2048 × 256)
  img_proj = W_img @ img_features

For text:
  W_txt ∈ ℝ^(768 × 256)
  txt_proj = W_txt @ txt_features

Both outputs: 256-dimensional vectors
```

**Step 2: Normalize in shared space**

```
# L2 normalize to unit length
img_proj = img_proj / ||img_proj||
txt_proj = txt_proj / ||txt_proj||

Result:
  Both vectors have magnitude 1
  Can use cosine similarity = dot product
  Similarity ∈ [-1, 1]
```

**Step 3: Compute similarity**

```
similarity = img_proj · txt_proj

= Σ(img_proj_i × txt_proj_i)

Result interpretation:
  > 0.8:   Very similar (matched pair)
  0.5-0.8: Similar
  0.3-0.5: Somewhat related
  < 0.3:   Different (unrelated pair)
```

### Learning the Projections

**Training objective:**

```
Goal: Maximize similarity of matched pairs
      Minimize similarity of unmatched pairs

Dataset: Pairs (image_i, text_i) where i means matched

Loss function (InfoNCE / Contrastive):

L = -log[ exp(sim(img_i, txt_i) / τ) /
          (exp(sim(img_i, txt_i) / τ) + Σ_j≠i exp(sim(img_i, txt_j) / τ)) ]

Intuition:
  Numerator: Similarity of correct pair (should be high)
  Denominator: All pairs including incorrect ones
  Loss: Make correct pair stand out from all others
```

**Batching strategy:**

```
Batch of 32 samples:

[img_1] ────────┐
[img_2] ────────┼─ All project to shared space
[img_3] ────────┤
...             │
[img_32] ───────┘

[txt_1] ────────┐
[txt_2] ────────┼─ All project to shared space
[txt_3] ────────┤
...             │
[txt_32] ───────┘

Similarity matrix (32×32):
  sim(img_1, txt_1) = 0.95  ← Matched
  sim(img_1, txt_2) = 0.2   ← Unmatched
  sim(img_2, txt_2) = 0.94  ← Matched
  ...

Loss: Make diagonal elements high
      Make off-diagonal elements low
```

**Dimension selection:**

```
Choice of shared space dimension:

Small (64D):
  ✓ Fast computation
  ✓ Less memory
  ✗ Information loss
  ✗ Can't capture fine details

Medium (256D):
  ✓ Good balance
  ✓ Standard choice
  ✓ Preserves information

Large (1024D):
  ✓ Maximum information
  ✗ Slow computation
  ✗ More memory
  ✗ Risk of overfitting

Typical sweet spot: 256-512D
```

### Practical Example

**Image-Text Retrieval System:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextAligner(nn.Module):
    def __init__(self, img_dim=2048, txt_dim=768, shared_dim=256):
        super().__init__()

        # Projection layers
        self.img_projection = nn.Linear(img_dim, shared_dim)
        self.txt_projection = nn.Linear(txt_dim, shared_dim)

    def forward(self, img_features, txt_features):
        # Project to shared space
        img_proj = self.img_projection(img_features)  # (batch, 256)
        txt_proj = self.txt_projection(txt_features)  # (batch, 256)

        # L2 normalize
        img_proj = F.normalize(img_proj, p=2, dim=1)
        txt_proj = F.normalize(txt_proj, p=2, dim=1)

        return img_proj, txt_proj

    def compute_similarity(self, img_proj, txt_proj):
        # Cosine similarity = dot product of normalized vectors
        similarity = torch.mm(img_proj, txt_proj.t())  # (batch, batch)
        return similarity

# Training
model = ImageTextAligner()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for batch in data_loader:
    images, texts = batch

    img_features = image_encoder(images)  # (batch, 2048)
    txt_features = text_encoder(texts)    # (batch, 768)

    # Align
    img_proj, txt_proj = model(img_features, txt_features)

    # Compute similarities
    similarities = model.compute_similarity(img_proj, txt_proj)

    # Contrastive loss
    batch_size = img_proj.shape[0]
    labels = torch.arange(batch_size).to(device)

    loss_img = F.cross_entropy(similarities / temperature, labels)
    loss_txt = F.cross_entropy(similarities.t() / temperature, labels)
    loss = (loss_img + loss_txt) / 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 4.3 Cross-Attention for Fine-Grained Alignment

### Motivation

**Problem with shared space:**

```
Shared embedding gives global similarity
But doesn't tell us WHAT matched

Image representation: Single 256D vector
  Represents entire image
  Loses spatial structure

Text representation: Single 256D vector
  Represents entire sentence
  Loses word-level information

Result: Good for retrieval
        Bad for understanding fine-grained relationships

Solution: Cross-attention!
```

### Cross-Attention Mechanism

**Core idea:**

```
Let each part of one modality
"look at" relevant parts of another modality

Text looks at image:
  Word "red" ← attends to → Red pixels in image
  Word "cat" ← attends to → Cat-shaped pixels in image

Image looks at text:
  Cat region ← attends to → "cat" and "tabby" words
  Background ← attends to → "room" word
```

**Mathematical formulation:**

```
Cross-attention (Text queries, Image keys/values):

Query = Text embeddings (sequence length = # words)
Key = Image patches from CNN (# patches = 196 for ViT)
Value = Image patches

Attention = softmax(Query @ Key^T / √d_k) @ Value

Result:
  Each word gets weighted combination of image patches
  Weights reflect relevance (attention)
```

**Concrete example:**

```
Text: "The [red] cat [sits]"
       Word 1  Word 2  Word 3

Image: Divided into 9 patches (3×3 grid)

┌─────┬─────┬─────┐
│ Sky │ Sky │ Sky │
├─────┼─────┼─────┤
│ Cat │ Cat │Chair│
├─────┼─────┼─────┤
│Grass│ Cat │Chair│
└─────┴─────┴─────┘

Cross-attention: Word "red" attends to image patches

Attention scores:
  Sky: 0.1
  Sky: 0.1
  Sky: 0.1
  Cat(red): 0.5  ← High! Red cat
  Cat: 0.3
  Chair: 0.0
  Grass: 0.0
  Cat: 0.3
  Chair: 0.0

Attention output: Weighted combination of patches
  0.5 × Cat_patch_red + 0.3 × Cat_patch + 0.3 × Cat_patch
  ≈ Feature vector emphasizing red cat region
```

### Multi-head Cross-Attention

**Why multiple heads?**

```
Different heads can attend to different aspects

Head 1: Color-sensitive
  Attends to: Patches with matching colors

Head 2: Shape-sensitive
  Attends to: Patches with matching shapes

Head 3: Texture-sensitive
  Attends to: Patches with matching textures

All heads run in parallel
Results concatenated
```

**Implementation:**

```
# Pseudo-code for single attention head

Query = W_q @ text_embedding      # (seq_len, d_k)
Key = W_k @ image_patches         # (num_patches, d_k)
Value = W_v @ image_patches       # (num_patches, d_v)

# Compute attention weights
scores = Query @ Key^T            # (seq_len, num_patches)
weights = softmax(scores / √d_k)  # (seq_len, num_patches)

# Apply to values
output = weights @ Value          # (seq_len, d_v)

# For multiple heads: repeat with different W_q, W_k, W_v
# Then concatenate outputs
```

### Bidirectional Alignment

**Problem:** Cross-attention only goes one way

```
Text attends to image: ✓ Good
  Words understand image

Image attends to text: ✗ Missing
  Image regions don't understand text
  Asymmetric!
```

**Solution: Bidirectional attention**

```
Step 1: Text cross-attends to image
  Query = Text
  Key/Value = Image
  Result: Text-aware-of-image

Step 2: Image cross-attends to text
  Query = Image patches
  Key/Value = Text
  Result: Image-aware-of-text

Step 3: Combine both representations
  Use refined representations for tasks
```

**Architecture with bidirectional alignment:**

```
Initial representations:
  Image patches: 196 vectors (from ViT)
  Text tokens: 10 vectors (from BERT)

Layer 1:
  ├─ Text cross-attends to Image
  │  └─ Output: Refined text (10 vectors)
  │
  └─ Image self-attends within itself
     └─ Output: Refined image (196 vectors)

Layer 2:
  ├─ Image cross-attends to Text
  │  └─ Output: Refined image (196 vectors)
  │
  └─ Text self-attends within itself
     └─ Output: Refined text (10 vectors)

Layer 3-6: Repeat above

Result:
  Both modalities refined with knowledge of other
  Bidirectional influence
```

**Example - VQA (Visual Question Answering):**

```
Image: [Photo of cat on chair]
Question: "What's on the chair?"

Processing with bidirectional alignment:

① Initial encoding:
   Image patches: 196 ViT features
   Question: "What's", "on", "the", "chair", "?" (5 tokens)

② Text understands image context:
   "chair" attends to chair-region patches
   "on" understands preposition in spatial context

   Result: Question tokens now image-aware

③ Image understands question:
   Chair region attends to "chair" token
   Surrounding region attends to "on" (preposition)

   Result: Image patches now question-aware

④ Predict answer:
   Question tokens generate: "A cat"

Benefits:
  - Question focuses on relevant image parts
  - Image highlights relevant content for question
  - Mutual refinement through layers
  - Better understanding than independent processing
```

## 4.4 Practical Alignment Challenges and Solutions

### Challenge 1: Handling Multiple Valid Alignments

**Problem:**

```
Image: [Multi-object scene: cat, dog, table]
Text options:
  ① "Pets on table"
  ② "Table with animals"
  ③ "Room with furniture"
  ④ "A table with a cat and dog"

All valid descriptions!
Which should model learn?
```

**Solutions:**

**Solution 1: All positives training**

```
Treat all valid descriptions as positive examples

Loss = -log[exp(sim(img, txt1)) + exp(sim(img, txt2)) + exp(sim(img, txt3))]
       / [exp(sim(img, txt1)) + exp(sim(img, txt2)) + exp(sim(img, txt3)) +
          exp(sim(img, neg1)) + exp(sim(img, neg2)) + ...]

Code:
  positives = [text1, text2, text3]  # All valid
  negatives = [text4, text5, ...]    # Invalid

  for pos in positives:
    loss += InfoNCE_loss(img, pos, negatives)
```

**Solution 2: Soft targets**

```
Assign soft probability to each description

Similarity scores: [0.9, 0.85, 0.7, 0.3, 0.1]
Probabilities: [0.4, 0.4, 0.15, 0.04, 0.01]

Distribution rather than hard binary labels
Model learns to match range of good descriptions
```

**Solution 3: Ranking-based loss**

```
Instead of absolute similarity,
optimize relative ranking

Constraint: sim(img, good_text) > sim(img, bad_text) + margin

Loss = max(0, margin + sim(img, bad) - sim(img, good))

Model ensures good descriptions rank higher
Not concerned with absolute values
```

### Challenge 2: Incomplete or Corrupted Data

**Problem 1: Image-text mismatch**

```
Web data contains misaligned pairs:

Website page:
[Image1: Beautiful sunset]
[Article about technology and computers]
[Image2: Laptop]

Problem:
  Image1 doesn't match article
  Article matches Image2 only

Solution: Robustness to noise
  - Training with some wrong pairs is okay
  - Model learns that MOST pairs are correct
  - Wrong pairs become negatives
  - Loss still works

  Empirically: Works even with 20-30% misaligned data
```

**Problem 2: Low-quality images or text**

```
Images:
  - Blurry photos
  - Extreme lighting
  - Occlusions
  - Irrelevant backgrounds

Text:
  - Spelling errors
  - Grammar mistakes
  - Abbreviations
  - Emotional/subjective language

Solution: Robust feature extraction
  - Use pre-trained encoders (already robust)
  - Encoders trained on diverse data
  - Can handle degraded inputs
  - Alignment robust if base features good
```

**Problem 3: Context-dependent meaning**

```
Text: "A record player"
Image: [Phonograph]

Challenge:
  "Record" = LP vinyl record vs historical record
  "Player" = music player vs sports player
  Multiple interpretations!

Solution: Context through attention
  - Image patches clarify which "record"
  - Text confirms image interpretation
  - Cross-attention resolves ambiguity
```

### Challenge 3: Scaling to Large Datasets

**Problem:**

```
Computing similarity matrix for large batch:

Batch size: 10,000 images and 10,000 texts
Similarity matrix size: 10,000 × 10,000 = 100M elements

Computation:
  ① Forward pass: 100M multiplies
  ② Softmax: 100M exponentials
  ③ Backward pass: 100M gradients

Result: Extremely slow!
GPU memory: 100M × float32 = 400MB just for similarities
```

**Solutions:**

**Solution 1: Smaller batches**

```
Batch size: 256 instead of 10,000
Similarity matrix: 256 × 256 = 65K elements

Trade-off:
  ✓ Faster training
  ✓ Less memory
  ✗ Noisier gradients (fewer negatives)
  ✗ More iterations needed
```

**Solution 2: Distributed training**

```
Split batch across multiple GPUs

GPU 1: 2500 images and texts
GPU 2: 2500 images and texts
GPU 3: 2500 images and texts
GPU 4: 2500 images and texts

Gradient computation happens locally
All-reduce aggregates gradients

Enables:
  - Larger effective batch size
  - Better negatives for learning
  - Faster training overall
```

**Solution 3: Hard negative mining**

```
Instead of all negatives,
select hard negatives (easily confused)

Full set: 10,000 possible negatives
Sample: 32 hard negatives (ones model struggles with)

Benefits:
  - Reduces computation
  - More efficient learning (focus on hard cases)
  - Still effective despite smaller negative set
```

## 4.5 Evaluating Alignment Quality

### Metrics for Alignment

**1. Retrieval Metrics**

```
Setup: Given 1000 images and 1000 texts (properly paired)
Task: For each image, rank texts by similarity

Metrics:

Recall@K:
  Did correct text appear in top K?

  Example (K=1):
    For each image, check if correct text in top 1
    Count successes / total images

  Recall@1: 75% (750/1000 correct)
  Recall@5: 95% (950/1000 correct)

  Interpretation:
    Recall@1 = Exact match retrieval rate
    Recall@5 = Reasonable match rate

Mean Reciprocal Rank (MRR):
  Average rank of correct match

  Example:
    Image 1: Correct text at rank 3 → 1/3
    Image 2: Correct text at rank 1 → 1/1
    Image 3: Correct text at rank 10 → 1/10
    MRR = (1/3 + 1/1 + 1/10) / 3 ≈ 0.44

Normalized DCGA (NDCG):
  Accounts for relevance scores
  Perfect ranking = 1.0
```

**2. Correlation Metrics**

```
Idea: Good alignment means
      similar images/texts have high correlation

Spearman Correlation:
  ① Rank pairs by human similarity judgment
  ② Rank same pairs by model similarity
  ③ Compute rank correlation

  Perfect: Correlation = 1.0
  Random: Correlation ≈ 0.0

Pearson Correlation:
  Linear correlation between human and model scores
```

**3. Classification Metrics**

```
Binary classification: Correct or incorrect pairing?

Dataset: 1000 correct pairs + 1000 incorrect pairs

Metrics:
  Accuracy: How many correct predictions?
  Precision: Of positive predictions, how many correct?
  Recall: Of correct pairs, how many identified?
  F1: Harmonic mean of precision and recall

Example results:
  Accuracy: 95% (1900/2000 correct)
  Precision: 96% (970/1010 predicted positive)
  Recall: 97% (970/1000 actually positive)
  F1: 0.965
```

### Example Evaluation Script

```python
from sklearn.metrics import recall_score, ndcg_score
import numpy as np

def evaluate_alignment(img_features, txt_features, labels):
    """
    img_features: (N, 256) aligned image embeddings
    txt_features: (N, 256) aligned text embeddings
    labels: (N,) ground truth labels (0=incorrect, 1=correct)
    """

    # Compute similarities
    similarities = np.dot(img_features, txt_features.T)

    # Recall@K
    recall_at_1 = compute_recall_at_k(similarities, labels, k=1)
    recall_at_5 = compute_recall_at_k(similarities, labels, k=5)

    # Classification metrics
    binary_predictions = (similarities > threshold).astype(int)
    accuracy = np.mean(binary_predictions == labels)
    precision = precision_score(labels, binary_predictions)
    recall = recall_score(labels, binary_predictions)
    f1 = f1_score(labels, binary_predictions)

    print(f"Recall@1: {recall_at_1:.3f}")
    print(f"Recall@5: {recall_at_5:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

    return {
        'recall_at_1': recall_at_1,
        'recall_at_5': recall_at_5,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_recall_at_k(similarities, labels, k):
    """Compute Recall@K metric"""
    n = similarities.shape[0]
    recall_sum = 0

    for i in range(n):
        # Get top-K most similar texts for this image
        top_k_idx = np.argsort(similarities[i])[-k:]

        # Check if any of top-K are correct (label=1)
        if np.any(labels[top_k_idx] == 1):
            recall_sum += 1

    return recall_sum / n
```

## Key Takeaways

- **Alignment is essential** for connecting different modalities
- **Shared embedding space** is standard, scalable solution
- **Cross-attention** enables fine-grained alignment
- **Bidirectional fusion** gives mutual understanding
- **Practical challenges** require careful handling
- **Multiple evaluation metrics** give comprehensive picture

## Exercises

**⭐ Beginner:**
1. Implement cosine similarity between image and text features
2. Visualize shared embedding space using t-SNE
3. Compute recall@K for sample retrieval task

**⭐⭐ Intermediate:**
4. Build shared embedding projection layers
5. Implement contrastive loss training
6. Evaluate alignment with multiple metrics

**⭐⭐⭐ Advanced:**
7. Implement bidirectional cross-attention from scratch
8. Build hard negative mining strategy
9. Compare different loss functions (InfoNCE, triplet, etc.)

---

