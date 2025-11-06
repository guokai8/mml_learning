# Chapter 2: Foundations and Core Concepts

---

**Previous**: [Chapter 1: Introduction to Multimodal Learning](chapter-01.md) | **Next**: [Chapter 3: Feature Representation for Each Modality](chapter-03.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand mathematical foundations of embeddings
- Calculate and interpret similarity metrics
- Recognize good vs. poor embedding properties
- Handle multimodal feature combinations correctly
- Avoid common pitfalls in cross-modal comparison

## 2.1 Mathematical Foundations

### Vectors and Embedding Spaces

**Definition:** An embedding is a representation of data as a vector in a high-dimensional space.

**Mathematical notation:**
```
For data point x, its embedding e is:
e ∈ ℝ^d

where:
d = dimensionality of embedding space
ℝ^d = d-dimensional real number space

Example:
Image of cat → **e**_image = [0.2, -0.5, 0.8, ..., 0.1] ∈ ℝ^2048
Text "cat" → **e**_text = [0.1, 0.3, -0.2, ..., 0.5] ∈ ℝ^768
```

### Distance and Similarity Metrics

#### Cosine Similarity (Recommended)

**Definition:**
```
For vectors **u** and **v**:

cosine_similarity(**u**, **v**) = (**u** · **v**) / (||**u**|| × ||**v**||)

where:
**u** · **v** = dot product = u₁v₁ + u₂v₂ + ... + uₙvₙ
||**u**|| = magnitude = √(u₁² + u₂² + ... + uₙ²)
```

**Range:** [-1, 1]
- 1 = identical direction (most similar)
- 0 = perpendicular (unrelated)
- -1 = opposite direction (most dissimilar)

**Geometric interpretation:**
```
Cosine similarity measures the angle between vectors:
- θ = 0° → cos(0°) = 1 (identical)
- θ = 90° → cos(90°) = 0 (orthogonal)
- θ = 180° → cos(180°) = -1 (opposite)
```

**Why preferred for embeddings:**
- Invariant to magnitude (direction matters, not size)
- Computationally efficient
- Interpretable as angle between vectors
- Works well in high-dimensional spaces

**Example calculation:**

```python
import numpy as np

# Example vectors
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Manual calculation
dot_product = np.dot(u, v)  # 1*4 + 2*5 + 3*6 = 32
norm_u = np.linalg.norm(u)  # √(1² + 2² + 3²) = √14 ≈ 3.74
norm_v = np.linalg.norm(v)  # √(4² + 5² + 6²) = √77 ≈ 8.77

cosine_sim = dot_product / (norm_u * norm_v)  # 32 / (3.74 × 8.77) ≈ 0.975

# Using sklearn
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity([u], [v])[0][0]  # Same result
```

#### Euclidean Distance (L2 Distance)

**Definition:**
```
For vectors **u** and **v**:

euclidean_distance(**u**, **v**) = ||**u** - **v**|| = √(Σᵢ(uᵢ - vᵢ)²)

Example:
**u** = [1, 2]
**v** = [4, 6]
distance = √((1-4)² + (2-6)²) = √(9 + 16) = √25 = 5
```

**When to use:**
- When absolute differences matter
- In clustering algorithms
- When coordinates have physical meaning

**Drawback for high dimensions:**
- "Curse of dimensionality" - distances become less meaningful
- Cosine similarity preferred for embeddings

#### Manhattan Distance (L1 Distance)

**Definition:**
```
manhattan_distance(**u**, **v**) = Σᵢ|uᵢ - vᵢ|

Example:
**u** = [1, 2]
**v** = [4, 6]
distance = |1-4| + |2-6| = 3 + 4 = 7
```

**When to use:**
- When features are independent
- Robust to outliers
- Some recommendation systems

### What Makes a Good Embedding?

A quality embedding space should satisfy several properties:

1. **Meaningfulness**
   - Similar inputs → similar vectors
   - Related concepts → nearby in space

2. **Efficiency**
   - Reasonable dimensionality (not too large)
   - Fast to compute
   - Fast to compare

3. **Stability**
   - Small input changes → small embedding changes
   - Noise in input shouldn't drastically change embedding

4. **Interpretability** (optional but helpful)
   - Can understand what dimensions represent
   - Some dimensions → face detection, others → color, etc.

5. **Transferability**
   - Learned embeddings work across tasks
   - Generalizes to new data

### Dimensionality Considerations

**Common embedding dimensions:**

```
Task                          Typical Dimension
────────────────────────────────────────────
Word embeddings (Word2Vec)    300
Sentence embeddings (BERT)    768
Image features (ResNet)       2048
Audio features (MFCC)         39
Hybrid multimodal            256-512
```

**Trade-offs:**

```
Low dimensions (64-256):
  ✓ Fast computation  
  ✓ Less memory
  ✓ Less prone to overfitting
  ✗ May lose important information
  ✗ Limited expressiveness

High dimensions (1024-4096):
  ✓ Can capture fine details
  ✓ Better expressiveness
  ✗ Slow computation
  ✗ High memory usage
  ✗ Prone to overfitting
  ✗ Curse of dimensionality (distances become uninformative)

Sweet spot: 256-2048D for most applications
```

## 2.3 Core Concepts Illustrated

### Embedding Space Visualization

While we can't visualize high-dimensional spaces, we can use dimensionality reduction:

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**

```
2048D embedding space
        ↓ [t-SNE projection]
2D visualization space

Purpose: Preserve local neighborhoods
Good for: Exploring clusters and relationships
Limitation: Global structure may be distorted
```

**UMAP (Uniform Manifold Approximation and Projection):**

```
Benefits over t-SNE:
- Preserves both local and global structure
- Faster for large datasets
- More theoretically grounded
- Better preservation of distances
```

**PCA (Principal Component Analysis):**

```
Benefits:
- Linear transformation (interpretable)
- Preserves global structure
- Fast computation

Limitations:
- May lose non-linear relationships
- First components may not be most semantically meaningful
```

### Semantic Relationships in Vector Space

**Key insight:** Semantic relationships become geometric relationships

**Example - Word Embeddings:**

```
Mathematical relationship:
**king** - **man** + **woman** ≈ **queen**

Geometric interpretation:
- Vector from "man" to "king" captures "royalty" direction
- Applying same direction to "woman" yields "queen"
- Gender and status become orthogonal dimensions
```

**Verification in embedding space:**

```python
# Hypothetical vectors (simplified to 3D for illustration)
king = np.array([0.8, 0.1, 0.9])    # High royalty, low gender, high status
man = np.array([0.1, -0.8, 0.3])    # Low royalty, male, medium status  
woman = np.array([0.1, 0.8, 0.3])   # Low royalty, female, medium status
queen = np.array([0.8, 0.8, 0.9])   # High royalty, female, high status

# Test the relationship
result = king - man + woman
print(f"Result: {result}")          # Should be close to queen
print(f"Queen: {queen}")
print(f"Cosine similarity: {cosine_similarity([result], [queen])[0][0]:.3f}")
```

## 2.4 Modality-Specific Representations

### Text Representations

**Challenge:** Convert discrete symbols (words) into continuous vectors

```
METHOD 1 - One-hot encoding:
  Vocabulary: [the, cat, sat, on, mat]
  "cat" → [0, 1, 0, 0, 0]
  Dimension = vocabulary size
  Problem: Very high-dimensional, no semantic meaning

METHOD 2 - Word embeddings:
  "cat" → [0.2, -0.5, 0.8, ..., 0.1]
  Dimension = 300 (fixed)
  Benefit: Captures semantic meaning
```

**Modern approaches:**

```
Word2Vec (2013):
  - Learns from word co-occurrence
  - 300D vectors typically
  - Captures semantic relationships
  - Limitation: Single vector per word (polysemy problem)

BERT (2018):
  - Contextual embeddings
  - 768D vectors typically
  - Different vector for same word in different contexts
  - "bank" in "river bank" vs "financial bank"
```

### Image Representations

**Challenge:** Convert pixel arrays into semantic features

**Representation hierarchy:**

```
LEVEL 1 - Pixel level:
  Image 224×224×3 → 150,528 values
  Problem: Too high-dimensional, redundant

LEVEL 2 - Low-level features:
  Edges, corners, textures
  Extracted by early CNN layers

LEVEL 3 - Mid-level features:
  Simple shapes, color regions
  Extracted by middle CNN layers

LEVEL 4 - High-level features:
  Object parts, semantic concepts
  Extracted by deep CNN layers

LEVEL 5 - Global representation:
  Final feature vector (e.g., 2048D)
  Represents entire image content
```

**CNN Feature Extraction Process:**

```
Input: 224×224×3 image (150,528 values)
        ↓
Conv layers: Extract hierarchical features
        ↓
Global Average Pool: Summarize spatial information
        ↓
Output: 2048D feature vector **v**_image

Properties:
- Captures visual content at multiple scales
- Translation and scale invariant (to some degree)  
- Trained on millions of images
- Transfer learning: Pre-trained features work for new tasks
```

**Key insight for multimodal:**
```
Images are not immediately interpretable
The 2048 dimensions don't correspond to human-understandable concepts
(Except through attention visualization)

This makes image-text alignment challenging
Must learn mappings between image features and text concepts
```

## 2.5 Cross-Modal Feature Combination

### The Fundamental Problem

When working with multiple modalities, we face the **representation gap**:

```
Image features: **v**_img ∈ ℝ^2048  (from ResNet)
Text features:  **v**_txt ∈ ℝ^768   (from BERT)

Problems:
1. Different dimensions (2048 ≠ 768)
2. Different ranges and scales
3. Different semantic spaces
4. No natural correspondence between dimensions
```

### Critical Issue: The Wrong Way to Combine Features

⚠️ **IMPORTANT WARNING** ⚠️

A common **MISTAKE** is to use separate dimensionality reduction techniques:

```python
# ❌ WRONG APPROACH - DON'T DO THIS!
from sklearn.decomposition import PCA

# Separate PCA for each modality
pca_img = PCA(n_components=256)
pca_txt = PCA(n_components=256)

# Project to same dimension BUT different coordinate systems
img_proj = pca_img.fit_transform(image_features)    # 256D in coordinate system A
txt_proj = pca_txt.fit_transform(text_features)     # 256D in coordinate system B

# Compute similarity
similarity = cosine_similarity(img_proj, txt_proj)  # ❌ MEANINGLESS!
```

**Why this is wrong:**

```
The problem: Different coordinate systems!

img_proj[0] might represent "redness" in image space
txt_proj[0] might represent "past_tense" in text space

These dimensions are UNRELATED!
Computing similarity between them gives random results
The similarity score will be close to 0 regardless of semantic content
```

**Analogy:**
```
This is like:
- Measuring image "temperature" in Fahrenheit  
- Measuring text "distance" in meters
- Then comparing 75°F with 100m

The numbers are unrelated even though both are valid measurements!
```

### The Correct Approach: Joint Embedding Space

✅ **CORRECT APPROACH**

Learn projections that map both modalities to a **shared semantic space**:

```python
# ✅ CORRECT: Joint training for shared space
import torch
import torch.nn as nn

class MultimodalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Learned projections to SHARED space
        self.img_projection = nn.Linear(2048, 256)  # Image → shared space
        self.txt_projection = nn.Linear(768, 256)   # Text → shared space
        
    def forward(self, img_features, txt_features):
        # Project to SAME coordinate system
        img_embedded = self.img_projection(img_features)  # 256D in shared space
        txt_embedded = self.txt_projection(txt_features)  # 256D in SAME shared space
        
        # Normalize for cosine similarity
        img_embedded = nn.functional.normalize(img_embedded, dim=-1)
        txt_embedded = nn.functional.normalize(txt_embedded, dim=-1)
        
        return img_embedded, txt_embedded

# Training process ensures both modalities map to MEANINGFUL shared space
# where similar content has similar representations
```

**Key insight:**
```
The projection matrices are learned jointly through training on paired data:
- Image of cat + text "cat" → should have high similarity
- Image of cat + text "dog" → should have low similarity

The learning process ensures:
- img_embedded[i] and txt_embedded[i] measure similar semantic properties
- Similarity calculations are meaningful
- Cross-modal retrieval actually works
```

### Proper Cross-Modal Similarity Calculation

Here's the complete correct approach:

```python
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
from torchvision import models
import numpy as np

class CorrectMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Pre-trained encoders
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()  # Remove final classifier
        
        # Learned projections to shared 256D space
        self.text_projection = nn.Linear(768, 256)
        self.image_projection = nn.Linear(2048, 256)
        
    def encode_text(self, text_inputs):
        # Extract BERT features
        with torch.no_grad():
            outputs = self.text_encoder(**text_inputs)
            text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Project to shared space
        text_embedded = self.text_projection(text_features)
        return nn.functional.normalize(text_embedded, dim=-1)
    
    def encode_image(self, image_inputs):
        # Extract ResNet features
        with torch.no_grad():
            image_features = self.image_encoder(image_inputs)
        
        # Project to shared space  
        image_embedded = self.image_projection(image_features)
        return nn.functional.normalize(image_embedded, dim=-1)
    
    def compute_similarity(self, image_embedded, text_embedded):
        # Now this is meaningful because both are in same coordinate system!
        return torch.mm(image_embedded, text_embedded.t())

# Usage example
model = CorrectMultimodalModel()

# For a cat image and "cat" text:
image_emb = model.encode_image(cat_image)      # Shape: (1, 256)
text_emb = model.encode_text(cat_text)         # Shape: (1, 256)

similarity = model.compute_similarity(image_emb, text_emb)  # Should be HIGH
print(f"Cat image vs cat text similarity: {similarity.item():.3f}")

# For a cat image and "dog" text:  
dog_text_emb = model.encode_text(dog_text)
similarity = model.compute_similarity(image_emb, dog_text_emb)  # Should be LOW
print(f"Cat image vs dog text similarity: {similarity.item():.3f}")
```

### Training the Joint Embedding

The key is **contrastive learning** with paired examples:

```python
def contrastive_loss(image_emb, text_emb, temperature=0.1):
    """
    image_emb: (batch_size, 256) - image embeddings
    text_emb: (batch_size, 256) - text embeddings
    Assumes corresponding indices are positive pairs
    """
    
    # Compute all pairwise similarities
    logits = torch.mm(image_emb, text_emb.t()) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(len(image_emb)).to(image_emb.device)
    
    # Cross-entropy loss
    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.t(), labels)
    
    return (loss_i2t + loss_t2i) / 2

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    images, texts = batch
    
    # Encode both modalities
    image_emb = model.encode_image(images)
    text_emb = model.encode_text(texts)
    
    # Contrastive loss
    loss = contrastive_loss(image_emb, text_emb)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 2.6 Evaluation and Debugging

### Checking Embedding Quality

```python
def evaluate_embeddings(model, test_data):
    """Evaluate quality of learned embeddings"""
    
    similarities_pos = []  # Similarities for positive pairs
    similarities_neg = []  # Similarities for negative pairs
    
    for image, pos_text, neg_text in test_data:
        img_emb = model.encode_image(image)
        pos_emb = model.encode_text(pos_text)
        neg_emb = model.encode_text(neg_text)
        
        pos_sim = torch.mm(img_emb, pos_emb.t()).item()
        neg_sim = torch.mm(img_emb, neg_emb.t()).item()
        
        similarities_pos.append(pos_sim)
        similarities_neg.append(neg_sim)
    
    print(f"Average positive similarity: {np.mean(similarities_pos):.3f}")
    print(f"Average negative similarity: {np.mean(similarities_neg):.3f}")
    print(f"Separation: {np.mean(similarities_pos) - np.mean(similarities_neg):.3f}")
    
    # Good embeddings should have:
    # - High positive similarities (> 0.3)
    # - Low negative similarities (< 0.1)  
    # - Large separation (> 0.2)
```

### Common Issues and Solutions

**Issue 1: All similarities are near zero**
```
Cause: Embeddings not properly normalized or poorly trained
Solution: Check normalization, increase learning rate, train longer
```

**Issue 2: Positive and negative similarities are similar**
```
Cause: Model hasn't learned to distinguish, need more training
Solution: Better negative sampling, harder negatives, more data
```

**Issue 3: Very high similarities for unrelated content**
```
Cause: Model collapsed to single representation
Solution: Better regularization, temperature tuning, diverse training data
```

## Key Takeaways

- **Embeddings** are vector representations that capture semantic meaning through geometry
- **Cosine similarity** is the preferred metric for comparing embeddings
- **Normalization** is essential when working with multimodal data
- **Different modalities** have different properties requiring specialized handling
- **❌ NEVER use separate PCA/dimensionality reduction** for different modalities
- **✅ ALWAYS learn joint projections** to shared semantic space through training
- **Feature extraction** is a pipeline from raw data to interpretable vectors
- **Cross-modal alignment** requires careful design and joint training

## Symbol and Variable Reference

Throughout this chapter, we use the following notation:

- **Bold lowercase** (**v**, **u**, **e**): Vectors
- **Regular lowercase** (d, n, i): Scalars and indices  
- **ℝ^d**: d-dimensional real vector space
- **||v||**: Vector magnitude/norm
- **v** · **u**: Dot product between vectors **v** and **u**
- cos(θ): Cosine of angle θ between vectors
- **e**_img: Image embedding vector
- **e**_txt: Text embedding vector
- **W**: Projection matrix (transforms one space to another)
- τ (tau): Temperature parameter in contrastive learning

## Further Reading

**Mathematical Foundations:**
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 2-3.
- Geron, A. (2019). *Hands-On Machine Learning*. O'Reilly. Chapter 8.

**Embedding Techniques:**
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv:1301.3781*.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *arXiv:1810.04805*.

**Multimodal Alignment:**
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.

---

**Previous**: [Chapter 1: Introduction to Multimodal Learning](chapter-01.md) | **Next**: [Chapter 3: Feature Representation for Each Modality](chapter-03.md) | **Home**: [Table of Contents](index.md)
