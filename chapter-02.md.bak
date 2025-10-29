# Chapter 2: Foundations and Core Concepts

---

**Previous**: [Chapter 1: Introduction to Multimodal Learning](chapter-01.md) | **Next**: [Chapter 3: Feature Representation for Each Modality](chapter-03.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand the mathematical foundations of multimodal learning
- Explain feature representation and embedding concepts
- Describe similarity metrics used in multimodal systems
- Understand modality normalization
- Apply fundamental concepts to real problems

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
Image of cat → e_image = [0.2, -0.5, 0.8, ..., 0.1] ∈ ℝ^2048
Text "cat" → e_text = [0.1, 0.3, -0.2, ..., 0.5] ∈ ℝ^768
```

**Why embeddings work:**

Embeddings capture semantic meaning through:
1. **Distance** - Similar items have vectors close together
2. **Direction** - Related concepts align directionally
3. **Magnitude** - Can encode confidence or importance
4. **Relationships** - Vector arithmetic can represent semantic operations

**Example - Word2Vec:**

```
Empirically discovered vector relationships:

king - man + woman ≈ queen

This works because:
- "king" and "queen" appear in similar contexts
- "man" and "woman" capture gender transformation
- Vector arithmetic preserves semantic relationships
```

### Similarity Metrics

**Core concept:** We need ways to measure how similar two embeddings are.

#### Cosine Similarity

**Definition:**
```
similarity(u, v) = (u · v) / (||u|| × ||v||)

where:
u · v = dot product
||u|| = L2 norm (magnitude)

Result: Score in [-1, 1]
```

**Geometric intuition:**

```
angle between vectors = arc_cos(similarity)

similarity = 1  → Same direction (identical)
similarity = 0  → Perpendicular (unrelated)
similarity = -1 → Opposite direction (contradictory)
```

**Why preferred for embeddings:**
- Invariant to magnitude (direction matters, not size)
- Computationally efficient
- Interpretable as angle between vectors
- Works well in high-dimensional spaces

**Example calculation:**

```python
import numpy as np

# Vectors
u = np.array([1, 0, 1, 0])
v = np.array([1, 0, 1, 0])

# Cosine similarity
dot_product = np.dot(u, v)  # 1*1 + 0*0 + 1*1 + 0*0 = 2
magnitude_u = np.linalg.norm(u)  # sqrt(1+0+1+0) = 1.414
magnitude_v = np.linalg.norm(v)  # sqrt(1+0+1+0) = 1.414

similarity = dot_product / (magnitude_u * magnitude_v)
# = 2 / (1.414 * 1.414) = 1.0 (identical vectors)
```

#### Euclidean Distance

**Definition:**
```
distance(u, v) = ||u - v|| = sqrt(Σ(u_i - v_i)^2)

Result: Score in [0, ∞)
- 0 means identical
- Larger means more different
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
distance(u, v) = Σ|u_i - v_i|

Result: Sum of absolute differences
```

**Advantages:**
- Computationally faster than Euclidean
- Robust to outliers
- Encourages sparsity

#### Dot Product Similarity

**Definition:**
```
similarity(u, v) = u · v = Σ(u_i × v_i)

Result: Score in (-∞, ∞)
- Unbounded, depends on magnitude
```

**When to use:**
- In contrastive learning (with temperature scaling)
- When magnitude carries meaning
- With normalized vectors (then equivalent to cosine)

### Normalization and Standardization

**Why normalize?**

Different modalities have different value ranges:

```
Image pixels: [0, 255] or [0, 1] after normalization
Text embeddings: [-1, 1] typically
Audio: [-1, 1] normalized

Without normalization:
- Similarity metrics behave unpredictably
- Model training becomes unstable
- Different modalities don't mix well
```

**Common normalization techniques:**

**1. Min-Max Normalization (Scaling)**
```
x_normalized = (x - min(x)) / (max(x) - min(x))

Result: All values in [0, 1]
Preserves: Relationships and shape of distribution
```

**2. Z-Score Normalization (Standardization)**
```
x_normalized = (x - mean(x)) / std(x)

Result: Mean = 0, Standard deviation = 1
Benefit: Works well for values with Gaussian distribution
```

**3. L2 Normalization (Unit Vector)**
```
x_normalized = x / ||x||

Result: Vector has magnitude 1
Property: Cosine similarity = dot product of L2-normalized vectors

Used in: CLIP, many embedding models
```

**Example - Normalizing image and text for comparison:**

```
Raw image features: [234, 1024, -500, ...]
Raw text features: [0.023, -0.18, 0.51, ...]

After L2 normalization:
Image: [0.234, 0.298, -0.145, ...] (magnitude = 1)
Text: [0.0412, -0.321, 0.911, ...] (magnitude = 1)

Now: Cosine similarity ≈ dot product
Both: Fair comparison despite different scales
```

## 2.2 Representing Data as Vectors

### Information-to-Vector Mapping

**Challenge:** Convert diverse data types to vectors

```
Text: "I love cats"
      ↓ (encoder)
      [0.23, -0.51, 0.82, ..., 0.15] (768D)

Image: [Cat photo]
       ↓ (encoder)
       [0.45, 0.12, -0.33, ..., 0.67] (2048D)

Audio: [Cat meow sound]
       ↓ (encoder)
       [0.11, -0.09, 0.54, ..., -0.22] (768D)
```

### Desiderata (Desired Properties) for Embeddings

A good embedding should have:

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
Contextual text (GPT)         1536-2048
Image (ResNet50)              2048
Image (Vision Transformer)    768
Audio (Wav2Vec2)              768
Multimodal shared space       256-512
```

**Dimensionality trade-off:**

```
Too small (e.g., 32D):
  ✗ Information loss
  ✗ Cannot capture complex relationships
  ✓ Fast computation
  ✓ Less memory
  ✓ Less prone to overfitting

Too large (e.g., 8192D):
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
        ↓ (project to 2D while preserving relationships)
2D visualization

Example - CLIP embeddings of common objects:

         "dog"
        /    \
      dog   cat -- "cat"
        \    /
         "cat image"

Similar items cluster together
Different items spread apart
```

### Semantic Relationships in Embedding Space

Embeddings capture semantic meaning:

```
Vector relationships:

┌─────────────────────────────────────────┐
│ SEMANTIC RELATIONSHIPS IN EMBEDDING     │
├─────────────────────────────────────────┤
│                                         │
│    queen                                │
│      •                                  │
│     /│                                  │
│    / │ (king - man + woman)             │
│   /  │                                  │
│  •───•───•                              │
│ king man woman                          │
│                                         │
│ Geometric interpretation:               │
│ - Parallelogram property               │
│ - Vector arithmetic = semantic ops      │
│                                         │
└─────────────────────────────────────────┘
```

**Real multimodal example - CLIP space:**

```
CLIP learns joint image-text space where:

Image of cat ────────────► [embedding]
    "A picture of a cat"──► [similar embedding]

Image of dog ────────────► [distant embedding]
    "A picture of a dog"──► [similar embedding]

Structure:
  - Cat images cluster with "cat" texts
  - Dog images cluster with "dog" texts
  - Cross-category items are far apart
```

## 2.4 Understanding Modality-Specific Properties

### Text as Vectors

**Properties:**
- Discrete symbols (words, subwords)
- Sequential structure (word order matters)
- Compositional (words combine into sentences)
- Abstract (can express concepts beyond physical)

**Representation levels:**

```
LEVEL 1 - Character level:
  "cat" → [c, a, t]
  Problem: Loses semantic meaning

LEVEL 2 - Word level:
  "The cat sat" → [[the], [cat], [sat]]
  Standard approach

LEVEL 3 - Subword level (BPE):
  "unbelievable" → [un, believable]
  Handles rare words

LEVEL 4 - Contextual:
  "The bank by the river" → [context-aware embeddings]
  Same word, different representation based on context
```

**Vector representation methods:**

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

METHOD 3 - Contextual embeddings:
  "The cat sat" →
  [context_embedding_1, context_embedding_2, context_embedding_3]
  Dimension = 768 (fixed) per token
  Benefit: Handles polysemy (multiple meanings)
```

**Key insight for multimodal:**
```
Text is interpretable!
Tokens map to meaningful units
We can reason about which text parts matter

This differs from image/audio where interpretation is harder
```

### Images as Vectors

**Properties:**
- Continuous values (pixel intensities)
- 2D spatial structure (nearby pixels correlated)
- Hierarchical features (edges → shapes → objects)
- Translational equivariance (object can be anywhere)

**Representation hierarchy:**

```
LEVEL 1 - Pixel level:
  Image 224×224×3 → 150,528 values
  Problem: Too high-dimensional, redundant

LEVEL 2 - Low-level features:
  Edges, corners, textures
  Extracted by early CNN layers

LEVEL 3 - Mid-level features:
  Shapes, parts, patterns
  From middle CNN layers

LEVEL 4 - High-level features:
  Objects, scenes, semantic concepts
  From final CNN layers

LEVEL 5 - Global representation:
  Single vector representing entire image
  From average pooling or final layer
```

**CNN feature hierarchy visualization:**

```
Input image (224×224×3)
        ↓
Layer 1: Edges [112×112×64]
        ↓
Layer 2: Textures [56×56×128]
        ↓
Layer 3: Shapes [28×28×256]
        ↓
Layer 4: Parts [14×14×512]
        ↓
Layer 5: Objects [7×7×512]
        ↓
Average Pool: [2048D vector]

Each layer extracts higher-level patterns
```

**Key insight for multimodal:**
```
Images are not immediately interpretable
The 2048 dimensions don't correspond to human-understandable concepts
(Except through attention visualization)

This makes image-text alignment challenging
Must learn mappings between image features and text concepts
```

### Audio as Vectors

**Properties:**
- 1D signal over time
- Varying frequency content
- Temporal structure (order matters)
- Perceptually motivated (frequency relevance to humans)

**Feature extraction process:**

```
Raw waveform (16kHz, 1 second) → 16,000 samples
            ↓
Split into frames (25ms each) → 40 frames
            ↓
Extract spectrogram → 40×513 (time × frequency)
            ↓
Apply Mel-scale + log → 40×128 (more human-like)
            ↓
Final MFCCs or spectrogram features
```

**Representation methods:**

```
METHOD 1 - MFCC (Mel-Frequency Cepstral Coefficients):
  Frame → Spectrum → Mel-scale → Cepstral → [39D per frame]
  Mimics human hearing
  Traditional approach

METHOD 2 - Spectrogram:
  Frame → FFT → Power spectrum → [513D per frame]
  All frequency information
  Used in deep learning

METHOD 3 - Learned features (Wav2Vec):
  Raw waveform → CNN → Quantized codes
  → Transformer → [768D learned representation]
  Modern approach
  Learns task-relevant features
```

**Key insight for multimodal:**
```
Audio is temporal but can be converted to spectral view
Frequency information is similar to visual features
(Both are "spectral" representations)

This can facilitate audio-visual alignment
(e.g., beat synchronization in music videos)
```

## 2.5 The Feature Extraction Pipeline

### General Pipeline Structure

```
Raw Data
    ↓
[Preprocessing]
  - Normalization
  - Augmentation
  - Format conversion
    ↓
[Feature Extraction]
  - Shallow features (SIFT, MFCC)
  - Or deep features (CNN, Transformer)
    ↓
[Post-processing]
  - Normalization (L2)
  - Dimensionality reduction
  - Feature selection
    ↓
[Embedding Vector]
  Ready for comparison or fusion
```

### Practical Considerations

**1. Batch processing efficiency:**
```
Don't extract features one at a time
Process batches for GPU efficiency

Batch size trade-offs:
  Larger batch: Better GPU utilization
  Smaller batch: Less memory, more iterations needed
  Typical: 32-256 depending on data type
```

**2. Feature caching:**
```
For large-scale retrieval systems:

Online phase (expensive):
  Extract features once, cache them

Retrieval phase (cheap):
  Query by similarity to cached features
  No need to re-extract

Example:
  E-commerce with 10M products
  Extract features once (hours of computation)
  Serve queries (milliseconds)
```

**3. Approximate similarity:**
```
Exact nearest neighbor search is slow for large datasets
Use approximate methods:

Hashing: Map similar embeddings to same hash bucket
LSH: Locality-Sensitive Hashing
FAISS: Facebook AI Similarity Search
ScaNN: Scalable Nearest Neighbors

Trade-off: Accuracy vs speed
```

## 2.6 Practical Example: Building Feature Extractors

### Image Feature Extractor

```python
import torch
import torchvision.models as models
from torchvision import transforms

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)
# Remove classification head
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Extract features
def extract_image_features(image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = model(image)

    # Flatten and L2-normalize
    features = features.flatten()
    features = features / torch.norm(features)

    return features.numpy()
```

### Text Feature Extractor

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

def extract_text_features(text):
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Extract embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # L2-normalize
    cls_embedding = cls_embedding / torch.norm(cls_embedding)

    return cls_embedding.numpy()
```

### Computing Similarity

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Extract features
image_feat = extract_image_features("cat.jpg")  # Shape: (2048,)
text_feat = extract_text_features("A cute cat")  # Shape: (768,)

# Problem: Different dimensions!
# Solution: Project to shared space

# Simple approach: L2 normalization and dimension reduction
from sklearn.decomposition import PCA

# Project both to 256D
pca_img = PCA(n_components=256)
pca_txt = PCA(n_components=256)

img_proj = pca_img.fit_transform(image_feat.reshape(1, -1))
txt_proj = pca_txt.fit_transform(text_feat.reshape(1, -1))

# Compute similarity
similarity = cosine_similarity(img_proj, txt_proj)
print(f"Similarity: {similarity[0][0]:.3f}")
```

## Key Takeaways

- **Embeddings** are vector representations that capture semantic meaning through geometry
- **Cosine similarity** is the preferred metric for comparing embeddings
- **Normalization** is essential when working with multimodal data
- **Different modalities** have different properties requiring specialized handling
- **Feature extraction** is a pipeline from raw data to interpretable vectors
- **Dimensionality** is a critical design choice balancing expressiveness and efficiency

## Exercises

**⭐ Beginner:**
1. Calculate cosine similarity between three vectors by hand
2. Explain why L2 normalization makes cosine similarity equal to dot product
3. Describe the properties of text, image, and audio modalities

**⭐⭐ Intermediate:**
4. Implement a similarity search system for 1000 pre-extracted embeddings
5. Compare cosine, Euclidean, and dot product similarity on sample data
6. Visualize embeddings using t-SNE and interpret clusters

**⭐⭐⭐ Advanced:**
7. Design a hybrid normalization scheme for multimodal data
8. Implement approximate nearest neighbor search with locality-sensitive hashing
9. Analyze how normalization affects downstream multimodal fusion

---

