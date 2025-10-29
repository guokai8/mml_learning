# Chapter 6: Attention Mechanisms in Multimodal Systems

---

**Previous**: [Chapter 5: Fusion Strategies](chapter-05.md) | **Next**: [Chapter 7: Contrastive Learning](chapter-07.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand attention mechanism fundamentals and intuition
- Implement scaled dot-product attention from scratch
- Understand multi-head attention and its role
- Apply cross-attention for multimodal fusion
- Visualize and interpret attention patterns
- Debug attention-based models
- Optimize attention for efficiency

## 6.1 Foundations of Attention

### The Problem Attention Solves

**Before attention (sequence-to-sequence models):**

```
Task: Translate English to French

English: "The quick brown fox jumps"
French:  "Le rapide renard brun saute"

RNN approach (encoder-decoder):

Encoder:
  Step 1: Process "The" → h₁
  Step 2: Process "quick" → h₂
  Step 3: Process "brown" → h₃
  Step 4: Process "fox" → h₄
  Step 5: Process "jumps" → h₅

  Final state: h₅ (tries to contain all information!)

Decoder:
  Uses only h₅ to generate entire translation

  Step 1: Generate "Le" from h₅
  Step 2: Generate "rapide" from h₅
  Step 3: Generate "renard" from h₅
  Step 4: Generate "brun" from h₅
  Step 5: Generate "saute" from h₅

Problem:
  ✗ All information bottlenecked into single vector h₅
  ✗ Cannot remember which input word to focus on
  ✗ Long sentences lose information
  ✗ No obvious alignment between input and output
```

**With attention:**

```
Encoder (same):
  Produces h₁, h₂, h₃, h₄, h₅

Decoder with attention:
  Step 1: Generate "Le"
    Where to look? "The" → attention to h₁
    Generate "Le" using context from h₁

  Step 2: Generate "rapide"
    Where to look? "quick" → attention to h₂
    Generate "rapide" using context from h₂

  Step 3: Generate "renard"
    Where to look? "brown" or "fox" → attention to h₃ and h₄
    Generate "renard" using blended context

  Step 4: Generate "brun"
    Where to look? "brown" → attention to h₃
    Generate "brun" using context from h₃

  Step 5: Generate "saute"
    Where to look? "jumps" → attention to h₅
    Generate "saute" using context from h₅

Benefits:
  ✓ Each output can look at relevant inputs
  ✓ No information bottleneck
  ✓ Explicit alignment learned
  ✓ Works better on long sequences
```

### Attention Intuition

**Analogy 1: Restaurant waiter**

```
Scene: Busy restaurant with 10 tables

Waiter's task: Serve Table 5

Process:
  1. Look around (attention mechanism)
  2. Pay attention to Table 5 specifically
  3. Focus 90% on Table 5
  4. Glance at nearby tables (10% split)
  5. Retrieve correct order from Table 5
  6. Serve Table 5

Attention score for each table:
  Table 1: 0.0  (far away)
  Table 2: 0.02 (nearby but not relevant)
  Table 3: 0.03
  Table 4: 0.05
  Table 5: 0.85 ← Focus here!
  Table 6: 0.03
  Table 7: 0.01
  Table 8: 0.01
  Table 9: 0.0
  Table 10: 0.0

Result: Service based on relevant information
```

**Analogy 2: Reading comprehension**

```
Question: "What did the fox do?"

Passage: "The quick brown fox jumped over the lazy dog"

Human reading process:
  1. Read question: "What did the fox do?"
  2. Scan passage
  3. Pay attention to parts mentioning "fox"
    - "brown fox" ← relevant
    - "jumped over" ← relevant
  4. Ignore irrelevant parts
    - "quick" ← less relevant
    - "lazy dog" ← not about fox
  5. Combine relevant information
  6. Answer: "jumped over the lazy dog"

Attention mechanism:
  Query: "fox" (what are we asking about?)
  Keys: [the, quick, brown, fox, jumped, over, the, lazy, dog]
  Attention: Focus on "fox", "jumped", "over"
  Values: Combine corresponding information
  Result: Answer the question
```

### Why Attention is Powerful

```
Key insight: Solve "what to look at" problem

Before attention:
  Model processes everything equally
  Must compress all info into fixed vector
  Gradient flow: Diluted through all positions

With attention:
  Model focuses on relevant information
  Can dynamically select what matters
  Gradient flow: Strong to important positions
  Learning: Faster and better
```

## 6.2 Scaled Dot-Product Attention (Complete)

### Mathematical Deep Dive

**Core formula:**

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

Components:
  Q (Query): (batch, seq_len, d_k)
  K (Key):   (batch, seq_len, d_k)
  V (Value): (batch, seq_len, d_v)

  Output: (batch, seq_len, d_v)

Dimensions typically:
  d_k = 64
  d_v = 64
  seq_len = 196 (for image patches) or 77 (for text tokens)
```

### Step-by-Step Computation

**Complete example with real numbers:**

```
Setup:
  Sequence: ["cat", "sat", "mat"]
  Query dimension: 2 (for simplicity)

Query vectors:
  Q = [
    [1.0, 0.5],      # "cat"
    [0.5, 1.0],      # "sat"
    [0.3, 0.7]       # "mat"
  ]

Key vectors (same as queries in self-attention):
  K = Q = [
    [1.0, 0.5],
    [0.5, 1.0],
    [0.3, 0.7]
  ]

Value vectors:
  V = [
    [2, 1],          # "cat" value
    [1, 2],          # "sat" value
    [1.5, 1.5]       # "mat" value
  ]

─────────────────────────────────────────

Step 1: Compute Q @ K^T (similarity)

Q @ K^T:
  Q[0] · K^T = [1.0, 0.5] @ [[1.0, 0.5, 0.3],
                              [0.5, 1.0, 0.7]]
             = [1.0*1.0 + 0.5*0.5,    1.0*0.5 + 0.5*1.0,   1.0*0.3 + 0.5*0.7]
             = [1.0 + 0.25,           0.5 + 0.5,           0.3 + 0.35]
             = [1.25,                 1.0,                 0.65]

  Q[1] · K^T = [0.5, 1.0] @ ...
             = [0.5*1.0 + 1.0*0.5,    0.5*0.5 + 1.0*1.0,   0.5*0.3 + 1.0*0.7]
             = [0.5 + 0.5,            0.25 + 1.0,          0.15 + 0.7]
             = [1.0,                  1.25,                0.85]

  Q[2] · K^T = [0.3, 0.7] @ ...
             = [0.3*1.0 + 0.7*0.5,    0.3*0.5 + 0.7*1.0,   0.3*0.3 + 0.7*0.7]
             = [0.3 + 0.35,           0.15 + 0.7,          0.09 + 0.49]
             = [0.65,                 0.85,                0.58]

Result: Similarity matrix
  [
    [1.25, 1.0,  0.65],
    [1.0,  1.25, 0.85],
    [0.65, 0.85, 0.58]
  ]

Interpretation:
  Position 0 most similar to: itself (1.25)
  Position 1 most similar to: itself (1.25)
  Position 2 most similar to: itself (0.58)

─────────────────────────────────────────

Step 2: Scale by 1/√d_k

d_k = 2, so √d_k = √2 ≈ 1.414

Scaled:
  [
    [1.25/1.414,  1.0/1.414,  0.65/1.414],
    [1.0/1.414,   1.25/1.414, 0.85/1.414],
    [0.65/1.414,  0.85/1.414, 0.58/1.414]
  ]
= [
    [0.884,  0.707, 0.460],
    [0.707,  0.884, 0.601],
    [0.460,  0.601, 0.410]
  ]

Why scale?
  Prevents dot product from getting too large
  Keeps gradients reasonable
  Stabilizes training

─────────────────────────────────────────

Step 3: Apply softmax

For position 0: [0.884, 0.707, 0.460]

First compute exponentials:
  e^0.884 ≈ 2.42
  e^0.707 ≈ 2.03
  e^0.460 ≈ 1.58
  Sum = 6.03

Softmax:
  [2.42/6.03,  2.03/6.03,  1.58/6.03]
= [0.401,      0.337,      0.262]

Interpretation:
  "cat" attends 40% to itself
  "cat" attends 34% to "sat"
  "cat" attends 26% to "mat"

For position 1: [0.707, 0.884, 0.601]
  e^0.707 ≈ 2.03
  e^0.884 ≈ 2.42
  e^0.601 ≈ 1.82
  Sum = 6.27

  Softmax: [0.324, 0.386, 0.290]

For position 2: [0.460, 0.601, 0.410]
  e^0.460 ≈ 1.58
  e^0.601 ≈ 1.82
  e^0.410 ≈ 1.51
  Sum = 4.91

  Softmax: [0.322, 0.371, 0.307]

Attention matrix (after softmax):
  [
    [0.401, 0.337, 0.262],
    [0.324, 0.386, 0.290],
    [0.322, 0.371, 0.307]
  ]

Each row sums to 1 ✓

─────────────────────────────────────────

Step 4: Apply to values

For position 0:
  attention_output[0] = 0.401 * V[0] + 0.337 * V[1] + 0.262 * V[2]
                      = 0.401 * [2, 1] + 0.337 * [1, 2] + 0.262 * [1.5, 1.5]
                      = [0.802, 0.401] + [0.337, 0.674] + [0.393, 0.393]
                      = [1.532, 1.468]

For position 1:
  attention_output[1] = 0.324 * [2, 1] + 0.386 * [1, 2] + 0.290 * [1.5, 1.5]
                      = [0.648, 0.324] + [0.386, 0.772] + [0.435, 0.435]
                      = [1.469, 1.531]

For position 2:
  attention_output[2] = 0.322 * [2, 1] + 0.371 * [1, 2] + 0.307 * [1.5, 1.5]
                      = [0.644, 0.322] + [0.371, 0.742] + [0.461, 0.461]
                      = [1.476, 1.525]

Final output:
  [
    [1.532, 1.468],
    [1.469, 1.531],
    [1.476, 1.525]
  ]

Interpretation:
  Each position now contains weighted combination of all values
  Weights determined by attention scores
  Result: Context-aware representations
```

### Implementation from Scratch

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention

    Args:
        Q: Query tensor (batch, seq_len, d_k)
        K: Key tensor (batch, seq_len, d_k)
        V: Value tensor (batch, seq_len, d_v)
        mask: Optional mask for positions to ignore

    Returns:
        output: Attention output (batch, seq_len, d_v)
        attention_weights: Attention scores (batch, seq_len, seq_len)
    """

    # Get dimension
    d_k = Q.shape[-1]

    # Step 1: Compute similarity scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)

    # Step 2: Scale by √d_k
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask (optional)
    if mask is not None:
        # Set masked positions to very negative number
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)

    # Handle NaN from softmax(-inf)
    attention_weights = torch.nan_to_num(attention_weights, 0.0)

    # Step 5: Apply to values
    output = torch.matmul(attention_weights, V)  # (batch, seq_len, d_v)

    return output, attention_weights

# Example usage
batch_size = 2
seq_len = 3
d_k = 2
d_v = 2

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"Output shape: {output.shape}")  # (2, 3, 2)
print(f"Attention weights shape: {attention_weights.shape}")  # (2, 3, 3)
print(f"Attention weights row sum: {attention_weights.sum(dim=-1)}")  # Should be all 1s
```

### Understanding Gradients

**Backpropagation through attention:**

```
Forward pass:
  Q @ K^T → Scale → Softmax → @ V

Backward pass:
  dL/dV: Direct gradient from output
  dL/dSoftmax: Chain from V gradient
  dL/dScale: Chain from softmax gradient
  dL/dScores: Chain from scale
  dL/dK, dL/dQ: Chain from scores

Key insight: Gradients flow through attention weights

If attention_weights[i,j] is high:
  Position i receives strong gradient from j
  Strong learning signal

If attention_weights[i,j] is low:
  Position i receives weak gradient from j
  Weak learning signal

Result: Model learns to attend to relevant positions
        through gradient flow
```

## 6.3 Multi-Head Attention

### Why Multiple Heads?

**Problem with single head:**

```
Single attention head learns one type of relationship

For text "The cat sat on the mat":

What if different relationships matter?
  Syntactic: Articles attend to nouns
  Semantic: Pronouns attend to antecedents
  Discourse: Later sentences attend to earlier context

Single head must learn all simultaneously
Difficult optimization problem
Limited capacity

Solution: Multiple heads
Each head learns different relationships
Parallel processing
Combine results
```

### Architecture

**Multi-head formula:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

h = number of heads (typically 8-16)
W_i^Q, W_i^K, W_i^V = Projection matrices for head i
W^O = Output projection
```

**Detailed breakdown:**

```
Input: (batch, seq_len, d_model)

For each head i = 1 to h:

  1. Project to smaller dimension
     Q_i = input @ W_i^Q     (batch, seq_len, d_k)
     K_i = input @ W_i^K     (batch, seq_len, d_k)
     V_i = input @ W_i^V     (batch, seq_len, d_v)

     Typical: d_model = 512, h = 8
              d_k = d_v = 512/8 = 64

  2. Compute attention
     head_i = Attention(Q_i, K_i, V_i)  (batch, seq_len, 64)

  3. Repeat for all 8 heads
     Result: 8 attention outputs
             Each (batch, seq_len, 64)

Concatenate all heads:
  Combined = [head_1 || head_2 || ... || head_8]
           (batch, seq_len, 512)

Output projection:
  output = Combined @ W^O
         (batch, seq_len, d_model)
```

### Implementation

```python
class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention layer"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query (batch, seq_len_q, d_model)
            K: Key (batch, seq_len_k, d_model)
            V: Value (batch, seq_len_v, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len_q, d_model)
        """
        batch_size = Q.shape[0]

        # Step 1: Linear projections
        Q = self.W_q(Q)  # (batch, seq_len_q, d_model)
        K = self.W_k(K)  # (batch, seq_len_k, d_model)
        V = self.W_v(V)  # (batch, seq_len_v, d_model)

        # Step 2: Reshape for multi-head attention
        # Split into h heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # (batch, num_heads, seq_len_q, d_k)

        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # (batch, num_heads, seq_len_k, d_k)

        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # (batch, num_heads, seq_len_v, d_k)

        # Step 3: Attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (batch, num_heads, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, 0.0)
        attention_weights = self.dropout(attention_weights)

        # Apply to values
        output = torch.matmul(attention_weights, V)
        # (batch, num_heads, seq_len_q, d_k)

        # Step 4: Concatenate heads
        output = output.transpose(1, 2).contiguous()
        # (batch, seq_len_q, num_heads, d_k)

        output = output.view(batch_size, -1, self.d_model)
        # (batch, seq_len_q, d_model)

        # Step 5: Output projection
        output = self.W_o(output)

        return output

# Example
mha = MultiHeadAttention(d_model=512, num_heads=8)
Q = torch.randn(2, 10, 512)  # batch_size=2, seq_len=10, d_model=512
K = torch.randn(2, 10, 512)
V = torch.randn(2, 10, 512)

output = mha(Q, K, V)
print(f"Output shape: {output.shape}")  # (2, 10, 512)
```

### Head Specialization

**What different heads learn:**

```
Example: Sentence "The cat sat on the mat"

Head 1 (Syntactic):
  Attention pattern:
    "The" → "cat" (article to noun)
    "sat" → "cat", "on", "mat" (verb to objects)
  Learns: Grammatical relationships

Head 2 (Semantic):
  Attention pattern:
    "cat" → "mat" (related nouns)
    "on" → "cat", "mat" (location relation)
  Learns: Semantic relationships

Head 3 (Long-range):
  Attention pattern:
    "mat" → "The" (distant words)
    "sat" → "cat" (key pairs)
  Learns: Global context

Head 4 (Rare/Noise):
  Attention pattern:
    "on" → "on", "the" (less obvious)
    "sat" → "sat" (self-attention)
  Learns: Residual patterns

Result: Complementary representations
        Ensemble of different perspectives
```

## 6.4 Cross-Attention for Multimodal Fusion

### Concept and Setup

**What is cross-attention?**

```
Self-attention:
  Q, K, V all from same source
  Example: Text attends to text
  "Which words are relevant to which other words?"

Cross-attention:
  Q from one modality, K/V from another
  Example: Text queries image features
  "Which image regions are relevant to this word?"

Benefits for multimodal:
  ① Explicit alignment between modalities
  ② Each modality can query the other
  ③ Information flow controlled by queries
```

### Example: Image-to-Text Cross-Attention

**Setup:**

```
Image: Visual features from CNN/ViT
  Shape: (batch, num_patches, d_image)
  Example: (2, 196, 2048) from ResNet50

Text: Token embeddings from BERT
  Shape: (batch, seq_len, d_text)
  Example: (2, 77, 768)

Goal: Text should understand image context
      Image should influence text processing
```

**Cross-attention computation:**

```
Query: Text embeddings
  Q = text_embeddings @ W_q
  Shape: (batch, seq_len_text, d_k)

Key/Value: Image features
  K = image_features @ W_k
  Shape: (batch, num_patches, d_k)

  V = image_features @ W_v
  Shape: (batch, num_patches, d_v)

Attention:
  scores = Q @ K^T / √d_k
  Shape: (batch, seq_len_text, num_patches)

  Interpretation:
    For each word (seq_len_text)
    How relevant is each image patch (num_patches)?

    Word "red" attends to:
      Red patches in image (high score)
      Other patches (low score)

Weighted sum:
  output = softmax(scores) @ V
  Shape: (batch, seq_len_text, d_v)

  Each word now contains information about
  relevant image regions
```

### Implementation

```python
class CrossAttention(torch.nn.Module):
    """Cross-attention between two modalities"""

    def __init__(self, d_q, d_k, d_v, num_heads=8):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_k // num_heads
        self.d_v = d_v // num_heads

        # Query projection (from modality 1)
        self.W_q = torch.nn.Linear(d_q, d_k)

        # Key/Value projection (from modality 2)
        self.W_k = torch.nn.Linear(d_k, d_k)
        self.W_v = torch.nn.Linear(d_k, d_v)

        # Output projection
        self.W_o = torch.nn.Linear(d_v, d_v)

    def forward(self, query_feats, key_value_feats, mask=None):
        """
        Args:
            query_feats: Queries from modality 1
                        (batch, len_q, d_q)
            key_value_feats: Keys/values from modality 2
                            (batch, len_k, d_k)
            mask: Optional mask

        Returns:
            output: (batch, len_q, d_v)
        """
        batch_size = query_feats.shape[0]

        # Project
        Q = self.W_q(query_feats)  # (batch, len_q, d_k)
        K = self.W_k(key_value_feats)  # (batch, len_k, d_k)
        V = self.W_v(key_value_feats)  # (batch, len_k, d_v)

        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.d_v)
        output = self.W_o(output)

        return output

# Example: Text attending to image
class ImageTextFusionLayer(torch.nn.Module):
    def __init__(self, d_text=768, d_image=2048):
        super().__init__()

        self.text_to_image = CrossAttention(
            d_q=d_text,
            d_k=d_image,
            d_v=d_image,
            num_heads=8
        )

        self.image_to_text = CrossAttention(
            d_q=d_image,
            d_k=d_text,
            d_v=d_text,
            num_heads=8
        )

    def forward(self, text_feats, image_feats):
        """
        Args:
            text_feats: (batch, len_text, d_text)
            image_feats: (batch, num_patches, d_image)

        Returns:
            text_out: Text enriched with image context
            image_out: Image enriched with text context
        """
        # Text queries image
        text_out = self.text_to_image(text_feats, image_feats)

        # Image queries text
        image_out = self.image_to_text(image_feats, text_feats)

        return text_out, image_out

# Usage
fusion_layer = ImageTextFusionLayer()

text_feats = torch.randn(2, 77, 768)  # Text features
image_feats = torch.randn(2, 196, 2048)  # Image patches

text_enhanced, image_enhanced = fusion_layer(text_feats, image_feats)

print(f"Text enhanced shape: {text_enhanced.shape}")  # (2, 77, 2048)
print(f"Image enhanced shape: {image_enhanced.shape}")  # (2, 196, 768)
```

### Bidirectional Fusion

**Why both directions matter:**

```
Text → Image only:
  Text understands image
  But image doesn't know what text is asking
  One-way flow

Image → Text only:
  Image influences text
  But text doesn't guide image processing
  Unbalanced

Both directions (bidirectional):
  Text and image mutually influence each other
  Balanced information flow
  Better alignment
```

**Architecture with bidirectional fusion:**

```python
class BidirectionalFusion(torch.nn.Module):
    """Bidirectional attention between text and image"""

    def __init__(self, d_text=768, d_image=2048, num_layers=6):
        super().__init__()

        self.num_layers = num_layers

        # Projections to common space
        self.text_project = torch.nn.Linear(d_text, 512)
        self.image_

-----

> continue

```python
        self.image_project = torch.nn.Linear(d_image, 512)

        # Layers of bidirectional attention
        self.layers = torch.nn.ModuleList([
            BidirectionalAttentionLayer(512, 512)
            for _ in range(num_layers)
        ])

    def forward(self, text_feats, image_feats):
        """
        Args:
            text_feats: (batch, len_text, d_text)
            image_feats: (batch, num_patches, d_image)

        Returns:
            text_out: (batch, len_text, 512)
            image_out: (batch, num_patches, 512)
        """
        # Project to common space
        text = self.text_project(text_feats)  # (batch, len_text, 512)
        image = self.image_project(image_feats)  # (batch, num_patches, 512)

        # Apply bidirectional fusion layers
        for layer in self.layers:
            text_new, image_new = layer(text, image)

            # Residual connections
            text = text + text_new
            image = image + image_new

        return text, image

class BidirectionalAttentionLayer(torch.nn.Module):
    """Single layer of bidirectional attention"""

    def __init__(self, d_model, d_ff):
        super().__init__()

        # Cross-attention: text queries image
        self.text_attn = torch.nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )

        # Cross-attention: image queries text
        self.image_attn = torch.nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )

        # Feed-forward networks
        self.text_ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model)
        )

        self.image_ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.text_norm1 = torch.nn.LayerNorm(d_model)
        self.text_norm2 = torch.nn.LayerNorm(d_model)
        self.image_norm1 = torch.nn.LayerNorm(d_model)
        self.image_norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, text, image):
        """
        Args:
            text: (batch, len_text, d_model)
            image: (batch, num_patches, d_model)

        Returns:
            text_out: (batch, len_text, d_model)
            image_out: (batch, num_patches, d_model)
        """
        # Text attends to image
        text_norm = self.text_norm1(text)
        text_attn_out, _ = self.text_attn(
            text_norm,  # Query
            image, image,  # Key, Value
            need_weights=False
        )
        text = text + text_attn_out

        # Text feed-forward
        text_norm = self.text_norm2(text)
        text = text + self.text_ff(text_norm)

        # Image attends to text
        image_norm = self.image_norm1(image)
        image_attn_out, _ = self.image_attn(
            image_norm,  # Query
            text, text,  # Key, Value
            need_weights=False
        )
        image = image + image_attn_out

        # Image feed-forward
        image_norm = self.image_norm2(image)
        image = image + self.image_ff(image_norm)

        return text, image

# Usage
fusion = BidirectionalFusion(d_text=768, d_image=2048, num_layers=6)

text_feats = torch.randn(2, 77, 768)
image_feats = torch.randn(2, 196, 2048)

text_out, image_out = fusion(text_feats, image_feats)

print(f"Text output shape: {text_out.shape}")  # (2, 77, 512)
print(f"Image output shape: {image_out.shape}")  # (2, 196, 512)
```

## 6.5 Attention Visualization and Interpretation

### Visualizing Attention Weights

**Text-to-text attention visualization:**

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(attention_weights, tokens, layer_idx=0, head_idx=0):
    """
    Visualize attention weights for a single layer and head

    Args:
        attention_weights: (num_layers, batch, num_heads, seq_len, seq_len)
        tokens: List of token strings
        layer_idx: Which layer to visualize
        head_idx: Which head to visualize
    """
    # Extract attention for specific layer and head
    attn = attention_weights[layer_idx, 0, head_idx]  # (seq_len, seq_len)
    attn = attn.detach().cpu().numpy()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn, cmap='viridis')

    # Set labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention weight')

    ax.set_title(f'Attention weights (Layer {layer_idx}, Head {head_idx})')
    ax.set_xlabel('Key (attended to)')
    ax.set_ylabel('Query (attending from)')

    plt.tight_layout()
    return fig

# Example usage
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
# attention_weights would come from model
fig = visualize_attention(attention_weights, tokens, layer_idx=0, head_idx=0)
plt.show()
```

**Pattern interpretation:**

```
Different attention patterns reveal model behavior:

Pattern 1: Diagonal (self-attention)
  ╱ (each token attends mostly to itself)
  Interpretation: Position focuses on its own context
  Meaning: Refines own representation

Pattern 2: Stripes (position-based)
  ║ ║ ║ (same columns attended)
  Interpretation: Multiple positions attend to same word
  Meaning: Word is important reference point

Pattern 3: Distributed
  ░ (uniform attention across sequence)
  Interpretation: No clear focus
  Meaning: Context comes from multiple sources

Pattern 4: Concentrated
  ◾ (attention on few positions)
  Interpretation: Clear focus
  Meaning: Strong alignment to specific positions
```

### Cross-Modal Attention Visualization

```python
def visualize_cross_attention(text_to_image_attn, text_tokens,
                              image_patches, head_idx=0):
    """
    Visualize what image regions text tokens attend to

    Args:
        text_to_image_attn: (seq_len_text, num_patches)
        text_tokens: List of text tokens
        image_patches: Could be image itself or placeholder
        head_idx: Which head (if multi-head)
    """
    attn = text_to_image_attn.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # For each text token, show what it attends to in image
    for i, token in enumerate(text_tokens[:6]):
        ax = axes[i]

        # Get attention for this token
        token_attn = attn[i]  # (num_patches,)

        # Reshape to image grid (assuming 14x14 patches for 196 total)
        grid_size = int(np.sqrt(len(token_attn)))
        attn_grid = token_attn.reshape(grid_size, grid_size)

        # Show as heatmap overlaid on image
        im = ax.imshow(attn_grid, cmap='hot')
        ax.set_title(f'Attention from "{token}"')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig

# Example usage
text_to_image = model.get_text_to_image_attention()
fig = visualize_cross_attention(text_to_image[0, 0],
                                text_tokens,
                                image)
plt.show()
```

## 6.6 Common Attention Patterns and Their Meanings

### Pattern 1: Positional Attention

**What it looks like:**
```
Attention matrix with clear bands:

       pos0  pos1  pos2  pos3  pos4
pos0   ▓▓░░░░░░░░░░░░░
pos1   ░▓▓░░░░░░░░░░░░
pos2   ░░▓▓░░░░░░░░░░░
pos3   ░░░▓▓░░░░░░░░░░
pos4   ░░░░▓▓░░░░░░░░░

(Each position mainly attends to neighbors)
```

**Interpretation:**
```
Model learns local structure
Effective for sequences with local dependencies
Examples: Natural language, time series
```

**When it occurs:**
```
Early layers of language models
Local relationships matter (syntax)
Limited context needed
```

### Pattern 2: Hub Attention

**What it looks like:**
```
One column has high values:

       pos0  pos1  pos2  pos3  pos4
pos0   ░▓▓▓▓▓▓▓▓▓▓▓▓▓
pos1   ░▓▓▓▓▓▓▓▓▓▓▓▓▓
pos2   ░▓▓▓▓▓▓▓▓▓▓▓▓▓
pos3   ░▓▓▓▓▓▓▓▓▓▓▓▓▓
pos4   ░▓▓▓▓▓▓▓▓▓▓▓▓▓

(All positions attend to pos1)
```

**Interpretation:**
```
"Hub" token is very important
All other tokens depend on it
Examples: [CLS] token in BERT, verb in sentence
```

**When it occurs:**
```
Late layers (higher abstraction)
Global information needed
One position summarizes all others
```

### Pattern 3: Diagonal + Off-Diagonal

**What it looks like:**
```
Self-attention plus other patterns:

       pos0  pos1  pos2  pos3  pos4
pos0   ▓▓░░░░░░░░▓░░░
pos1   ░▓▓░░░░░░░░▓░░
pos2   ░░▓▓░░░░░░░░▓░
pos3   ░░░▓▓░░░░░░░░▓
pos4   ░░░░▓▓░░░░░░░░

(Diagonal + secondary pattern)
```

**Interpretation:**
```
Self-attention + specific relationships
Example: Each word attends to self + its subject
Complex linguistic structure
```

### Pattern 4: Random/Noise

**What it looks like:**
```
No clear pattern:

       pos0  pos1  pos2  pos3  pos4
pos0   ▓░▓░▓░▓░▓░▓░▓░
pos1   ░▓░▓░▓░▓░▓░▓░
pos2   ▓░▓░▓░▓░▓░▓░▓
pos3   ░▓░▓░▓░▓░▓░▓░
pos4   ▓░▓░▓░▓░▓░▓░▓

(Uniform or random)
```

**Interpretation:**
```
Head not learning clear patterns
Could indicate:
  - Poor training
  - Redundant head
  - Learning different subspace
```

## 6.7 Debugging Attention Problems

### Problem 1: Attention Collapse

**Symptoms:**
```
Attention weights become nearly uniform
Example: [0.25, 0.25, 0.25, 0.25] instead of [0.8, 0.1, 0.05, 0.05]

Effects:
  No clear focus
  All positions equally weighted
  Information not well integrated
  Model performance poor
```

**Causes:**
```
① Temperature scaling issue
   Softmax too smooth
   All values similar

② Poorly initialized queries/keys
   Q and K nearly orthogonal
   All dot products similar

③ Gradients not flowing
   Attention not updating during training
```

**Solutions:**
```python
# Debug: Check attention entropy
def check_attention_collapse(attention_weights):
    """
    High entropy = collapse (uniform distribution)
    Low entropy = focused attention
    """
    # entropy = -sum(p * log(p))
    entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1)

    print(f"Attention entropy: {entropy.mean().item():.4f}")
    print(f"Max entropy (uniform): {torch.log(torch.tensor(attention_weights.shape[-1])).item():.4f}")

    if entropy.mean() > 0.8 * max_entropy:
        print("WARNING: Attention may be collapsing!")
        return True
    return False

# Fix: Increase temperature (smooth more)
# Or fix: Reduce temperature (sharpen more)
# Or fix: Check initialization
```

### Problem 2: Attention Not Converging

**Symptoms:**
```
Attention weights don't change during training
Always [0.333, 0.333, 0.333] for 3 positions

Effects:
  Model can't learn what to focus on
  No improvement over training
```

**Causes:**
```
① Learning rate too low
   Gradients too tiny
   No meaningful updates

② Attention parameters frozen
   Not being updated

③ No gradient signal
   Previous layers not helping
```

**Debugging code:**
```python
def debug_attention_convergence(model, initial_weights, final_weights):
    """Check if attention changed"""

    change = (final_weights - initial_weights).abs().mean()

    print(f"Attention weight change: {change.item():.6f}")

    if change < 1e-6:
        print("WARNING: Attention not converging!")

        # Check gradients
        for name, param in model.named_parameters():
            if 'attention' in name:
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    print(f"  {name}: grad_norm = {grad_norm.item():.6f}")
                else:
                    print(f"  {name}: NO GRADIENT")

        return False
    return True
```

### Problem 3: Misaligned Cross-Attention

**Symptoms:**
```
Cross-attention between modalities doesn't make sense
Example: Word "red" attends to random image patches, not red regions

Effects:
  Poor multimodal alignment
  Model can't understand relationship between modalities
```

**Debugging:**
```python
def analyze_cross_attention_alignment(text_tokens, image_labels,
                                     cross_attn_weights):
    """
    Check if cross-attention makes semantic sense

    Args:
        text_tokens: ['red', 'cat', 'on', 'mat']
        image_labels: ['red_region', 'cat_region', 'ground', 'background']
        cross_attn_weights: (len_text, num_patches)
    """

    for i, token in enumerate(text_tokens):
        attn = cross_attn_weights[i]  # Attention for this token
        top_indices = torch.topk(attn, k=3).indices  # Top 3 attended regions

        attended_regions = [image_labels[idx] for idx in top_indices]

        print(f"Token '{token}' attends to: {attended_regions}")

        # Simple heuristic: check if token and attended regions match
        if token in ' '.join(attended_regions).lower():
            print(f"  ✓ Makes sense!")
        else:
            print(f"  ✗ Misaligned!")
```

## 6.8 Attention Efficiency Optimizations

### Challenge: Quadratic Complexity

**Problem:**
```
Attention complexity: O(n²) where n = sequence length

Examples:
  n = 100: 10,000 operations
  n = 1000: 1,000,000 operations
  n = 10,000: 100,000,000 operations

For images with 196 patches: Manageable
For long documents with 4096 tokens: Problematic
For videos with 1000+ frames: Very difficult
```

### Solution 1: Sparse Attention

**Idea: Don't attend to all positions**

```python
class SparseAttention(torch.nn.Module):
    """Attention with sparse connections"""

    def __init__(self, window_size=32):
        super().__init__()
        self.window_size = window_size

    def forward(self, Q, K, V):
        """
        Only attend to nearby positions

        Each position attends to:
          - Itself
          - window_size//2 positions before
          - window_size//2 positions after
        """
        seq_len = Q.shape[1]

        # Create sparse mask
        mask = torch.ones(seq_len, seq_len, device=Q.device)

        for i in range(seq_len):
            # Mask everything outside window
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2)
            mask[i, :start] = 0
            mask[i, end:] = 0

        # Standard attention with mask
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output

# Complexity: O(n * window_size) instead of O(n²)
```

### Solution 2: Linear Attention

**Idea: Approximate softmax with kernel methods**

```python
class LinearAttention(torch.nn.Module):
    """Linear complexity attention"""

    def forward(self, Q, K, V):
        """
        Standard attention:
          Attention(Q,K,V) = softmax(QK^T) @ V
          Complexity: O(n²)

        Linear attention:
          Approximate softmax with kernel
          φ(QK^T) can be computed differently
          Complexity: O(n)
        """

        # Apply kernel function (e.g., elu + 1)
        Q_proj = torch.nn.functional.elu(Q) + 1  # Ensure positivity
        K_proj = torch.nn.functional.elu(K) + 1

        # Rewrite attention:
        # standard: softmax(QK^T) @ V
        # linear: φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)

        numerator = torch.einsum('bne,bnd->bnd', K_proj, V)  # (batch, seq, d)
        numerator = torch.einsum('bnd,bne->bnd', Q_proj, numerator)

        denominator = torch.einsum('bne,bn->bne', Q_proj,
                                   K_proj.sum(dim=1))  # (batch, seq, 1)
        denominator = denominator + 1e-6  # Avoid division by zero

        output = numerator / denominator

        return output

# Complexity: O(n * d²) where d is embedding dim
# For n >> d: Linear in n
```

### Solution 3: Flash Attention

**Idea: GPU-friendly attention computation**

```
Standard attention:
  1. Compute QK^T: O(n²) memory
  2. Apply softmax
  3. Multiply by V

Flash Attention:
  1. Compute attention in blocks
  2. Fuse operations (CUDA)
  3. Reduce memory and computation

Result:
  2-4× faster
  Less memory
  Same result

Implementation: Use existing libraries
  torch.nn.functional.scaled_dot_product_attention  (PyTorch 2.0+)
  flash-attn package
```

### Practical Optimization Example

```python
# Before: Standard attention
attention = torch.nn.MultiheadAttention(d_model=512, num_heads=8)

# Memory: O(batch * seq_len²)
# Speed: Slower

# After: Optimized attention
class OptimizedAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        # Option 1: Use Flash Attention (PyTorch 2.0+)
        self.use_flash = True

        # Option 2: Use sparse attention for long sequences
        if seq_len > 1000:
            self.attention = SparseAttention(window_size=64)
        else:
            self.attention = torch.nn.MultiheadAttention(d_model, num_heads)

    def forward(self, Q, K, V):
        if self.use_flash:
            return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        else:
            return self.attention(Q, K, V)
```

## Key Takeaways

- **Attention solves "what to look at" problem** efficiently
- **Scaled dot-product is the foundation** - normalize by √d_k
- **Multi-head attention learns diverse patterns** in parallel
- **Cross-attention connects modalities** bidirectionally
- **Visualization reveals model behavior** - debug with patterns
- **Efficiency matters** - use sparse, linear, or flash attention for long sequences

## Exercises

**⭐ Beginner:**
1. Implement scaled dot-product attention by hand
2. Visualize attention weights from pre-trained model
3. Understand what each attention head specializes in

**⭐⭐ Intermediate:**
4. Build cross-attention fusion layer
5. Implement bidirectional attention
6. Debug attention collapse in custom model

**⭐⭐⭐ Advanced:**
7. Implement sparse attention
8. Optimize attention with flash mechanisms
9. Analyze cross-modal alignment quality

---

