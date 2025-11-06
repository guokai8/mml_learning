# Chapter 8: Transformer Architecture

---

**Previous**: [Chapter 7: Contrastive Learning](chapter-07.md) | **Next**: [Chapter 9: Generative Models for Multimodal Data](chapter-09.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand why transformers replaced RNNs for sequence modeling
- Implement scaled dot-product attention from scratch
- Explain the role of positional encoding
- Understand the complete transformer architecture
- Apply transformers to multimodal problems

## 8.1 Motivation: Why Transformers?

### Sequential Processing Limitations

**RNN Problem Example:**

```
Processing sequence: w₁, w₂, w₃, w₄, w₅

RNN forward pass (sequential):
  h₀ = initialization
  h₁ = RNN(w₁, h₀)  ← Must wait for h₀
  h₂ = RNN(w₂, h₁)  ← Must wait for h₁
  h₃ = RNN(w₃, h₂)  ← Must wait for h₂
  h₄ = RNN(w₄, h₃)  ← Must wait for h₃
  h₅ = RNN(w₅, h₄)  ← Must wait for h₄

Problems:
① Cannot parallelize
   Each step depends on previous
   Sequential bottleneck

② Gradient flow issues during backpropagation
   Let's analyze this carefully using the chain rule...
```

### Gradient Flow Analysis (Chain Rule)

**Mathematical derivation of the vanishing gradient problem:**

For an RNN with hidden states h₀, h₁, h₂, ..., h_T, let's compute the gradient of the loss L with respect to the initial hidden state h₀.

Using the chain rule:
```
∂L/∂h₀ = ∂L/∂h_T × ∂h_T/∂h₀

To compute ∂h_T/∂h₀, we apply the chain rule through all intermediate states:

∂h_T/∂h₀ = (∂h_T/∂h_{T-1}) × (∂h_{T-1}/∂h_{T-2}) × ... × (∂h₂/∂h₁) × (∂h₁/∂h₀)

This is a product of T partial derivatives!
```

**Why this causes problems:**

```
For a typical RNN transition function:
h_t = tanh(W_h h_{t-1} + W_x x_t + b)

The partial derivative is:
∂h_t/∂h_{t-1} = diag(tanh'(W_h h_{t-1} + W_x x_t + b)) × W_h

Where tanh'(z) = 1 - tanh²(z) ∈ (0, 1]

Key insights:
1. tanh'(z) ≤ 1 always
2. ||W_h|| (matrix norm) is typically ≤ 1 to prevent exploding gradients
3. Therefore ||∂h_t/∂h_{t-1}|| ≤ ||W_h|| ≤ 1

Final gradient magnitude:
||∂h_T/∂h₀|| ≤ ∏(t=1 to T) ||∂h_t/∂h_{t-1}|| ≤ (max_norm)^T

If max_norm = 0.9 and T = 100:
Gradient magnitude ≤ 0.9^100 ≈ 2.7 × 10^-5 (vanishing!)

If max_norm = 1.1 and T = 100:
Gradient magnitude ≥ 1.1^100 ≈ 1.4 × 10^4 (exploding!)
```

**The fundamental issue:**
```
Long sequences require gradients to flow through many multiplicative steps.
Each step either shrinks (vanishing) or grows (exploding) the gradient.
Stable training requires delicate balance that's hard to achieve.

Result: RNNs struggle with long-term dependencies
```

### CNN Limitations for Sequences

**CNN characteristics:**

```
Local receptive field:
  3×3 kernel sees 3 neighbors in 1D sequence
  To connect positions distance d apart:
  Need O(log d) layers with exponential growth
  Or O(d) layers with linear growth

For long sequences (length 1000):
  Need many layers to see full context
  Computational depth becomes prohibitive
  
Path length between distant positions:
  RNN: O(n) sequential steps
  CNN: O(log n) with careful design, O(n) worst case
```

### The Transformer Solution

**Key insight:** Use attention to connect all positions directly

```
Attention mechanism:
  Path length between ANY two positions: O(1)
  All connections computed in parallel
  No sequential dependencies
  
Computational benefits:
  ✓ Parallelizable across sequence length
  ✓ Constant path length for long-range dependencies
  ✓ No gradient vanishing through temporal steps
  ✓ More interpretable attention patterns
```

## 8.2 Attention Mechanism Deep Dive

### Scaled Dot-Product Attention

**Mathematical formulation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Where:
Q ∈ ℝ^(n×d_k) = Query matrix (what we're looking for)
K ∈ ℝ^(m×d_k) = Key matrix (what we're looking at)  
V ∈ ℝ^(m×d_v) = Value matrix (what we retrieve)
n = sequence length of queries
m = sequence length of keys/values
d_k = dimension of queries and keys
d_v = dimension of values
```

**Step-by-step computation:**

```
Step 1: Compute similarity scores
  Scores = Q @ K^T                    # Shape: (n, m)
  
  Each element Scores[i,j] = Q[i,:] · K[j,:]
  Interpretation: How much query i "matches" key j

Step 2: Scale by √d_k
  Scaled_scores = Scores / √d_k
  
  Purpose: Prevent dot products from becoming too large
  (Details explained below)

Step 3: Apply softmax
  Attention_weights = softmax(Scaled_scores)  # Shape: (n, m)
  
  Each row sums to 1: Σⱼ Attention_weights[i,j] = 1
  Interpretation: Probability distribution over keys for each query

Step 4: Compute weighted sum of values
  Output = Attention_weights @ V      # Shape: (n, d_v)
  
  Each output[i,:] = Σⱼ Attention_weights[i,j] × V[j,:]
  Interpretation: Weighted combination of values based on attention
```

### Concrete Example

**Setup:**
```
Sequence: ["The", "cat", "sat"]
Embedding dimension: d_k = 4

Query vectors (what each word is "asking about"):
  Q = [
    [0.1, 0.2, 0.3, 0.1],    # "The" 
    [0.4, 0.1, 0.2, 0.3],    # "cat"
    [0.2, 0.3, 0.1, 0.4]     # "sat"
  ]

Key vectors (what each word "offers"):
  K = [
    [0.2, 0.1, 0.4, 0.2],    # "The"
    [0.3, 0.4, 0.1, 0.3],    # "cat"  
    [0.1, 0.2, 0.3, 0.4]     # "sat"
  ]

Value vectors (what each word "contains"):
  V = [
    [1.0, 0.0],              # "The" content
    [0.0, 1.0],              # "cat" content
    [0.5, 0.5]               # "sat" content
  ]
```

**Step 1: Compute scores**
```
Scores = Q @ K^T = [
  [0.1×0.2 + 0.2×0.1 + 0.3×0.4 + 0.1×0.2,  # "The" looks at "The"
   0.1×0.3 + 0.2×0.4 + 0.3×0.1 + 0.1×0.3,  # "The" looks at "cat"
   0.1×0.1 + 0.2×0.2 + 0.3×0.3 + 0.1×0.4], # "The" looks at "sat"
  [...],  # "cat" row
  [...]   # "sat" row
] = [
  [0.16, 0.14, 0.18],
  [0.25, 0.30, 0.32],
  [0.22, 0.28, 0.26]
]
```

**Step 2: Scale**
```
Scaled = Scores / √4 = Scores / 2 = [
  [0.08, 0.07, 0.09],
  [0.125, 0.15, 0.16],
  [0.11, 0.14, 0.13]
]
```

**Step 3: Softmax**
```
For row 1 (word "The"):
exp_values = [exp(0.08), exp(0.07), exp(0.09)] = [1.083, 1.073, 1.094]
sum_exp = 1.083 + 1.073 + 1.094 = 3.25
attention_weights[0,:] = [1.083/3.25, 1.073/3.25, 1.094/3.25] 
                       = [0.333, 0.330, 0.337]

Similarly for other rows:
Attention_weights = [
  [0.333, 0.330, 0.337],   # "The" attention distribution
  [0.312, 0.342, 0.346],   # "cat" attention distribution  
  [0.325, 0.349, 0.326]    # "sat" attention distribution
]
```

**Step 4: Output**
```
Output = Attention_weights @ V = [
  [0.333×1.0 + 0.330×0.0 + 0.337×0.5,    # "The" output dim 1
   0.333×0.0 + 0.330×1.0 + 0.337×0.5],   # "The" output dim 2
  [...],  # "cat" output
  [...]   # "sat" output
] = [
  [0.502, 0.498],
  [0.481, 0.519],
  [0.488, 0.512]
]

Interpretation:
- "The" gets roughly equal mix of all word contents
- "cat" pays slightly more attention to later words
- "sat" pays balanced attention across the sequence
```

### Why Scale by √d_k?

**Mathematical reasoning:**

```
Problem: Large dot products → extreme softmax values

Assume Q[i,:] and K[j,:] have independent components ~ N(0,1)
Then Q[i,:] · K[j,:] = Σₖ Q[i,k] × K[j,k]

Expected value: E[Q[i,:] · K[j,:]] = 0
Variance: Var[Q[i,:] · K[j,:]] = d_k

Standard deviation: √d_k

Example with d_k = 64:
Dot products have std = 8
Values often in range [-24, 24]

Softmax of large values:
exp(24) / (exp(24) + exp(20) + exp(16)) ≈ 1.0
exp(20) / (exp(24) + exp(20) + exp(16)) ≈ 0.0  
exp(16) / (exp(24) + exp(20) + exp(16)) ≈ 0.0

Result: Attention becomes nearly one-hot (peaks at single position)
Gradients become very small (softmax saturation)
```

**Solution: Scale by √d_k**
```
Scaled dot products: (Q[i,:] · K[j,:]) / √d_k
New standard deviation: √d_k / √d_k = 1

Keeps dot products in reasonable range [-3, 3] typically
Softmax remains smooth and differentiable
Gradients flow properly during training
```

### Implementation

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query tensor (batch_size, seq_len_q, d_k)
        K: Key tensor (batch_size, seq_len_k, d_k)  
        V: Value tensor (batch_size, seq_len_v, d_v)
        mask: Optional mask tensor (batch_size, seq_len_q, seq_len_k)
    
    Returns:
        output: Attention output (batch_size, seq_len_q, d_v)
        attention_weights: Attention scores (batch_size, seq_len_q, seq_len_k)
    """
    
    # Get dimension
    d_k = Q.shape[-1]
    
    # Step 1: Compute similarity scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len_q, seq_len_k)
    
    # Step 2: Scale
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 4: Softmax
    attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len_q, seq_len_k)
    
    # Step 5: Apply attention to values
    output = torch.matmul(attention_weights, V)  # (batch, seq_len_q, d_v)
    
    return output, attention_weights

# Example usage
batch_size, seq_len, d_model = 2, 5, 64
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)  
V = torch.randn(batch_size, seq_len, d_model)

output, attention = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")      # (2, 5, 64)
print(f"Attention shape: {attention.shape}") # (2, 5, 5)
```

### Gradient Flow in Attention

**Why attention helps with gradient flow:**

```
Forward pass:
  Q @ K^T → Scale → Softmax → @ V

Backward pass (chain rule):
  ∂L/∂V: Direct gradient from output
  ∂L/∂(attention_weights): Computed from V gradient  
  ∂L/∂(scores): Computed from softmax gradient
  ∂L/∂K, ∂L/∂Q: Computed from scores gradient

Key insight: Gradients flow directly through attention weights

If attention_weights[i,j] is high:
  Position i receives strong gradient signal from position j
  Direct connection enables strong learning

If attention_weights[i,j] is low:
  Position i receives weak gradient signal from position j
  But connection still exists (not zero)

Result: Every position can attend to every other position
        Gradient paths length = 1 (constant!)
        No vanishing through sequential multiplication
```

## 8.3 Multi-Head Attention

### Why Multiple Heads?

**Single attention limitation:**
```
One attention mechanism learns ONE type of relationship:
- Might focus on syntactic dependencies ("cat" → "sat")
- Or semantic similarities ("cat" → "animal")  
- Or positional patterns ("first" → "word")

But we need MULTIPLE types of relationships simultaneously!
```

**Multi-head solution:**
```
Learn H different attention functions in parallel:
- Head 1: Syntactic relationships
- Head 2: Semantic relationships  
- Head 3: Positional patterns
- Head 4: Long-range dependencies
- ...

Each head can specialize in different patterns
```

### Mathematical Formulation

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

Parameters:
  W_i^Q ∈ ℝ^(d_model × d_k)  # Query projection for head i
  W_i^K ∈ ℝ^(d_model × d_k)  # Key projection for head i  
  W_i^V ∈ ℝ^(d_model × d_v)  # Value projection for head i
  W^O ∈ ℝ^(h×d_v × d_model)  # Output projection

Typical choices:
  h = 8 heads
  d_k = d_v = d_model / h = 64 (for d_model = 512)
```

**Process:**
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
     head_i = Attention(Q_i, K_i, V_i)  (batch, seq_len, d_v)

3. Concatenate heads
   concat = Concat(head₁, head₂, ..., head_h)  (batch, seq_len, h×d_v)

4. Final projection  
   output = concat @ W^O                        (batch, seq_len, d_model)
```

**Example - 8 heads with d_model=512:**

```
Each head operates in d_k = 512/8 = 64 dimensional space
8 different projection matrices per Q, K, V

Result:
  8 independent attention mechanisms
  Each learns different patterns
  Combined through learned output projection

Computational cost:
  Same as single large head (512×512)
  But more expressive due to multiple subspaces
```

### Implementation

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for all heads (computed in parallel)
        self.W_q = nn.Linear(d_model, d_model)  # All Q projections
        self.W_k = nn.Linear(d_model, d_model)  # All K projections  
        self.W_v = nn.Linear(d_model, d_model)  # All V projections
        self.W_o = nn.Linear(d_model, d_model)  # Output projection
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)    # (batch, seq_len, d_model)
        V = self.W_v(value)  # (batch, seq_len, d_model)
        
        # 2. Reshape for multi-head: (batch, seq_len, n_heads, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # 3. Transpose: (batch, n_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)  
        V = V.transpose(1, 2)
        
        # 4. Apply attention to each head
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 5. Concatenate heads: (batch, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        
        # 6. Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights

# Usage
model = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output, attention = model(x, x, x)
print(f"Output shape: {output.shape}")      # (2, 10, 512)
print(f"Attention shape: {attention.shape}") # (2, 8, 10, 10)
```

## 8.4 Positional Encoding

### The Position Problem

**Issue:** Attention is permutation-invariant
```
Sequences: "cat sat mat" and "mat cat sat"
→ Same attention outputs (ignoring word embeddings)
→ No notion of word order!

But word order matters:
- "dog bit man" ≠ "man bit dog" 
- "not happy" ≠ "happy not"
```

**Solution:** Add positional information to embeddings

### Sinusoidal Positional Encoding

**Formula:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
  pos = position in sequence (0, 1, 2, ...)
  i = dimension index (0, 1, 2, ..., d_model/2)

Example: Position 0, dimension 0
  PE(0, 0) = sin(0) = 0

  Position 1, dimension 0:
  PE(1, 0) = sin(1 / 10000^0) = sin(1) ≈ 0.84

Properties:
  ① Different positions have different encodings
  ② Patterns repeat at different frequencies
  ③ Model can learn relative positions
  ④ Extrapolates to longer sequences than seen in training
```

**Intuition:**
```
Each dimension oscillates at different frequency:
- Dimension 0: Changes every position (high frequency)
- Dimension 1: Changes every 2 positions  
- Dimension 2: Changes every 4 positions
- ...
- Dimension d-1: Changes very slowly (low frequency)

Like binary counting:
Position 0: [0, 0, 0, 0]
Position 1: [1, 0, 0, 0]  
Position 2: [0, 1, 0, 0]
Position 3: [1, 1, 0, 0]
Position 4: [0, 0, 1, 0]

But with smooth sinusoidal patterns instead of discrete bits
```

### Implementation

```python
import torch
import math

def get_positional_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings"""
    
    # Create position indices: [0, 1, 2, ..., seq_len-1]
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    
    # Create dimension indices: [0, 2, 4, ..., d_model-2]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                        -(math.log(10000.0) / d_model))
    
    # Initialize positional encoding matrix
    pe = torch.zeros(seq_len, d_model)
    
    # Apply sine to even dimensions
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Apply cosine to odd dimensions  
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Example
seq_len, d_model = 10, 512
pe = get_positional_encoding(seq_len, d_model)
print(f"Positional encoding shape: {pe.shape}")  # (10, 512)

# Visualize first few dimensions
print("Position encoding for first 3 positions, first 8 dimensions:")
print(pe[:3, :8])
```

**Adding to embeddings:**
```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encodings (fixed, not learned)
        self.register_buffer('pos_encoding', 
                           get_positional_encoding(max_seq_len, d_model))
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Word embeddings
        word_emb = self.word_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        pos_emb = self.pos_encoding[:seq_len, :]
        
        return word_emb + pos_emb
```

## 8.5 Complete Transformer Architecture

### Transformer Encoder

**Single encoder layer:**
```
Input: (batch, seq_len, d_model)
    ↓
Multi-Head Self-Attention
    ↓  
Add & Norm (Residual connection + Layer normalization)
    ↓
Position-wise Feed-Forward Network
    ↓
Add & Norm (Residual connection + Layer normalization)  
    ↓
Output: (batch, seq_len, d_model)
```

**Detailed components:**

```
Multi-Head Self-Attention:
  input → Q, K, V (all same input)
  Allows each position to attend to all positions
  
Residual connection:
  output = attention_output + input
  
  Why?
  ① Preserves original information
  ② Enables deep networks (gradient flows directly)
  ③ Output can learn "residual" (difference)

Layer Normalization:
  Normalize across feature dimension
  
  mean = mean(x along d_model dimension)
  variance = var(x along d_model dimension)
  normalized = (x - mean) / sqrt(variance + epsilon)
  output = γ * normalized + β
  
  γ, β are learnable parameters

Position-wise FFN:
  FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
  
  Typical: d_model=512, d_ff=2048
  Applied to each position independently
  Same network for all positions
```

### Implementation

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        
        # Position-wise feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, 
                 max_seq_len=1000, dropout=0.1):
        super().__init__()
        
        # Embedding layer
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

# Usage
model = TransformerEncoder(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6
)

# Example input
batch_size, seq_len = 2, 20
input_ids = torch.randint(0, 10000, (batch_size, seq_len))

output = model(input_ids)
print(f"Output shape: {output.shape}")  # (2, 20, 512)
```

## 8.6 Transformer for Multimodal Applications

### Cross-Modal Attention

**Key idea:** Attention between different modalities

```
Standard self-attention:
  Q, K, V all from same modality
  
Cross-modal attention:
  Q from modality A (e.g., text)
  K, V from modality B (e.g., image)
  
Result: Text tokens attend to image regions
```

**Example - Image Captioning:**
```
Image: Processed by CNN → image features (49 regions × 2048D)
Text: Previous words → text features (seq_len × 512D)

Cross attention:
  Q = text features (what text is asking about)
  K = image features (what image regions are available)  
  V = image features (what image regions contain)
  
Output: Text representation informed by relevant image regions
```

### Multimodal Transformer Architecture

```python
class MultimodalTransformer(nn.Module):
    def __init__(self, text_vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        
        # Text encoding
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        
        # Image encoding  
        self.image_projection = nn.Linear(2048, d_model)  # ResNet → d_model
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Cross-modal attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
    def forward(self, text_tokens, image_features):
        # Encode text
        text_emb = self.text_embedding(text_tokens)  # (batch, text_len, d_model)
        text_emb += self.pos_encoding[:text_tokens.size(1)]
        
        # Encode image  
        img_emb = self.image_projection(image_features)  # (batch, img_regions, d_model)
        
        # Cross-modal processing
        for layer in self.cross_attention_layers:
            text_emb, img_emb = layer(text_emb, img_emb)
            
        return text_emb, img_emb

class CrossModalAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        # Text self-attention
        self.text_self_attn = MultiHeadAttention(d_model, n_heads)
        
        # Image self-attention  
        self.img_self_attn = MultiHeadAttention(d_model, n_heads)
        
        # Cross-attention: text attends to image
        self.text_to_img_attn = MultiHeadAttention(d_model, n_heads)
        
        # Cross-attention: image attends to text
        self.img_to_text_attn = MultiHeadAttention(d_model, n_heads)
        
        # Layer norms and FFNs
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)
        self.img_norm1 = nn.LayerNorm(d_model)
        self.img_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, text_emb, img_emb):
        # Text self-attention
        text_self, _ = self.text_self_attn(text_emb, text_emb, text_emb)
        text_emb = self.text_norm1(text_emb + text_self)
        
        # Text-to-image cross-attention
        text_cross, _ = self.text_to_img_attn(text_emb, img_emb, img_emb)
        text_emb = self.text_norm2(text_emb + text_cross)
        
        # Similar for image...
        
        return text_emb, img_emb
```

## 8.7 Key Advantages and Limitations

### Advantages

```
✓ Parallelizable: All positions processed simultaneously
✓ Long-range dependencies: O(1) path length between any positions  
✓ Interpretable: Attention weights show what model focuses on
✓ Flexible: Easy to adapt for different modalities
✓ Transfer learning: Pre-trained models work across tasks
✓ Scalable: Performance improves with model size and data
```

### Limitations

```
✗ Quadratic complexity: O(n²) memory and computation for sequence length n
✗ No inductive bias: Needs more data than CNNs/RNNs
✗ Positional encoding: Limited extrapolation to longer sequences
✗ Attention collapse: All tokens might attend to same positions
✗ Computational cost: Large models require significant resources
```

### When to Use Transformers

**Good for:**
- Long sequences where parallelization matters
- Tasks requiring long-range dependencies  
- Multimodal problems
- Transfer learning scenarios
- Large-scale pre-training

**Consider alternatives for:**
- Very long sequences (>10k tokens) due to quadratic cost
- Small datasets without pre-training
- Real-time applications with strict latency requirements
- Tasks where inductive bias helps (CNNs for images, RNNs for certain sequential patterns)

## Key Takeaways

- **Transformers solve fundamental RNN limitations** through parallelization and constant path length
- **Self-attention enables direct connections** between all sequence positions  
- **Multi-head attention captures multiple relationship types** simultaneously
- **Positional encoding provides order information** to permutation-invariant attention
- **Cross-modal attention enables multimodal understanding** by connecting different modalities
- **Gradient flow is improved** compared to sequential architectures
- **Computational cost scales quadratically** with sequence length

## Further Reading

**Original Papers:**
- Vaswani, A., et al. (2017). Attention Is All You Need. *NIPS 2017*.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *arXiv:1810.04805*.

**Multimodal Applications:**
- Lu, J., et al. (2019). ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations. *NeurIPS 2019*.
- Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training. *ICML 2022*.

**Mathematical Analysis:**
- Rogers, A., et al. (2020). A Primer on Neural Network Models for Natural Language Processing. *Journal of AI Research*.

---

**Previous**: [Chapter 7: Contrastive Learning](chapter-07.md) | **Next**: [Chapter 9: Generative Models for Multimodal Data](chapter-09.md) | **Home**: [Table of Contents](index.md)
