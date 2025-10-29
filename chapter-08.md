# Chapter 8: Transformer Architecture

---

**Previous**: [Chapter 7: Contrastive Learning](chapter-07.md) | **Next**: [Chapter 9: Generative Models for Multimodal Data](chapter-09.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand transformer fundamentals
- Explain self-attention mechanism
- Implement multi-head attention
- Understand encoder-decoder architecture
- Apply transformers to multimodal tasks

## 8.1 The Problem Transformers Solve

### Limitations of Sequential Models (RNNs)

**RNN limitations:**

```
Processing sequence: w1, w2, w3, w4, w5

RNN forward pass (sequential):
  h0 = initialization
  h1 = RNN(w1, h0)  ← Must wait for h0
  h2 = RNN(w2, h1)  ← Must wait for h1
  h3 = RNN(w3, h2)  ← Must wait for h2
  h4 = RNN(w4, h3)  ← Must wait for h3
  h5 = RNN(w5, h4)  ← Must wait for h4

Problems:
① Cannot parallelize
   Each step depends on previous
   Sequential bottleneck

② Gradient flow issues
   Backprop through 5 steps:
   gradient = ∂h5/∂h4 × ∂h4/∂h3 × ∂h3/∂h2 × ∂h2/∂h1 × ∂h1/∂h0

   Each factor typically < 1:
   0.9^5 = 0.59  (50% loss)
   0.9^100 ≈ 0   (vanishing gradient)

③ Limited context window
   Position t can see positions [0, t-1]
   Cannot look ahead (in some RNNs)
   Information degrades over long sequences
```

### CNN Limitations for Sequences

**CNN characteristics:**

```
Local receptive field:
  3×3 kernel sees 9 neighbors
  To see position distance 10:
  Need log(10) ≈ 4 layers

  For long sequences:
  Need many layers
  Deep networks = hard to train
```

### Transformer Solution

**Key insight:**

```
Why wait for sequential dependencies?

What if every position could see every other position simultaneously?

Query: Position i
Key/Value: All positions (including i)

Attention: Position i attends to all positions
Result: Global context immediately available!

Benefit:
① Fully parallelizable
   All positions process simultaneously
   Each GPU core handles one position

② No sequential bottleneck

③ Long-range dependencies captured immediately
   Position 0 can "see" position 100 in layer 1
   No need for deep networks
```

## 8.2 Self-Attention Mechanism

### Intuition

**Example - Machine translation:**

```
English: "The animal didn't cross the street because it was too tired"

Ambiguity: What does "it" refer to?
  Option A: "animal" (correct)
  Option B: "street" (incorrect)

How humans understand:
  Focus on "it" (pronoun)
  Look back at possible referents: "animal", "street"
  "Animal" makes more sense in context
  → "it" = "animal"

Self-attention for "it":
  Query: "it"
  Key/Value options: ["The", "animal", "didn't", ..., "tired"]
  Attention: Which words help interpret "it"?
    "animal": High attention (antecedent)
    "cross": Medium attention (related event)
    "The": Low attention (not informative)
  Result: "it" representation influenced mainly by "animal"
```

### Mathematical Definition

**Components:**

```
Query (Q): What am I asking about?
Key (K): What information is available?
Value (V): What to retrieve?

Analogy - Database:
  Query: Search terms ("animal")
  Keys: Database field names and values
  Values: Data to retrieve

Example:
  Query: "hungry"
  Key matches: "starving" (high similarity), "tired" (medium)
  Values: Corresponding word embeddings
  Result: Weighted sum of values based on key similarity to query
```

**Formula:**

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

Breakdown:

Q @ K^T:
  Query dot Key
  Shape: (seq_len, seq_len)
  Result: similarity matrix
  Element [i,j] = how much query_i matches key_j

  / √d_k:
  Normalization by embedding dimension
  Prevents gradient explosion

softmax(...):
  Convert similarities to probabilities [0,1]
  Sum to 1 per row
  Interpretation: How much to "pay attention" to each position

@ V:
  Weight values by attention weights
  Result: Weighted combination of value vectors
  Each query gets context-specific value
```

### Numerical Example

**Setup:**

```
Sequence: ["The", "cat", "sat"]
Embedding dimension: d_k = 4

Query vectors:
  Q1 = [0.1, 0.2, 0.3, 0.1]  for "The"
  Q2 = [0.4, 0.1, 0.2, 0.3]  for "cat"
  Q3 = [0.2, 0.3, 0.1, 0.4]  for "sat"

Key vectors (same as query in self-attention):
  K1 = [0.1, 0.2, 0.3, 0.1]  for "The"
  K2 = [0.4, 0.1, 0.2, 0.3]  for "cat"
  K3 = [0.2, 0.3, 0.1, 0.4]  for "sat"

Value vectors:
  V1 = [1, 0, 0, 0]  for "The"
  V2 = [0, 1, 0, 0]  for "cat"
  V3 = [0, 0, 1, 0]  for "sat"
```

**Computation for first query (position 0: "The"):**

```
Step 1: Q1 @ K^T (similarity scores)
  Q1·K1 = 0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.1*0.1 = 0.15
  Q1·K2 = 0.1*0.4 + 0.2*0.1 + 0.3*0.2 + 0.1*0.3 = 0.15
  Q1·K3 = 0.1*0.2 + 0.2*0.3 + 0.3*0.1 + 0.1*0.4 = 0.15

  Scores: [0.15, 0.15, 0.15]  (all equal - new in training)

Step 2: Divide by √d_k = √4 = 2
  [0.075, 0.075, 0.075]

Step 3: Softmax
  exp(0.075) ≈ 1.078
  exp(0.075) ≈ 1.078
  exp(0.075) ≈ 1.078

  Sum: 3.234

  Softmax: [1.078/3.234, 1.078/3.234, 1.078/3.234]
         = [0.333, 0.333, 0.333]
         (uniform distribution)

Step 4: Weight values
  0.333 * V1 + 0.333 * V2 + 0.333 * V3
  = 0.333 * [1,0,0,0] + 0.333 * [0,1,0,0] + 0.333 * [0,0,1,0]
  = [0.333, 0.333, 0.333, 0]
```

**After training:**

```
With learned embeddings, differences emerge:

Step 1: Q1 @ K^T
  Q1·K1 = 0.8   (high - "The" attends to itself)
  Q1·K2 = 0.2   (low - "The" doesn't attend to "cat")
  Q1·K3 = 0.3   (low - "The" doesn't attend to "sat")

Step 2: After scaling and softmax
  [0.7, 0.15, 0.15]

Step 3: Weighted values
  0.7 * V1 + 0.15 * V2 + 0.15 * V3
  = [0.7, 0.15, 0.15, 0]

  Interpretation:
  "The" mostly looks at itself
  Some information from neighboring words
  Reasonable: "The" is article, not much context needed
```

### Multi-Head Attention

**Why multiple heads?**

```
Single attention head learns one type of relationship
Different heads can learn different patterns

Head 1: Syntactic (grammar)
  "verb" attends to "object"
  "noun" attends to "adjective"

Head 2: Semantic (meaning)
  "pronoun" attends to "antecedent"
  "reference" attends to "entity"

Head 3: Long-range
  "end of sentence" attends to "beginning"
  Captures discourse structure

Head 4: Word type
  Different parts of speech have different patterns

Multiple heads = multiple representation subspaces
More expressive than single head
```

**Architecture:**

```
Input: x (seq_len, d_model)

For each head h = 1 to num_heads:
  ① Project to query space
     Q_h = x @ W_q^(h)    (seq_len, d_k)

  ② Project to key space
     K_h = x @ W_k^(h)    (seq_len, d_k)

  ③ Project to value space
     V_h = x @ W_v^(h)    (seq_len, d_v)

  ④ Compute attention
     head_h = Attention(Q_h, K_h, V_h)  (seq_len, d_v)

Concatenate all heads:
  MultiHead = [head_1 || head_2 || ... || head_h]
              (seq_len, h*d_v)

Linear projection:
  output = MultiHead @ W_o
           (seq_len, d_model)
```

**Example - 8 heads with d_model=512:**

```
Each head operates in d_k = 512/8 = 64 dimensional space
8 different projection matrices per Q, K, V

Result:
  8 independent attention mechanisms
  Each learns different patterns
  Combined through concatenation and final projection

Total parameters for multi-head attention:
  Q projections: 8 × 512 × 64 = 262K
  K projections: 8 × 512 × 64 = 262K
  V projections: 8 × 512 × 64 = 262K
  Output projection: 512 × 512 = 262K
  Total: ~1M parameters per multi-head attention layer
```

### Scaled Dot-Product Attention Revisited

**Why scale by 1/√d_k?**

```
Reason: Prevents gradient vanishing

Without scaling:
  For large d_k:
  Q @ K^T values become very large

  Example: Q and K each 64D
  Dot product: 64 independent terms
  Average value: 64 * (avg term)

  Large values → softmax saturates → gradients → 0

Scaling by 1/√d_k:
  Normalizes dot product variance
  Keep values in reasonable range [-1, 1] roughly
  Softmax doesn't saturate
  Gradients flow properly

Mathematical justification:
  Var(Q @ K^T) = Var(Σ q_i * k_i)
                = Σ Var(q_i * k_i)
                = d_k  (if independent)

  Std dev = √d_k

  Scaling by 1/√d_k makes std dev = 1
  Keeps gradients stable
```

## 8.3 Transformer Encoder

### Architecture Overview

```
Input sequence
    ↓
Embedding + Positional Encoding
    ↓
┌─────────────────────────────┐
│  Transformer Encoder Layer  │ ×N (typically 12)
│  ┌────────────────────────┐ │
│  │ Multi-Head Attention   │ │
│  └────────┬───────────────┘ │
│           ↓                  │
│  ┌─────────────────────────┐ │
│  │ Add & Normalize         │ │
│  └────────┬────────────────┘ │
│           ↓                  │
│  ┌─────────────────────────┐ │
│  │ Feed-Forward Network    │ │
│  │ (2 linear layers, ReLU) │ │
│  └────────┬────────────────┘ │
│           ↓                  │
│  ┌─────────────────────────┐ │
│  │ Add & Normalize         │ │
│  └────────┬────────────────┘ │
└─────────────────────────────┘
    ↓
Output (same shape as input)
```

### Detailed Layer Breakdown

**1. Positional Encoding**

```
Problem: Self-attention is permutation invariant
  Meaning: Word order doesn't matter!

  Attention doesn't care about position
  Just about content similarity

  Example:
    "dog bites man" vs "man bites dog"
    Same words, different meaning
    But attention treats them the same!

Solution: Add position information

Sinusoidal encoding:
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
  ④ Can extrapolate to longer sequences than training
```

**2. Multi-Head Self-Attention**

```
All positions attend to all positions
8-12 heads typically
Each head learns different patterns

Output same shape as input
```

**3. Add & Normalize (Residual Connection + Layer Normalization)**

```
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

  Why LN instead of Batch Norm?
  ① Batch norm depends on batch statistics
     Different at train/test time

  ② Layer norm is deterministic
     Doesn't depend on batch
     Same at train/test

  ③ Works better for sequences
```

**4. Feed-Forward Network**

```
MLP with 2 layers and ReLU:

FFN(x) = Linear2(ReLU(Linear1(x)))

Dimensions:
  Linear1: d_model → d_ff (usually 4*d_model)
  ReLU: d_ff → d_ff
  Linear2: d_ff → d_model

Example with d_model = 512:
  Linear1: 512 → 2048
  ReLU: 2048 → 2048
  Linear2: 2048 → 512

Why expand then contract?
  ① Increases non-linearity
  ② More expressive intermediate representation
  ③ Standard in deep learning
```

### Full Transformer Encoder

**BERT (Bidirectional Encoder Representations from Transformers):**

```
Architecture:
  ① Input: Tokens or subword units

  ② Embeddings:
     - Token embedding (word id → vector)
     - Positional embedding (position → vector)
     - Segment embedding (which sentence)
     - Sum all three

  ③ 12 layers (BERT-base) or 24 (BERT-large) of:
     - Multi-head attention (8-12 heads)
     - Feed-forward network
     - Residual connections + Layer norm

  ④ Output: Contextual embedding for each token

Example input: "The cat sat on the mat"

Processing:
  [CLS] The cat sat on the mat [SEP]
    ↓
  Embed each token
    ↓
  Add positional info
    ↓
  Layer 1:
    All tokens attend to all tokens
    Self-attention with 12 different heads

    "The" attends to: "The" (self), "cat", "sat", "on", "the", "mat"
    Different heads pay attention to different words
    Results concatenated

    Feed-forward applied to each position
    Residual + Layer norm

  Layer 2-12:
    Same process, but input is output from previous layer
    Further refinement

Output:
  12 vectors for each token
  Plus one special [CLS] token representing full sequence
```

## 8.4 Transformer Decoder

### Causal Masking

**Problem:**

```
During inference, generate one token at a time:
  Step 1: Predict token 1 (no prior tokens)
  Step 2: Predict token 2 (given token 1)
  Step 3: Predict token 3 (given tokens 1, 2)

During training (teacher forcing):
  Complete sequence available: token 1, 2, 3, 4, 5

To prepare model for inference:
  Hide future tokens during training

Mechanism: Causal mask
```

**Causal mask visualization:**

```
Attention positions (what can attend to what):

Sequence: [token_1, token_2, token_3, token_4, token_5]

Position 1 (token_1):
  Can attend to: position 1
  Cannot attend to: positions 2, 3, 4, 5

Position 2 (token_2):
  Can attend to: positions 1, 2
  Cannot attend to: positions 3, 4, 5

Position 3 (token_3):
  Can attend to: positions 1, 2, 3
  Cannot attend to: positions 4, 5

Attention matrix (✓ = can attend, ✗ = masked):

       pos1  pos2  pos3  pos4  pos5
pos1    ✓     ✗     ✗     ✗     ✗
pos2    ✓     ✓     ✗     ✗     ✗
pos3    ✓     ✓     ✓     ✗     ✗
pos4    ✓     ✓     ✓     ✓     ✗
pos5    ✓     ✓     ✓     ✓     ✓

Mask implementation:
  Before softmax, set masked positions to -∞
  softmax(-∞) = 0
  Effect: Attention weight = 0 for masked positions
```

### Autoregressive Generation

**Process:**

```
① Start: Input special token [START]
         Decoder produces distribution over vocabulary

② Step 1: Sample/select token with highest probability
          Let's say we get "A"

③ Step 2: Input "[START] A"
          Decoder predicts next token
          Get "red"

④ Step 3: Input "[START] A red"
          Decoder predicts next token
          Get "cat"

⑤ Continue until [END] token or max length

Generated: "A red cat"
```

**Key points:**

```
① Causal masking ensures only past tokens visible
② Gradual refinement of representation
③ Can use greedy (highest probability) or sampling
④ Sampling: More diverse but less controlled
   Greedy: More consistent but can repeat
```

### Cross-Attention in Decoder

**Integration with encoder:**

```
Encoder processes source:
  e.g., Image encoded to 196 patch embeddings

Decoder:
  ① Self-attention: Decoder attends to previously generated tokens
  ② Cross-attention: Decoder attends to encoder output
  ③ Feed-forward

Cross-attention details:
  Query: Current decoder hidden state
  Key/Value: Encoder output

  Result: Decoder can look at source modality
          Ground generation in input
```

**Example - Image captioning:**

```
Image: [Cat photo]
       ↓
Image encoder: 196 patch embeddings

Decoder generating caption:

Step 1:
  Decoder input: [START]
  Self-attention: Only [START], attends to itself
  Cross-attention: [START] attends to all 196 patches
                   "What's in image?"
  Output: Probability distribution for first word

Step 2:
  Decoder input: [START] A
  Self-attention: "A" attends to [START] and itself
                  "What context?"
  Cross-attention: "A" attends to patches
                   "What modifies 'A'?"
  Output: "cat"

Step 3:
  Decoder input: [START] A cat
  Self-attention: "cat" attends to [START], "A", "cat"
  Cross-attention: "cat" attends to patches
                   "What object is this?"
  Output: "sitting"

...

Caption: "A cat sitting on a couch"
```

## 8.5 Putting it Together: Vision Transformer (ViT)

### Architecture for Images

```
Image (224×224×3)
    ↓
Divide into patches (16×16)
    ↓
14 × 14 = 196 patches
    ↓
Each patch: 16×16×3 = 768D
    ↓
Linear projection: 768D → 768D embedding
    ↓
Add [CLS] special token
    ↓
Positional encoding (196 + 1 positions)
    ↓
Concatenate: [[CLS]; patch_1; patch_2; ...; patch_196]
             (197 tokens of 768D each)
    ↓
Transformer encoder (12 layers):
  Multi-head attention (12 heads)
  Feed-forward (3072D intermediate)
  Residual connections + Layer norm
    ↓
Extract [CLS] token representation
    ↓
Linear classifier: 768D → num_classes
    ↓
Output: Class probabilities
```

### Why ViT Works

```
Key insight 1: Patches as tokens
  Images have spatial structure
  Patches preserve local information
  Transformer learns global relationships

Key insight 2: Transformer is universal
  Can process any sequence of tokens
  Doesn't care if tokens are image or text
  Same architecture works for both!

Key insight 3: Attention gives global context
  CNNs need many layers for global receptive field
  ViT has global attention from layer 1
  Enables fast learning

Empirical finding:
  With small data: CNN >> ViT
  With large data: ViT >> CNN

  Trade-off: Inductive bias vs expressive power
  CNN: Strong inductive bias (local structure)
       Works with limited data
  ViT: Weak inductive bias
       Needs large data to learn structure
```

## Key Takeaways

- **Transformers** solve sequential bottleneck through attention
- **Self-attention** computes context through similarity
- **Multi-head attention** learns diverse patterns
- **Positional encoding** preserves sequence order
- **Residual connections** enable deep networks
- **Feed-forward networks** add non-linearity
- **Causal masking** enables autoregressive generation
- **Cross-attention** connects source and target
- **Vision Transformer** shows transformers work for images too

## Exercises

**⭐ Beginner:**
1. Compute self-attention by hand
2. Understand causal masking
3. Visualize positional encoding patterns

**⭐⭐ Intermediate:**
4. Implement multi-head attention from scratch
5. Build simple transformer encoder
6. Visualize attention patterns

**⭐⭐⭐ Advanced:**
7. Implement full transformer encoder-decoder
8. Build Vision Transformer
9. Implement efficient attention variants

---

