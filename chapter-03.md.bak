# Chapter 3: Feature Representation for Each Modality

---

**Previous**: [Chapter 2: Foundations and Core Concepts](chapter-02.md) | **Next**: [Chapter 4: Feature Alignment and Bridging Modalities](chapter-04.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand text representation methods from BoW to BERT
- Explain CNNs and Vision Transformers for images
- Describe MFCC and self-supervised learning for audio
- Compare different modality representations
- Choose appropriate representations for specific tasks

## 3.1 Text Representation: Evolution and Methods

### Historical Evolution

```
Timeline of text representation:

1950s-1990s:    Manual feature engineering
  ↓
1990s-2000s:    Bag-of-Words, TF-IDF
  ↓
2000s-2010s:    Word embeddings (Word2Vec, GloVe)
  ↓
2013-2018:      RNN, LSTM, GRU with embeddings
  ↓
2017+:          Transformer-based (BERT, GPT)
  ↓
2022+:          Large language models (GPT-3, LLaMA)
  ↓
2024+:          Multimodal LLMs
```

### Method 1: Bag-of-Words (BoW)

**Concept:**
Treat text as unordered collection of words, ignoring sequence and grammar.

**Process:**

```
Input:     "The cat sat on the mat"
             ↓
Tokenize:  ["the", "cat", "sat", "on", "the", "mat"]
             ↓
Count:     {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
             ↓
Vectorize: [2, 1, 1, 1, 1]  (in vocabulary order)
```

**Formal definition:**

```
For vocabulary V = {w_1, w_2, ..., w_N}
Text represented as: x = [c_1, c_2, ..., c_N]
where c_i = count of word w_i in text

Dimension = vocabulary size (can be 10,000-50,000)
```

**Example - Classification:**

```
Training data:
  Text 1: "I love this movie" → Label: Positive
  Text 2: "This movie is bad" → Label: Negative

BoW vectors:
  Text 1: {love: 1, movie: 1, positive words}
  Text 2: {bad: 1, movie: 1, negative words}

Classifier learns:
  "love" → +positive contribution
  "bad" → +negative contribution
```

**Advantages:**
✓ Simple and fast
✓ Interpretable
✓ Works surprisingly well for many tasks

**Disadvantages:**
✗ Loses word order ("dog bit man" = "man bit dog")
✗ No semantic relationships ("happy" vs "joyful" treated as completely different)
✗ All words equally important (doesn't distinguish important from common words)
✗ Very high dimensionality

**When to use:**
- Spam detection
- Topic modeling
- Simple text classification
- When simplicity and speed are priorities

### Method 2: TF-IDF (Term Frequency-Inverse Document Frequency)

**Motivation:**
BoW treats all words equally. But some words are more informative than others.

**Concept:**
```
Importance = (word frequency in document) × (rarity across corpus)

Words appearing everywhere ("the", "is") get low weight
Words appearing rarely but specifically ("CEO", "algorithm") get high weight
```

**Formal definition:**

```
TF (Term Frequency):
  TF(t,d) = count(t in d) / total_words(d)
  Normalized frequency of term t in document d

IDF (Inverse Document Frequency):
  IDF(t) = log(total_documents / documents_containing_t)
  How rare is this term across all documents?

TF-IDF:
  TF-IDF(t,d) = TF(t,d) × IDF(t)
```

**Example calculation:**

```
Corpus: 1,000 documents
Term "cat": appears in 100 documents, 5 times in document D

TF = 5 / total_words_in_D = 0.05
IDF = log(1000/100) = log(10) = 1.0
TF-IDF = 0.05 × 1.0 = 0.05

Compare to:
Term "the": appears in 900 documents, 50 times in document D

TF = 50 / total_words_in_D = 0.50
IDF = log(1000/900) = log(1.11) ≈ 0.1
TF-IDF = 0.50 × 0.1 = 0.05

Wait, same score! That's the point - importance normalized.
```

**Benefits over BoW:**
✓ Handles different document lengths better
✓ Downweights common words
✓ Emphasizes distinctive terms

**Disadvantages:**
✗ Still ignores word order
✗ No semantic understanding
✗ Requires corpus statistics
✗ Doesn't handle synonyms

**When to use:**
- Information retrieval and search
- TF-IDF is foundation of many search engines
- Document classification
- When you have many documents and limited compute

### Method 3: Word2Vec - Learning Word Meaning

**Revolutionary idea (Mikolov et al., 2013):**
"Words with similar contexts have similar meanings"

**Learning through prediction:**

```
Idea: If we can predict context words from a word,
      we've learned what that word means.

Process:

Text: "The dog barked loudly at the mailman"
              ↓
Focus on "barked", predict context:
  Context: {dog, loudly, at, the}
  Prediction task: Given "barked", predict these

Loss: How well did we predict?
  If good prediction → "barked" representation is good
  If poor → Update "barked" vector

After training on millions of sentences:
  "barked" vector captures:
  - Associated with actions
  - Related to animals
  - Past tense
  - Physical events
```

**Key discovery:**

```
Vector arithmetic works!

king - man + woman ≈ queen

Explanation:
- "king" and "queen" appear in similar contexts (monarchy)
- "man" and "woman" capture gender dimension
- Vector subtraction removes gender from "king"
- Vector addition applies gender to result
- Result: "queen"

This algebraic structure wasn't hand-designed!
It emerged from learning word contexts.
```

**Technical details - Two approaches:**

**Skip-gram:**
```
Input: Target word "barked"
Task: Predict context words {dog, loudly, at, the}

Model: Two embedding matrices
  Input embedding: What is "barked"?
  Output embedding: What patterns lead to context?

Optimization:
  Maximize: P(context | barked)
  Network learns useful representations
```

**CBOW (Continuous Bag of Words):**
```
Input: Context words {the, dog, barked, loudly}
Task: Predict center word

Reverse of skip-gram
Can be faster to train
```

**Properties:**
- Fixed embedding per word (doesn't handle polysemy)
- 300D vectors typical
- Can be trained on unlabeled data
- Transferable to downstream tasks

**Example - Semantic relationships:**

```
cos_sim(king, queen) ≈ 0.7   (high, related)
cos_sim(king, man) ≈ 0.65     (high, overlapping)
cos_sim(queen, woman) ≈ 0.68  (high, overlapping)
cos_sim(king, dog) ≈ 0.2      (low, unrelated)

Structure emerges in embedding space!
```

**Limitations:**
✗ One vector per word (ignores context and polysemy)
✗ "Bank" (financial) and "bank" (river) have identical vectors
✗ Same word might mean different things in different contexts
✗ Doesn't capture longer-range dependencies

**When to use:**
- Quick baseline for text tasks
- When you need interpretable word relationships
- Transfer learning where only word similarity needed
- When computational resources are limited

### Method 4: BERT - Context-Aware Embeddings

**Motivation:**

Word2Vec limitation - context blindness:

```
Sentence 1: "I went to the bank to deposit money"
Sentence 2: "I sat on the bank of the river"

Word2Vec:
  "bank" in both sentences → IDENTICAL vector
  Problem: Different meanings!

What we need:
  Context-aware "bank" for finance sentence
  Different context-aware "bank" for river sentence
```

**BERT Innovation (Devlin et al., 2018):**
"Use entire sentence context to generate embeddings"

**Architecture overview:**

```
Input text: "The cat sat on the mat"
             ↓
Tokenization (using WordPiece):
  [CLS] The cat sat on the mat [SEP]
             ↓
Embedding:
  - Token embedding (which word)
  - Position embedding (where in sequence)
  - Segment embedding (which sentence)
             ↓
Transformer encoder (12 layers):
  Each layer:
    - Self-attention (how relevant is each token to others)
    - Feed-forward network
    - Normalization
             ↓
Output: 12 vectors of 768D each
  Each token has representation influenced by entire sequence
```

**Key innovation - Bidirectional context:**

```
Traditional RNN: Left-to-right only
  Input: "The cat sat..."
         Process: The → cat → sat
         When processing "sat", don't know what comes after

BERT: Bidirectional
  Input: "The cat sat on the mat"
         Process: Entire sequence simultaneously
         All positions see all other positions
         Through self-attention in first layer
```

**Training procedure - Masked Language Modeling:**

```
Goal: Learn good representations for any language task

Method: Predict masked words

Original:      "The [MASK] sat on the mat"
Task:          Predict the masked word
Expected:      "cat"

Training:
  ① Randomly mask 15% of tokens
  ② Model predicts masked tokens
  ③ Loss = cross-entropy between predicted and actual
  ④ Update all parameters

Result:
  Model learns representations that contain
  information about what words should appear
  = learns semantic and syntactic patterns
```

**Using BERT embeddings:**

```
For sentence classification:
  ① Process sentence through BERT
  ② Extract [CLS] token (special classification token)
  ③ [CLS] vector = sentence representation (768D)
  ④ Add linear classifier on top
  ⑤ Train classifier on downstream task

For token classification (e.g., NER):
  ① Process sentence through BERT
  ② Extract all token vectors (each is 768D)
  ③ Each token has context-aware representation
  ④ Add classifier for each token
  ⑤ Predict label for each token

Benefit:
  - No task-specific feature engineering needed
  - Transfer learning from massive pre-training
  - Strong performance on small datasets
```

**Concrete example - Polysemy handling:**

```
Sentence 1: "I went to the bank to deposit money"
  "bank" → BERT embedding with finance context

Sentence 2: "I sat on the bank of the river"
  "bank" → BERT embedding with geography context

Different embeddings!
BERT captures context from surrounding words
```

**Properties:**
- Context-dependent embeddings
- 768D vectors (BERT-base)
- Larger versions available (BERT-large: 1024D)
- Pre-trained on 3.3B words
- Extremely effective for transfer learning

**Advantages over Word2Vec:**
✓ Handles polysemy (same word, different contexts)
✓ Bidirectional context
✓ Pre-trained on massive corpus
✓ Strong transfer learning
✓ Achieves SOTA on many tasks

**Disadvantages:**
✗ Computationally expensive
✗ Slower inference than Word2Vec
✗ Requires more compute resources
✗ Less interpretable (768D vectors hard to understand)

**When to use:**
- Text classification (sentiment, topic)
- Named entity recognition
- Question answering
- Semantic similarity
- When accuracy more important than speed
- When GPU resources available

### Method 5: Large Language Models (LLMs)

**Further evolution - GPT family:**

```
BERT (2018):        Encoder-only, bidirectional
GPT (2018):         Decoder-only, left-to-right
GPT-2 (2019):       1.5B parameters
GPT-3 (2020):       175B parameters - in-context learning
GPT-4 (2023):       ~1.76T parameters - multimodal
```

**LLM representations:**

```
GPT-3 embeddings:
  Layer 1:    Basic patterns
  Layer 16:   Mid-level concepts
  Layer 32:   High-level semantics
  Layer 48 (final): Task-specific representations

Properties:
  - 12,288D vectors (very high-dimensional)
  - Captures vast knowledge
  - Can be used as semantic features
  - More interpretable than BERT in some ways
```

**Using LLM embeddings for multimodal tasks:**

```
Instead of using fixed word embeddings,
use representations from large language models

Benefit:
  - Captures world knowledge from pre-training
  - Understands complex semantics
  - Better for rare/unusual concepts
  - Can be adapted to specific domains

Cost:
  - Expensive API calls (if using services like OpenAI)
  - Privacy concerns (data sent to external servers)
  - Latency (requires API round-trip)
```

**Comparison of text representations:**

```
Method          Dimension   Context-aware   Speed   Pre-training
────────────────────────────────────────────────────────────────
BoW             10K-50K     No              Fast    None needed
TF-IDF          10K-50K     No              Fast    Corpus stats
Word2Vec        300         No              Fast    Large corpus
GloVe           300         No              Fast    Large corpus
FastText        300         No              Fast    Large corpus
ELMo            1024        Yes             Slow    Large corpus
BERT            768         Yes             Medium  Huge corpus
RoBERTa         768         Yes             Medium  Huge corpus
GPT-2           1600        Yes             Slow    Huge corpus
GPT-3           12288       Yes             Very slow API
```

## 3.2 Image Representation: From Pixels to Concepts

### Historical Evolution

```
Timeline:

1980s-1990s:    Edge detection (Canny, Sobel)
  ↓
1990s-2000s:    Hand-crafted features (SIFT, HOG)
  ↓
2012:           AlexNet - Deep learning breakthrough
  ↓
2014:           VGGNet, GoogleNet
  ↓
2015:           ResNet - Skip connections, very deep networks
  ↓
2020:           Vision Transformer - Attention-based vision
  ↓
2024:           Large multimodal models processing images
```

### Method 1: Hand-Crafted Features

**SIFT (Scale-Invariant Feature Transform)**

```
Problem solved:
  "Find the same building in photos taken at different times,
   different angles, different zoom levels"

SIFT features are invariant to:
  - Translation (where object is in image)
  - Scaling (zoom level)
  - Rotation (camera angle)
  - Illumination (lighting changes)

Process:
  1. Find keypoints (interest points)
     - Corners, edges, distinctive regions

  2. Describe neighborhoods around keypoints
     - Direction and magnitude of gradients
     - Histogram of edge orientations

  3. Result: Keypoint descriptor (128D vector)
     - Invariant to many transformations
     - Can match same keypoint across images

Example:
  Building in Photo 1 (summer, noon, straight angle)
  Same building in Photo 2 (winter, sunset, aerial view)

  SIFT can find matching keypoints!
  Enables: Panorama stitching, 3D reconstruction
```

**HOG (Histogram of Oriented Gradients)**

```
Key insight:
  Human shape recognition relies on edge directions
  (Horizontal edges on top = head, vertical on sides = body)

Process:
  1. Divide image into cells (8×8 pixels)

  2. For each cell:
     - Compute edge direction at each pixel
     - Create histogram of edge directions

  3. Result: Concatenate all histograms
     - Captures shape and edge structure
     - Dimension: ~3,780 for 64×128 image

Application:
  Pedestrian detection
  - HOG captures distinctive human silhouette
  - Works well because human shape is distinctive
  - Fast computation (no deep learning needed)

  Limitation:
  - Only works for rigid objects (humans, faces)
  - Fails for abstract categories
```

**Bag-of-Visual-Words**

```
Idea: Apply Bag-of-Words concept to images

Process:
  1. Extract SIFT features from image
     → Get 100-1000 keypoint descriptors per image

  2. Cluster descriptors (k-means)
     → Create "visual vocabulary" (e.g., 1000 clusters)
     → Each cluster = one "visual word"

  3. Histogram of visual words
     → Count which words appear in image
     → Result: Bag-of-words vector

  4. Classify or compare based on histogram

Example:
  Image 1 has: {30 "corner edges", 20 "smooth curves", ...}
  Image 2 has: {5 "corner edges", 45 "smooth curves", ...}

  More curve words → Perhaps a cat
  More corner words → Perhaps a building
```

**Advantages of hand-crafted features:**
✓ Interpretable (understand what they measure)
✓ Fast computation
✓ Works with small datasets
✓ Explicit mathematical basis

**Disadvantages:**
✗ Requires domain expertise to design
✗ Limited to specific feature types
✗ Poor generalization to new domains
✗ Cannot capture complex semantic patterns
✗ Manually chosen → not optimized for task

**When to use:**
- When you understand the specific patterns to detect
- Limited computational resources
- Small datasets
- Tasks where hand-crafted features are well-suited (e.g., pedestrian detection)

### Method 2: CNNs - Automatic Feature Learning

**The Breakthrough (AlexNet, 2012):**

```
Revolutionary insight:
  "Stop hand-crafting features!
   Let neural networks learn what's important."

Results:
  ImageNet competition:
  - 2011 (hand-crafted): 25.8% error
  - 2012 (AlexNet): 15.3% error  ← 38% error reduction!
  - 2015 (ResNet): 3.6% error   ← Human-level performance
```

**Hierarchical Feature Learning:**

```
Raw image (224×224×3 pixels)
        ↓
Layer 1-2: Low-level features
  - Edge detection
  - Simple curves
  - Corners
  └─→ What: Detects local patterns
      Why: Edges are building blocks
      Output: 64 feature maps (32×32)

Layer 3-4: Mid-level features
  - Textures
  - Shapes
  - Parts
  └─→ What: Combines local patterns
      Why: Shapes emerge from edges
      Output: 256 feature maps (16×16)

Layer 5: High-level features
  - Objects
  - Semantic concepts
  - Scene context
  └─→ What: Object detectors
      Why: Objects are concepts
      Output: 512 feature maps (8×8)

Global pooling & Dense layers:
  - Aggregate spatial info
  - Predict class probabilities
  └─→ Output: Class predictions
```

**Why CNNs work:**

```
1. Inductive bias toward images
   - Local connectivity: Nearby pixels related
   - Shared weights: Same pattern recognized anywhere
   - Translation invariance: "Cat is a cat" whether left/right

2. Hierarchical composition
   - Edges → Shapes → Objects
   - Matches how we see

3. Parameter sharing
   - Filters reused across space
   - Reduces parameters vs fully connected
   - Enables learning on larger images
```

**Key architecture - ResNet (Residual Networks):**

```
Problem with deep networks:
  Deeper = more parameters = better?
  But: Very deep networks are hard to train!

  Cause: Gradient vanishing
    Backprop through 100 layers:
    gradient = g₁ × g₂ × g₃ × ... × g₁₀₀

    If each gᵢ = 0.9:
    0.9¹⁰⁰ ≈ 0.0000027  (essentially zero!)

    Can't learn early layers

Solution: Skip connections (residual connections)

Normal layer: y = f(x)
Residual layer: y = x + f(x)

Benefit:
  Even if f(x) learns nothing (f(x)=0),
  y = x still flows information through

  Gradient paths:
  Without skip: gradient = ∂f/∂x × ∂f/∂x × ...
  With skip: gradient = ... + 1 + 1 + ...

  The "+1" terms prevent vanishing!
```

**ResNet architecture example (ResNet-50):**

```
Input: Image (224×224×3)
  ↓
Conv 7×7, stride 2
→ (112×112×64)
  ↓
MaxPool 3×3, stride 2
→ (56×56×64)
  ↓
Residual Block 1: [16 conv blocks]
→ (56×56×256)
  ↓
Residual Block 2: [33 conv blocks]
→ (28×28×512)
  ↓
Residual Block 3: [36 conv blocks]
→ (14×14×1024)
  ↓
Residual Block 4: [3 conv blocks]
→ (7×7×2048)
  ↓
Average Pool
→ (2048,)
  ↓
Linear layer (1000 classes)
→ Predictions

Total parameters: 25.5M
Depth: 50 layers
Performance: 76% ImageNet top-1 accuracy
```

**Properties:**
- 2048D global feature vector (before classification)
- Pre-trained on ImageNet (1.4M images)
- Can fine-tune on downstream tasks
- Very stable training (skip connections)

**Advantages:**
✓ Learns task-relevant features
✓ Transfers well to other tasks
✓ Stable training (deep networks possible)
✓ Interpretable to some extent (visualize activations)
✓ Efficient inference

**Disadvantages:**
✗ Black-box decisions (what does each dimension mean?)
✗ Requires large labeled datasets to train from scratch
✗ Inherits biases from ImageNet

**When to use:**
- Most modern computer vision tasks
- Transfer learning (fine-tune on new task)
- When you want strong off-the-shelf features
- Production systems (mature, optimized, proven)

### Method 3: Vision Transformers (ViT)

**Paradigm shift (Dosovitskiy et al., 2020):**

```
Traditional thinking:
  "Images need CNNs!"
  Reason: Spatial structure, translational equivariance

ViT question:
  "What if we just use Transformers like NLP?"
  Insight: Pure attention can learn spatial patterns

Result:
  Vision Transformer outperforms ResNet
  When trained on large datasets!
```

**Architecture:**

```
Input image (224×224×3)
        ↓
Divide into patches (16×16)
        ↓
14×14 = 196 patches
        ↓
Each patch: 16×16×3 = 768D
        ↓
Linear projection
        ↓
196 vectors of 768D
        ↓
Add positional encoding
(so model knows spatial position)
        ↓
Add [CLS] token
(like BERT for images)
        ↓
Transformer encoder (12 layers)
        ↓
Extract [CLS] token
        ↓
768D image representation
```

**How it works:**

```
Key insight: Patches are like words

In NLP:
  Word tokens → Transformer → Semantic relationships

In ViT:
  Image patches → Transformer → Spatial relationships

Layer 1:
  Each patch attends to all other patches
  Learns: Which patches are related?

Layer 2-12:
  Progressively integrate information
  Layer 6: Coarse spatial understanding
  Layer 12: Fine-grained semantic understanding
```

**Why this works:**

1. **Global receptive field from Layer 1**

   CNN needs many layers to see globally
   ViT sees all patches from first layer
   Enables faster learning of global patterns

2. **Flexible to patches**

   Can use any patch size
   Trade-off:
   - Larger patches (32×32): Fewer tokens, less detail
   - Smaller patches (8×8): More tokens, finer detail

3. **Scales with data**

   CNNs strong with small data (inductive biases)
   ViT weak with small data, strong with large

   Modern datasets massive
   → ViT wins

**Example - ViT-Base vs ResNet-50:**

```
                ViT-Base       ResNet-50
────────────────────────────────────
Parameters      86M            25.5M
ImageNet acc    77.9%          76%
Training data   1.4M+JFT      1.4M
Pre-training    224×224        1000×1000
Fine-tuning     Excellent      Good

Interpretation:
  ViT needs more data to train
  But then performs better
  Especially when transferring to new tasks
```

**Advantages:**
✓ Better scaling properties
✓ Transfers better to downstream tasks
✓ Simpler architecture (no CNN-specific tricks needed)
✓ More interpretable (attention patterns show what matters)
✓ Unified with NLP (same architecture for both)

**Disadvantages:**
✗ Worse with small datasets
✗ Requires more computation than CNN equivalents
✗ Training unstable (needs careful tuning)
✗ Slower inference in some hardware

**When to use:**
- Large-scale applications
- Transfer learning to new visual tasks
- When computational resources abundant
- When interpretability matters (attention visualization)
- New research (faster progress with transformers)

**Attention visualization:**

```
For each query patch, show which patches it attends to

Example - Query at cat's head position:

Attention heatmap:
[   0    0    0  ]
[   0   0.9   0.8]  (high attention to nearby patches)
[   0    0.6   0  ]

Shows:
- Model focuses on cat head region
- Attends to surrounding patches (context)
- Ignores background regions
```

## 3.3 Audio Representation: From Waveforms to Features

### Method 1: MFCC (Mel-Frequency Cepstral Coefficients)

**Principle:**
"Extract features that match human hearing, not physics"

**Why needed:**

```
Raw audio at 16kHz:
  1 second = 16,000 samples
  10 seconds = 160,000 samples

Problem:
  Too many numbers to process
  Not perceptually relevant (e.g., 16kHz vs 16.1kHz)

Solution:
  Extract ~39 MFCCs per frame (25ms)
  Much more compact and perceptually meaningful
```

**Extraction process step-by-step:**

```
① Raw waveform
   Sample audio: 16kHz, mono
   Duration: 10 seconds

② Pre-emphasis
   Amplify high frequencies
   Reason: High frequencies carry important information
   Filter: y[n] = x[n] - 0.95*x[n-1]

③ Frame division
   Split into overlapping frames
   Frame length: 25ms = 400 samples
   Hop size: 10ms
   Result: ~980 frames for 10-second audio

④ Window each frame
   Apply Hamming window: reduces edge artifacts

⑤ Fourier Transform (FFT)
   Convert time domain → frequency domain
   For each frame: 400 samples → 200 frequency bins

⑥ Mel-scale warping
   Map frequency to Mel scale (human perception)

   Linear frequency: 125Hz, 250Hz, 500Hz, 1000Hz, 2000Hz
   Mel frequency:     0Mel,   250Mel, 500Mel, 1000Mel, 1700Mel

   Why?
   Humans more sensitive to low frequencies
   High frequencies sound similar to each other
   (1000Hz difference matters less at 10,000Hz)

⑦ Logarithm
   Human loudness perception is logarithmic
   log(power) more perceptually uniform than power

⑧ Discrete Cosine Transform (DCT)
   Decorrelate the Mel-scale powers
   Result: Typically 13-39 coefficients

Result: MFCC vector
  Dimensions: 39 (or 13, 26 depending on config)
  One vector per 10ms
  Represents spectral shape at that time
```

**Visualization:**

```
Raw waveform:          Spectrogram:           MFCCs:
Amplitude              Frequency vs Time      Features vs Time
   ↑                      High ▲               ↑
   │ ~~~~               ▓▓▓▓▓│▓▓▓          ▓▓▓│▓▓▓
   │~  ~  ~  ~~       ▓▓▓  │▓▓▓          ▓▓ │▓▓
   │ ~ ~~  ~ ~       ▓▓   │▓           ▓  │▓
   └──────────→      ▓▓    │            ▓  │
   Time (s)         Low ▼  └─────────→ Coeff│
                         Time (s)         └─→
                                        Dim 1-39
```

**Example - Speech recognition:**

```
Audio: "Hello"
        ↓
MFCC extraction (39D per frame)
        ↓
10 frames of audio (each 10ms):
  Frame 1: [0.2, -0.1, 0.5, ..., 0.3] (39D)
  Frame 2: [0.21, -0.08, 0.52, ..., 0.31] (39D)
  ...
  Frame 10: [0.15, -0.12, 0.45, ..., 0.25] (39D)
        ↓
Sequence of MFCCs: 10×39 matrix
        ↓
Feed to speech recognition model
        ↓
Output: Text "Hello"
```

**Properties:**
- Fixed dimensionality (39D)
- Perceptually meaningful
- Low computational cost
- Standard for speech tasks

**Advantages:**
✓ Fast to compute
✓ Well-understood (40+ years research)
✓ Works well for speech (the main audio task)
✓ Low dimensionality
✓ Perceptually meaningful

**Disadvantages:**
✗ Not learnable (fixed formula)
✗ May discard useful information
✗ Optimized for speech, not music
✗ Doesn't handle music well

**When to use:**
- Speech recognition
- Speaker identification
- Emotion recognition from speech
- Music genre classification (acceptable)
- Limited compute resources

### Method 2: Spectrogram

**Alternative to MFCC:**
Keep all frequency information, don't apply Mel-scale or DCT.

**Process:**

```
① Raw audio
② Frame division
③ FFT
④ Magnitude spectrum
⑤ Spectrogram: stacked magnitude spectra over time

Result: 2D matrix
  Dimensions: Time × Frequency
  Values: Power at each time-frequency bin

Example: 10-second audio at 16kHz
  Time: 980 frames
  Frequency: 513 bins
  Size: 980×513
```

**Visualization:**

```
Spectrogram of "Hello":

Frequency
(Hz)    |▓▓ ▓▓▓▓    ▓▓    |
        |▓▓▓▓▓▓▓  ▓▓▓▓▓▓ | High freq
        |  ▓▓▓▓▓▓▓▓▓▓▓▓  |
  8000  |─────────────────|
        | ▓▓▓▓ ▓▓▓▓▓  ▓▓  |
        |▓▓▓▓ ▓▓▓▓▓▓▓▓▓   |
        |▓▓ ▓ ▓▓▓▓▓ ▓▓    | Low freq
    0   |___________________|
        0    2    4    6    8    10
              Time (seconds)

Darker = higher power
Different time positions → different audio
```

**Advantages over MFCC:**
✓ More information preserved
✓ Raw frequency content visible
✓ Can apply deep learning directly
✓ Works for any audio (not just speech)

**Disadvantages:**
✗ High dimensionality (harder to process)
✗ Not perceptually normalized
✗ Less standard for speech

**When to use:**
- Music processing and generation
- Sound event detection
- When using deep learning (CNN/Transformer)
- When frequency content important

### Method 3: Wav2Vec2 - Self-Supervised Learning

**Modern approach (Meta AI, 2020):**

```
Problem:
  Need thousands of hours transcribed audio for ASR
  Transcription is expensive

Solution:
  Learn from UNLABELED audio
  Use self-supervised learning
```

**Training mechanism:**

```
Phase 1: Pretraining (on unlabeled data)

  ① Feature extraction (CNN)
     Raw waveform → discrete codes

     Intuition: Compress speech to meaningful units

  ② Contrastive loss
     Predict masked codes from context
     Similar to BERT for speech

  Result: Model learns speech patterns
          Without any transcriptions!

Phase 2: Fine-tuning (with small labeled dataset)

  ① Load pretrained model
  ② Add task-specific head (classification)
  ③ Train on labeled examples

  Benefit: Needs much less labeled data!
```

**Quantization step:**

```
Why quantize speech?

Raw features: Continuous values
Problem: Too flexible, model can memorize

Quantized features: Discrete codes (e.g., 1-512)
Benefit:
  - Reduces search space
  - Forces learning of essential patterns
  - Similar to VQ-VAE for images

Example:
  Raw feature: [0.234, -0.512, 0.891, ...]
  ↓ (vector quantization)
  Nearest code ID: 147

  Code vector: Learned codebook entry 147
```

**Architecture:**

```
Raw waveform (16kHz)
        ↓
CNN feature extraction
        ↓
Quantization to codes
        ↓
Transformer encoder (contextual understanding)
        ↓
768D representation per frame
```

**Training details:**

```
Objective:
  Predict masked codes from surrounding codes

  Input: [code_1, [MASK], code_3, [MASK], code_5]
  Task: Predict masked codes

  Loss: Contrastive - predict correct code among negatives

Result:
  Encoder learns to represent speech meaningfully
  Ready for downstream tasks
```

**Fine-tuning for tasks:**

```
Task 1: Speech Recognition (ASR)
  Add: Linear layer for character/phoneme classification
  Train: On (audio, transcription) pairs

  Data needed: 10-100 hours labeled
  Without pretraining: 10,000+ hours needed!

Task 2: Speaker Identification
  Add: Linear layer for speaker classification
  Train: On (audio, speaker_id) pairs

Task 3: Emotion Recognition
  Add: Linear layer for emotion classification
  Train: On (audio, emotion) pairs
```

**Empirical results:**

```
Without Wav2Vec2 pretraining:
  ASR with 100 hours data: 25% WER (Word Error Rate)

With Wav2Vec2 pretraining:
  ASR with 100 hours data: 10% WER
  ASR with 10 hours data: 12% WER

Improvement:
  50% error reduction with same data
  Or 10× less labeled data for same performance
```

**Properties:**
- 768D representation per frame
- Learned from unlabeled data
- Transferable across tasks
- Works for any audio

**Advantages:**
✓ Leverages massive unlabeled data
✓ Strong transfer learning
✓ Handles diverse audio types
✓ Better than MFCC for complex tasks

**Disadvantages:**
✗ Complex training procedure
✗ Requires large unlabeled dataset for pretraining
✗ Longer inference than MFCC

**When to use:**
- Speech recognition (SOTA approach)
- Multi-speaker systems
- Low-resource languages
- When accuracy is critical

## 3.4 Comparison and Selection Guide

### Dimension and Computational Cost

```
                Dimension   Speed       Training Data
────────────────────────────────────────────────────
MFCC            39          Very fast   Hundreds hours
Spectrogram     513         Fast        Thousands hours
Wav2Vec2        768         Slow        Millions hours unlabeled

Hand-crafted    1000-5000   Fast        Medium
SIFT            128/keypoint Fast       Medium
HOG             3780        Fast        Medium

ResNet50        2048        Medium      1.4M images
ViT-Base        768         Medium      14M images
BERT            768         Medium      3.3B words
GPT-3           12288       Slow        Huge
```

### Modality Comparison Summary

```
                Text            Image           Audio
────────────────────────────────────────────────────
Modern rep.     BERT/GPT        ResNet/ViT      Wav2Vec2
Dimension       768             2048/768        768
Interpretable   Somewhat        Little          Very little
Speed           Medium          Fast            Medium
Pre-training    Easy (text web) Requires labels Can be unsupervised
Transfer        Excellent       Good            Good
Multimodal fit  Good            Excellent       Good
```

### Choosing Representation

**Decision flowchart:**

```
Is computational budget limited?
  YES → Use hand-crafted or MFCC
  NO → Continue
       ↓
Is this a production system?
  YES → Use proven methods (ResNet, BERT)
  NO → Continue
       ↓
Do you have massive labeled data?
  YES → Consider training from scratch
  NO → Use pre-trained features
       ↓
Do you have unlabeled data?
  YES → Consider self-supervised (Wav2Vec2)
  NO → Use supervised pre-trained models
```

## Key Takeaways

- **Text:** Evolution from BoW to BERT shows power of context
- **Images:** CNNs dominate but ViT shows promising future
- **Audio:** MFCC traditional, Wav2Vec2 is modern frontier
- **Pre-training is key:** Leveraging unlabeled data essential
- **Different modalities need different approaches**
- **Trade-offs exist:** accuracy vs speed, interpretability vs performance

## Exercises

**⭐ Beginner:**
1. Implement TF-IDF from scratch
2. Extract MFCC features from an audio file
3. Visualize a spectrogram

**⭐⭐ Intermediate:**
4. Compare MFCC vs spectrogram representations
5. Fine-tune BERT on text classification
6. Extract ResNet features and cluster images

**⭐⭐⭐ Advanced:**
7. Implement self-attention for images (simplified ViT)
8. Build Wav2Vec2 from scratch (simplified)
9. Compare different dimensionality reduction techniques

---

