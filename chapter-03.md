# Chapter 3: Feature Representation for Each Modality

---

**Previous**: [Chapter 2: Foundations and Core Concepts](chapter-02.md) | **Next**: [Chapter 4: Feature Alignment and Bridging Modalities](chapter-04.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Extract features from text using various methods
- Understand CNN architectures for image processing  
- Process audio signals for machine learning
- Choose appropriate feature extraction for different modalities
- Debug common issues in feature extraction pipelines

## 3.1 Text Representation

### Bag of Words (BoW) - The Foundation

**Basic concept:** Represent text as word frequency counts

**Example:**
```
Text: "The cat sat on the mat"
Vocabulary: [the, cat, sat, on, mat, dog, run, ...]

BoW representation:
[2, 1, 1, 1, 1, 0, 0, ...]
↑  ↑  ↑  ↑  ↑  ↑  ↑
"the" appears 2 times
"cat" appears 1 time  
"sat" appears 1 time
"on" appears 1 time
"mat" appears 1 time
"dog" appears 0 times
"run" appears 0 times
```

**Advantages:**
✓ Simple to understand and implement
✓ Fast computation
✓ Good baseline for many tasks
✓ Interpretable (can see which words matter)

**Disadvantages:**
✗ Loses word order ("dog bit man" = "man bit dog")
✗ No semantic relationships ("happy" vs "joyful" treated as completely different)
✗ All words equally important (doesn't distinguish important from common words)
✗ Very high dimensionality

**When to use:**
- Spam detection
- Topic modeling
- Simple text classification
- Document similarity (basic)

### TF-IDF (Term Frequency-Inverse Document Frequency)

**Improvement over BoW:** Weight words by importance

**Mathematical formulation:**
```
For term t in document d within corpus D:

TF(t,d) = count(t,d) / |d|
where:
  count(t,d) = number of times term t appears in document d
  |d| = total number of terms in document d

IDF(t,D) = log(|D| / |{d ∈ D : t ∈ d}|)
where:
  |D| = total number of documents in corpus
  |{d ∈ D : t ∈ d}| = number of documents containing term t

TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
```

**Intuition:**
```
TF (Term Frequency):
  Higher if word appears more in this document
  "This document is about cats. Cats are amazing. I love cats."
  → "cats" gets high TF score

IDF (Inverse Document Frequency):  
  Higher if word appears in fewer documents overall
  Common words like "the", "is", "a" appear everywhere → low IDF
  Specific words like "photosynthesis" appear rarely → high IDF

Combined TF-IDF:
  High score = word is frequent in this document AND rare overall
  → Indicates this word is important for characterizing this document
```

**Example calculation:**
```
Corpus: 1000 documents
Document: "The cat sat on the mat. The cat was happy."

For word "cat":
  TF = 2/8 = 0.25 (appears 2 times out of 8 total words)
  IDF = log(1000/50) = log(20) ≈ 3.0 (assuming "cat" appears in 50 documents)
  TF-IDF = 0.25 × 3.0 = 0.75

For word "the":
  TF = 3/8 = 0.375 (appears 3 times)  
  IDF = log(1000/900) = log(1.11) ≈ 0.1 (appears in most documents)
  TF-IDF = 0.375 × 0.1 = 0.0375

Result: "cat" gets much higher score than "the"
```

### Word Embeddings - Semantic Vectors

**Key insight:** Words with similar meanings should have similar representations

#### Word2Vec (2013)

**Core idea:** Learn embeddings from word co-occurrence patterns

**Two algorithms:**
1. **Skip-gram:** Given center word, predict context words
2. **CBOW (Continuous Bag of Words):** Given context words, predict center word

**Training example (Skip-gram):**
```
Sentence: "The cat sat on the mat"
Window size: 2

Training pairs:
Input → Output
"cat" → "The"     (context word 2 positions left)
"cat" → "sat"     (context word 1 position right)  
"sat" → "cat"     (context word 1 position left)
"sat" → "on"      (context word 1 position right)
...

Model learns: words appearing in similar contexts get similar embeddings
```

**Remarkable property - Semantic arithmetic:**
```
**king** - **man** + **woman** ≈ **queen**

Explanation:
- "king" and "queen" appear in similar contexts (monarchy)
- "man" and "woman" capture gender dimension
- Vector subtraction removes gender from "king"
- Vector addition applies gender to result
- Result: "queen"

This algebraic structure wasn't hand-designed!
It emerged from learning co-occurrence patterns.
```

**Typical dimensions:** 300D
**Training corpus:** Billions of words from Wikipedia, news, web text

#### BERT (2018) - Contextual Embeddings

**Key improvement:** Same word gets different embeddings in different contexts

**Problem Word2Vec couldn't solve:**
```
Sentence 1: "I went to the bank to deposit money"
Sentence 2: "I sat by the river bank to watch sunset"

Word2Vec: "bank" gets SAME embedding in both sentences
BERT: "bank" gets DIFFERENT embeddings based on context
```

**Architecture:** Transformer encoder with bidirectional attention

**Training:** Masked Language Modeling
```
Input: "The cat [MASK] on the mat"
Task: Predict "[MASK]" = "sat"

BERT learns to use context from BOTH sides:
- Left context: "The cat"  
- Right context: "on the mat"
- Combined context suggests "sat" is most likely
```

**Embedding extraction:**
```
Input: Tokenize text into [CLS] + words + [SEP]
Process: 12 transformer layers (BERT-base) or 24 layers (BERT-large)  
Output: Contextual embedding for each token

Common approaches:
1. Use [CLS] token embedding (768D) for sentence representation
2. Average word embeddings for sentence representation
3. Use specific word embeddings for word-level tasks
```

#### Modern Large Language Models (2020+)

**GPT series:**
```
GPT-1 (2018): 117M parameters, decoder-only
GPT-2 (2019): 1.5B parameters, "too dangerous to release"
GPT-3 (2020): 175B parameters, few-shot learning
GPT-4 (2023): Estimated 1T+ parameters, multimodal

Properties:
  - 12,288D vectors (very high-dimensional)
  - Captures vast knowledge
  - Can be used as semantic features
  - More interpretable than BERT in some ways
```

## 3.2 Image Representation

### Classical Approaches (Pre-Deep Learning)

#### SIFT (Scale-Invariant Feature Transform)

**Purpose:** Detect and describe local features in images that are invariant to scale, rotation, and illumination

**Process:**
```
1. Find keypoints (interest points)
   - Corners, edges, distinctive regions

2. Describe neighborhoods around keypoints
   - Direction and magnitude of gradients
   - Histogram of edge orientations

3. Result: Keypoint descriptor (128D vector)
   - Invariant to many transformations
   - Can match same keypoint across images
```

**Example application:**
```
Image 1: Photo of building from front
Image 2: Photo of same building from side, different lighting

SIFT can find corresponding points:
- Corner of window in both images
- Door handle in both images  
- Logo on building in both images

Use cases:
- Image stitching (panoramas)
- Object recognition
- 3D reconstruction
```

**Advantages:**
✓ Mathematically well-understood
✓ Invariant to common transformations
✓ Works without training data
✓ Interpretable features

**Disadvantages:**
✗ Hand-crafted (not learned from data)
✗ Limited to certain types of features
✗ Not end-to-end optimizable
✗ Slower than modern CNN features

### Deep Learning Approaches

#### Convolutional Neural Networks (CNNs)

**Key insight:** Learn hierarchical features automatically from data

**Convolution operation:**
```
Mathematical definition:
(I * K)[i,j] = ΣΣ I[i+m, j+n] × K[m,n]
               m n

where:
I = input image/feature map
K = kernel/filter  
* = convolution operator

Interpretation:
- Slide kernel across image
- Compute dot product at each position
- Result: Feature map showing kernel responses
```

**Example convolution:**
```
Input (5×5):          Kernel (3×3):
[1 2 3 4 5]          [1  0 -1]
[2 3 4 5 6]          [1  0 -1]  
[3 4 5 6 7]          [1  0 -1]
[4 5 6 7 8]
[5 6 7 8 9]

Output (3×3):
[0  0  0]     # Each value computed as dot product
[0  0  0]     # of kernel with corresponding image region
[0  0  0]
```

**Feature hierarchy:**
```
Layer 1 (early): Edge detectors
  - Vertical edges: [-1 0 1; -1 0 1; -1 0 1]
  - Horizontal edges: [-1 -1 -1; 0 0 0; 1 1 1]
  - Diagonal edges: [1 0 -1; 0 0 0; -1 0 1]

Layer 2: Simple patterns
  - Corners (combination of edges)
  - Curves  
  - Textures

Layer 3: Object parts
  - Eyes, noses (for faces)
  - Wheels, windows (for cars)
  - Leaves, branches (for trees)

Layer 4: Full objects
  - Complete faces
  - Full cars
  - Entire trees
```

#### ResNet (Residual Networks)

**Motivation:** Very deep networks are hard to train

**The problem:**
```
Intuition: Deeper = more parameters = better?
But: Very deep networks are hard to train!

Cause: Gradient vanishing during backpropagation
Backprop through L layers:

∂Loss/∂θ₁ = ∂Loss/∂h_L × ∂h_L/∂h_{L-1} × ∂h_{L-1}/∂h_{L-2} × ... × ∂h₂/∂h₁ × ∂h₁/∂θ₁

Chain rule multiplication: If each ∂h_i/∂h_{i-1} ≈ g < 1:
Final gradient ≈ g^L × (initial gradient)

Example with L=100 layers and g=0.9:
0.9¹⁰⁰ ≈ 0.0000027 (essentially zero!)

Result: Early layers receive almost no gradient signal
```

**Solution: Skip connections (residual connections)**

**Architecture change:**
```
Traditional layer: h_{i+1} = f(h_i)
Residual layer: h_{i+1} = h_i + f(h_i)

where f(h_i) is typically:
Conv → BatchNorm → ReLU → Conv → BatchNorm
```

**Why this helps:**
```
Benefit:
Even if f(h_i) learns nothing (f(h_i)=0),
h_{i+1} = h_i still flows information through

Gradient paths (using chain rule correctly):
Without skip connections:
  ∂h_{i+1}/∂h_i = f'(h_i)

With skip connections:  
  ∂h_{i+1}/∂h_i = ∂(h_i + f(h_i))/∂h_i = 1 + f'(h_i)

The "+1" term provides direct gradient pathway!

Through L layers:
Without skip: gradient ∝ ∏ᵢ f'(h_i) (product of derivatives < 1)
With skip: gradient includes terms with ∏ᵢ (1 + f'(h_i)) (always ≥ 1)

The identity mappings prevent gradient vanishing!
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
Stage 1: 3 residual blocks
→ (56×56×256)
  ↓
Stage 2: 4 residual blocks  
→ (28×28×512)
  ↓
Stage 3: 6 residual blocks
→ (14×14×1024)
  ↓
Stage 4: 3 residual blocks
→ (7×7×2048)
  ↓
Global Average Pool
→ (2048,)
  ↓
Fully Connected
→ (num_classes,)
```

**Properties:**
```
ResNet-50 output:
- 2048-dimensional feature vector
- Captures high-level semantic content
- Pre-trained on ImageNet (1.2M images, 1000 classes)
- Transfer learning: Works well for new tasks
```

**Advantages:**
✓ Much deeper networks possible (50, 101, 152 layers)
✓ Better performance than shallow networks
✓ Stable training (deep networks possible)
✓ Interpretable to some extent (visualize activations)
✓ Efficient inference

**Disadvantages:**
✗ Black-box decisions (what does each dimension mean?)
✗ Requires large labeled datasets to train from scratch
✗ Inherits biases from ImageNet

**When to use:**
- Most modern computer vision tasks
- Transfer learning base
- Feature extraction for multimodal systems

### Vision Transformer (ViT) - Modern Alternative

**Key idea:** Treat image patches as sequence tokens, apply transformer

**Process:**
```
1. Split image into patches (e.g., 16×16 patches)
   224×224 image → 14×14 = 196 patches

2. Linear projection of each patch
   16×16×3 = 768D → Linear layer → 768D embedding

3. Add positional embeddings
   Patch embeddings + position info

4. Transformer encoder  
   Self-attention across patches

5. Classification token [CLS]
   Final representation for whole image
```

**Comparison with CNNs:**
```
CNNs:                    ViTs:
- Inductive bias         - Less inductive bias
- Local connectivity     - Global attention
- Translation equivariance - Learned spatial relationships  
- Smaller datasets OK    - Needs large datasets
- More efficient         - More computation
```

## 3.3 Audio Representation

### Traditional Signal Processing

#### Mel-frequency Cepstral Coefficients (MFCC)

**Purpose:** Extract perceptually meaningful features from audio

**Process:**
```
1. Pre-emphasis filter
   Boost high frequencies
   
2. Windowing  
   Split audio into overlapping frames (25ms windows, 10ms step)
   
3. FFT (Fast Fourier Transform)
   Time domain → Frequency domain
   
4. Mel filter bank
   Human auditory perception-based frequency spacing
   
5. Logarithm
   Compress dynamic range
   
6. DCT (Discrete Cosine Transform)
   Decorrelate features
   
Output: Typically 13 MFCC coefficients per frame
```

**Visual representation:**
```
Audio waveform:
Time: 0----1----2----3----4----5 seconds
Amplitude: ∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼
        ↓
MFCC features (13 × num_frames):
Frame: 1    2    3    4    5   ...
c₁:   [2.1  1.8  2.3  1.9  2.0]
c₂:   [0.5  0.3  0.7  0.4  0.6]  
c₃:   [-0.2 0.1 -0.3  0.0 -0.1]
...
c₁₃:  [0.8  0.9  0.7  1.0  0.8]
        ↓
Final representation per utterance:
Statistical summary (mean, std) → 26D vector
Or sequence of frame vectors for RNN processing
```

**Example use case:**
```
Input: Audio "Hello"
        ↓
MFCC extraction
        ↓
Output: Text "Hello"
```

**Properties:**
- Fixed dimensionality (39D total: 13 MFCC + 13 Δ + 13 ΔΔ)
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
✗ Designed specifically for speech
✗ Not optimal for music or environmental sounds

#### Spectrograms

**Purpose:** Visualize frequency content over time

**Types:**
```
1. Linear spectrogram:
   FFT magnitudes plotted over time
   Y-axis: Frequency (0 to Nyquist)
   X-axis: Time
   Color: Magnitude

2. Log spectrogram:
   Log-scale frequency axis
   Better for human perception

3. Mel spectrogram:
   Mel-scale frequency axis
   Even better perceptual modeling
```

**Advantages:**
✓ Complete frequency information preserved
✓ Raw frequency content visible
✓ Can apply deep learning directly
✓ Works for any audio (not just speech)

**Disadvantages:**
✗ High dimensionality (harder to process)
✗ Not perceptually normalized
✗ Less standard for speech

**When to use:**
- Music processing and generation
- Environmental sound classification
- Any audio task where full frequency content matters

### Modern Deep Learning Approaches

#### Wav2Vec 2.0

**Purpose:** Learn audio representations from raw waveforms

**Architecture:**
```
Raw audio waveform
        ↓
CNN encoder (6 layers)
        ↓  
Quantization module
        ↓
Transformer (12 layers)
        ↓
Contextualized representations (768D per timestep)
```

**Training:** Self-supervised contrastive learning
```
1. Mask portions of audio  
2. Learn to predict masked regions
3. Use contrastive loss (similar to BERT for text)

Result: Rich audio representations without labeled data
```

**Advantages:**
✓ Learned from data (not hand-crafted)
✓ Works across different audio domains
✓ State-of-the-art for many audio tasks
✓ Can fine-tune for specific tasks

**Disadvantages:**
✗ Requires large amounts of training data
✗ Computationally expensive
✗ Black-box (hard to interpret)

## 3.4 Practical Implementation Examples

### Text Feature Extraction

```python
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class TextFeatureExtractor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
    
    def extract_features(self, text):
        """Extract BERT features from text"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', 
                              padding=True, truncation=True, max_length=512)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use [CLS] token embedding as sentence representation
        sentence_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
        
        return sentence_embedding.numpy()

# Usage
extractor = TextFeatureExtractor()
features = extractor.extract_features("A cute cat sitting on a mat")
print(f"Text features shape: {features.shape}")  # (1, 768)
```

### Image Feature Extraction

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageFeatureExtractor:
    def __init__(self):
        # Load pre-trained ResNet
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()  # Remove final classifier
        self.model.eval()
        
        # Standard ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_features(self, image_path):
        """Extract ResNet features from image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)  # Shape: (1, 2048)
            
        return features.numpy()

# Usage  
extractor = ImageFeatureExtractor()
features = extractor.extract_features("cat.jpg")
print(f"Image features shape: {features.shape}")  # (1, 2048)
```

### Audio Feature Extraction

```python
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class AudioFeatureExtractor:
    def __init__(self):
        # Traditional MFCC
        self.sample_rate = 16000
        
        # Modern Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.eval()
    
    def extract_mfcc(self, audio_path):
        """Extract MFCC features"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Take statistical summary
        mfcc_mean = np.mean(mfcc, axis=1)  # (13,)
        mfcc_std = np.std(mfcc, axis=1)    # (13,)
        
        return np.concatenate([mfcc_mean, mfcc_std])  # (26,)
    
    def extract_wav2vec(self, audio_path):
        """Extract Wav2Vec2 features"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Process audio
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Average over time dimension
        features = outputs.last_hidden_state.mean(dim=1)  # Shape: (1, 768)
        
        return features.numpy()

# Usage
extractor = AudioFeatureExtractor()
mfcc_features = extractor.extract_mfcc("hello.wav")
wav2vec_features = extractor.extract_wav2vec("hello.wav")

print(f"MFCC features shape: {mfcc_features.shape}")      # (26,)
print(f"Wav2Vec features shape: {wav2vec_features.shape}") # (1, 768)
```

## 3.5 Debugging Feature Extraction

### Common Issues and Solutions

**Issue 1: Features are all zeros or very small**
```python
def debug_features(features, name="features"):
    print(f"{name} statistics:")
    print(f"  Shape: {features.shape}")
    print(f"  Min: {features.min():.6f}")
    print(f"  Max: {features.max():.6f}")
    print(f"  Mean: {features.mean():.6f}")
    print(f"  Std: {features.std():.6f}")
    print(f"  Zeros: {(features == 0).sum()} / {features.size}")
    
    if features.std() < 1e-6:
        print("  WARNING: Very low variance - check preprocessing!")
    if np.isnan(features).any():
        print("  WARNING: NaN values detected!")
    if np.isinf(features).any():
        print("  WARNING: Infinite values detected!")
```

**Issue 2: Inconsistent feature scales across modalities**
```python
def normalize_features(features, method='l2'):
    """Normalize features for consistent scale"""
    if method == 'l2':
        # L2 normalization (unit length)
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / (norm + 1e-8)
    elif method == 'zscore':
        # Z-score normalization  
        mean = features.mean(axis=-1, keepdims=True)
        std = features.std(axis=-1, keepdims=True)
        return (features - mean) / (std + 1e-8)
    elif method == 'minmax':
        # Min-max normalization
        min_val = features.min(axis=-1, keepdims=True)
        max_val = features.max(axis=-1, keepdims=True)
        return (features - min_val) / (max_val - min_val + 1e-8)
```

**Issue 3: Memory issues with large feature matrices**
```python
def batch_feature_extraction(file_paths, extractor, batch_size=32):
    """Process files in batches to avoid memory issues"""
    features = []
    
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i+batch_size]
        batch_features = []
        
        for path in batch_paths:
            feat = extractor.extract_features(path)
            batch_features.append(feat)
            
        # Stack batch and free memory
        batch_features = np.vstack(batch_features)
        features.append(batch_features)
        
        # Progress indicator
        print(f"Processed {min(i+batch_size, len(file_paths))}/{len(file_paths)} files")
    
    return np.vstack(features)
```

## 3.6 Exercises and Projects

**⭐ Beginner:**
1. Implement BoW and TF-IDF from scratch
2. Extract MFCC features from audio files
3. Visualize CNN filter responses on images
4. Compare different text representations on sentiment analysis

**⭐⭐ Intermediate:**
5. Fine-tune BERT on domain-specific text
6. Extract ResNet features and cluster images

**⭐⭐⭐ Advanced:**
7. Implement self-attention for images (simplified ViT)
8. Build Wav2Vec2 from scratch (simplified)
9. Compare different dimensionality reduction techniques

---

## Key Takeaways

- **Text representations** evolved from simple BoW to contextual embeddings (BERT, GPT)
- **Image features** benefit from hierarchical processing (CNNs) and skip connections (ResNet)
- **Audio processing** uses both traditional signal processing (MFCC) and modern deep learning (Wav2Vec2)
- **Feature quality** is crucial for downstream multimodal tasks
- **Normalization** is essential when combining features from different modalities
- **Debugging tools** help identify and fix common feature extraction issues

## Further Reading

**Text Representations:**
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv:1301.3781*
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *arXiv:1810.04805*

**Computer Vision:**
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*

**Audio Processing:**
- Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS 2020*

---

**Previous**: [Chapter 2: Foundations and Core Concepts](chapter-02.md) | **Next**: [Chapter 4: Feature Alignment and Bridging Modalities](chapter-04.md) | **Home**: [Table of Contents](index.md)
