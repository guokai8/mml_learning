# Chapter 5: Fusion Strategies

---

**Previous**: [Chapter 4: Feature Alignment and Bridging Modalities](chapter-04.md) | **Next**: [Chapter 6: Attention Mechanisms in Multimodal Systems](chapter-06.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Understand three levels of fusion (early, mid, late)
- Implement each fusion strategy
- Know when to use each approach
- Combine multiple fusion methods
- Handle missing modalities

## 5.1 Three Fusion Architectures

### Level 1: Early Fusion (Raw Data Level)

**Concept:**
Combine raw or minimally processed data before any feature extraction

**Process:**

```
Raw modality 1: Image pixels (224×224×3 = 150,528 values)
Raw modality 2: Text words (50 tokens × 300D = 15,000 values)
                ↓
         Concatenation
                ↓
    Combined vector (165,528D)
                ↓
         Joint model (CNN/Transformer)
                ↓
            Prediction
```

**Example architecture:**

```python
class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Combined input: 165,528D
        self.fc1 = nn.Linear(165528, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, image_pixels, text_embeddings):
        # Flatten and concatenate
        image_flat = image_pixels.reshape(image_pixels.shape[0], -1)
        text_flat = text_embeddings.reshape(text_embeddings.shape[0], -1)
        combined = torch.cat([image_flat, text_flat], dim=1)

        # Process through network
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)

        return out
```

**Analysis:**

**Advantages:**
✓ Model can learn all interactions (nothing hidden)
✓ Simple to understand
✓ In theory, powerful (can learn anything)

**Disadvantages:**
✗ Extremely high dimensionality (165K features!)
✗ Serious overfitting risk with limited data
✗ Ignores modality-specific structure
  - CNN architectures don't help
  - Text structure not leveraged
  - Image spatial correlations ignored
✗ Model must learn modality-specific patterns from scratch
✗ One noisy modality ruins everything
✗ No transfer learning possible (no pre-trained models)

**When to use:**
- Tiny datasets (where all information essential)
- Unlimited computational resources
- Modalities are tightly coupled (rare)
- As baseline only (not recommended for practice)

**Real-world likelihood:**
❌ Almost never used in practice
❌ Only for academic comparisons

---

### Level 2: Mid-Fusion (Feature Level) - Most Common

**Concept:**
Process each modality separately, then fuse features

**Process:**

```
Image (224×224×3)
    ├─ Image encoder (ResNet50)
    └─ Image features (2048D)
                ├─ Projection to shared space (256D)
                │
Text (50 tokens)        │
    ├─ Text encoder (BERT)     │
    └─ Text features (768D)    │
                ├─ Projection to shared space (256D)
                │
                └─ Fusion module
                        ├─ Concatenation
                        ├─ Addition
                        ├─ Multiplication
                        ├─ Bilinear
                        └─ Attention
                        ↓
                    Fused features
                        ↓
                    Classifier
                        ↓
                    Prediction
```

**Multiple fusion options:**

**Option 1: Simple concatenation**

```
image_proj: [0.1, 0.2, ..., 0.5] (256D)
text_proj: [0.3, -0.1, ..., 0.8] (256D)

Concatenated: [0.1, 0.2, ..., 0.5, 0.3, -0.1, ..., 0.8] (512D)

Code:
  fused = torch.cat([img_proj, txt_proj], dim=1)  # (batch, 512)
```

**Option 2: Element-wise addition**

```
Both projected to same dimension (256D)

image_proj: [0.1, 0.2, 0.3, ...]
text_proj:  [0.3, -0.1, 0.2, ...]
            ───────────────────
Sum:        [0.4, 0.1, 0.5, ...]  (256D)

Code:
  fused = img_proj + txt_proj

Interpretation:
  Dimensions with high values in both → amplified
  Dimensions with opposite signs → cancel out
  Result: Finds agreement between modalities
```

**Option 3: Element-wise multiplication (Hadamard product)**

```
image_proj: [0.5, 0.2, 0.8, ...]
text_proj:  [0.9, 0.1, 0.3, ...]
            ─────────────────────
Product:    [0.45, 0.02, 0.24, ...]  (256D)

Code:
  fused = img_proj * text_proj

Interpretation:
  Emphasizes dimensions where BOTH are large
  Downplays where one is small
  Creates AND-like interaction (both must agree)

Example:
  Image dimension "red": 0.9 (strong red feature)
  Text dimension "red": 0.8 (word "red" present)
  Product: 0.72 (strong agreement on red)

  Image dimension "square": 0.1 (weak square feature)
  Text dimension "square": 0.05 (weak word mention)
  Product: 0.005 (both weak, product weaker)
```

**Option 4: Bilinear pooling**

```
Captures pairwise interactions

fused = img_proj^T @ W @ txt_proj

where W ∈ ℝ^(256 × 256) is learnable matrix

Result: Single scalar (interaction strength)

Code:
  W = nn.Parameter(torch.randn(256, 256))
  interaction = torch.einsum('bi,ij,bj->b', img_proj, W, txt_proj)

Interpretation:
  All-pairs interaction between dimensions
  More expressive than element-wise operations
  But higher computational cost
```

**Option 5: Concatenation + attention**

```
Keep both representations separate
Use attention to combine

Query: image features (256D)
Key/Value: text features (256D)

attention_weights = softmax(Query @ Key^T / sqrt(d))
text_attended = attention_weights @ Value

Combined: [image_features, text_attended]  (512D)

Code:
  attention_scores = img_proj @ txt_proj.t()
  attention_weights = softmax(attention_scores / sqrt(256))
  txt_attended = attention_weights @ txt_proj
  combined = torch.cat([img_proj, txt_attended], dim=1)
```

**Example - Sentiment analysis with mid-fusion:**

```
Task: Predict sentiment from image + text

Input:
  Image: [Happy face]
  Text: "I love this!"

Processing:

① Feature extraction
   Image → ResNet50 → 2048D features
   Text → BERT → 768D features

② Dimensionality reduction (optional)
   Image → Linear(2048→256) → 256D
   Text → Linear(768→256) → 256D

③ Fusion options:

   Option A - Addition:
     fused = img + text = [0.5, 0.3, ..., 0.2] (256D)
     Interpretation: Aggregate all information

   Option B - Multiplication:
     fused = img * text = [0.45, 0.06, ..., 0.04] (256D)
     Interpretation: Emphasize agreement

   Option C - Concatenation:
     fused = [img; text] = [512D]
     Interpretation: Keep all information separate

④ Classification
   Linear layer: 256D/512D → 3 (pos/neutral/neg)

⑤ Prediction
   Output: Positive sentiment (0.92 confidence)
```

**Advantages of mid-fusion:**
✓ Each modality processed with appropriate encoder
✓ Transfer learning from pre-trained models
✓ Reasonable dimensionality (512D vs 165K)
✓ Flexible fusion options
✓ Each modality can be fine-tuned independently
✓ Good balance of modeling power and efficiency

**Disadvantages:**
✗ Some cross-modal interactions missed (due to independent encoding)
✗ Requires projecting to common space
✗ Hyperparameter choices (dimension, fusion method)

**When to use:**
✓ Most standard applications
✓ When each modality has good encoder
✓ Balanced importance across modalities
✓ **Most recommended approach for practice**

---

### Level 3: Late Fusion (Decision Level)

**Concept:**
Each modality makes independent prediction, then combine decisions

**Process:**

```
Image
    ├─ Image encoder
    ├─ Image classifier
    └─ Image prediction: [0.7, 0.2, 0.1]  (3 class probs)
                ├─ Combine predictions
                ├─ Voting
Text            ├─ Averaging
    ├─ Text encoder    ├─ Weighted sum
    ├─ Text classifier ├─ Bayesian fusion
    └─ Text prediction: [0.3, 0.5, 0.2]
                    ↓
                Final prediction
                    ↓
                Output class
```

**Multiple combination strategies:**

**Strategy 1: Voting (Majority)**

```
Image prediction: Class 0 (highest prob 0.7)
Text prediction: Class 1 (highest prob 0.5)

Vote:
  Class 0: 1 vote
  Class 1: 1 vote

Result: Tie!
Tiebreaker needed: Pick randomly or use confidence

Code:
  img_pred = torch.argmax(img_logits)
  txt_pred = torch.argmax(txt_logits)

  if img_pred == txt_pred:
    final_pred = img_pred
  else:
    # Use highest confidence
    img_conf = torch.max(img_logits)
    txt_conf = torch.max(txt_logits)
    final_pred = img_pred if img_conf > txt_conf else txt_pred
```

**Strategy 2: Averaging probabilities**

```
Image probs:     [0.7, 0.2, 0.1]
Text probs:      [0.3, 0.5, 0.2]
                 ─────────────────
Average:         [0.5, 0.35, 0.15]

Final prediction: Class 0 (0.5 probability)

Code:
  avg_probs = (img_probs + txt_probs) / 2
  final_pred = torch.argmax(avg_probs)
```

**Strategy 3: Weighted averaging**

```
Weight image more (assume it's more reliable):
  w_img = 0.7
  w_txt = 0.3

Weighted combination:
  [0.7*0.7 + 0.3*0.3, 0.7*0.2 + 0.3*0.5, 0.7*0.1 + 0.3*0.2]
= [0.49+0.09, 0.14+0.15, 0.07+0.06]
= [0.58, 0.29, 0.13]

Final prediction: Class 0

Code:
  weighted_probs = w_img * img_probs + w_txt * txt_probs
  final_pred = torch.argmax(weighted_probs)
```

**Strategy 4: Product of probabilities (Bayesian)**

```
Idea: Multiply probabilities across modalities
Assumption: Modalities independent given true class

Image probs:     [0.7, 0.2, 0.1]
Text probs:      [0.3, 0.5, 0.2]
                 ──────────────────
Product:         [0.21, 0.10, 0.02]
Normalized:      [0.66, 0.31, 0.03]

Final: Class 0

Code:
  combined = img_probs * txt_probs
  combined = combined / torch.sum(combined)  # Normalize
  final_pred = torch.argmax(combined)
```

**Strategy 5: Maximum (Optimistic)**

```
Take maximum probability for each class

Class 0: max(0.7, 0.3) = 0.7
Class 1: max(0.2, 0.5) = 0.5
Class 2: max(0.1, 0.2) = 0.2

Result: [0.7, 0.5, 0.2]
But: Doesn't sum to 1! (Renormalize: [0.538, 0.385, 0.154])

Code:
  combined = torch.max(img_probs, txt_probs)
  combined = combined / torch.sum(combined)
```

**Example - Medical diagnosis:**

```
Patient data: CT scan + blood tests + symptoms

Model 1 (Image-based):
  Analyzes CT scan
  Prediction: "Likely cancer" (0.85)
                "Uncertain" (0.12)
                "Unlikely" (0.03)

Model 2 (Lab-based):
  Analyzes blood markers
  Prediction: "Likely cancer" (0.62)
                "Uncertain" (0.25)
                "Unlikely" (0.13)

Combination strategies:

① Averaging:
   Result: [0.735, 0.185, 0.08]
   → "Likely cancer" (73.5%)

② Weighted (trust image more):
   0.7 × Image + 0.3 × Lab
   → "Likely cancer" (76.9%)

③ Product (Bayesian):
   0.85*0.62 / Z = [0.781, 0.069, 0.15]
   → "Likely cancer" (78.1%)

Different strategies give similar but slightly different results
```

**Advantages of late fusion:**
✓ Highest modularity
✓ Each modality completely independent
✓ Easy to add/remove modalities
✓ Handles missing modalities gracefully (just skip that classifier)
✓ Easy to debug (know which modality failed)
✓ Can use completely different model types per modality

**Disadvantages:**
✗ Lost fine-grained cross-modal interactions
✗ Each modality must predict well independently
✗ Weaker modality can't be helped by stronger one
✗ Higher computational cost (multiple full pipelines)
✗ Information not shared during training

**When to use:**
✓ Modalities strongly independent
✓ Need robustness to missing modalities
✓ Interpretability important
✓ Modalities have very different characteristics
✓ Each modality has mature specialized model

**Real-world example - Autonomous driving:**

```
Sensor fusion in self-driving car:

Camera → Object detection → Predictions
LIDAR → Distance measurement → Predictions
Radar → Velocity detection → Predictions

Late fusion:
  Each sensor makes independent decision
  "Object at x=100m, y=50m"

  Decisions combined:
  Average positions across sensors
  Confidence = agreement between sensors

Result: Robust detection even if one sensor fails
```

---

## 5.2 Comparison of Fusion Levels

**Summary table:**

```
                Early          Mid            Late
              Fusion         Fusion         Fusion
─────────────────────────────────────────────────
Input         Raw data       Features       Predictions
              (pixels)       (2048D, 768D)  (probabilities)

Computation   Slow           Medium         Fast (per fusion)
Memory        Very high      Medium         Low
Overfitting   High risk      Moderate       Low risk
risk

Cross-modal   Very strong    Strong         None
interaction

Interpretab   Low            Medium         High
-ility

Transfer      Impossible     Excellent      Good
learning      (no pre-trains)

Robustness    Poor           Good           Excellent
to noise

Modularity    Low            Medium         High

When to use   Rare           Most tasks     Special cases
```

**Decision flowchart:**

```
Start
  │
  ├─ Are modalities completely independent?
  │   YES ─→ Consider LATE fusion (modularity)
  │   NO ──→ Continue
  │
  ├─ Must handle missing modalities?
  │   YES ─→ LATE fusion preferred
  │   NO ──→ Continue
  │
  ├─ Have pre-trained encoders?
  │   YES ─→ MID fusion (use them!)
  │   NO ──→ Continue
  │
  ├─ Very small dataset?
  │   YES ─→ MID fusion (leverage pre-training)
  │   NO ──→ Continue
  │
  ├─ Importance of cross-modal interaction?
  │   HIGH ─→ EARLY fusion (but risky!)
  │   LOW ──→ MID or LATE fusion
  │
  └─→ DEFAULT: MID FUSION (best balance)
```

## 5.3 Advanced Fusion: Multimodal Transformers

**Modern approach:** Use transformer architecture for fusion

**Key insight:**
```
Transformers naturally handle multi-modal inputs
Just treat different modalities as different token types
```

**Architecture:**

```
Image patches: [patch_1, patch_2, ..., patch_196]  (from ViT)
Text tokens:   [word_1, word_2, ..., word_N]       (from BERT tokenizer)

Unified input: [[IMG:patch_1], [IMG:patch_2], ...,
                [TXT:word_1], [TXT:word_2], ...]

Modality markers: [IMG, IMG, ..., TXT, TXT, ...]
Position encoding: [0, 1, ..., 196, 197, 198, ...]

Combined tokens + markers + positions
        ↓
Transformer encoder (12 layers)
        ↓
Self-attention between all tokens
(image patches attend to text, vice versa)
        ↓
Output: Multimodal representations
```

**Why this works:**

```
Transformer doesn't care about modality type
Pure attention-based fusion
Each position (patch or token) can attend to all others
Learns how to combine automatically

Example attention pattern:

Word "red" attends to:
  - Red-colored image patches (high weight)
  - Other color-related words (medium weight)
  - Unrelated patches/words (low weight)

Image patch attends to:
  - Corresponding text description (high weight)
  - Related patches (medium weight)
  - Unrelated text (low weight)

All learned without explicit rules!
```

**Example architecture:**

```python
class MultimodalTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=12):
        super().__init__()

        # Embeddings
        self.img_embed = nn.Linear(2048, hidden_dim)  # Project image patches
        self.txt_embed = nn.Embedding(vocab_size, hidden_dim)

        # Modality tokens
        self.img_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.txt_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Position encoding
        self.pos_embed = nn.Embedding(1000, hidden_dim)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, images, text_ids):
        batch_size = images.shape[0]

        # Embed images (196 patches per image)
        img_emb = self.img_embed(images)  # (batch, 196, 768)
        img_emb = img_emb + self.img_token

        # Embed text
        txt_emb = self.txt_embed(text_ids)  # (batch, seq_len, 768)
        txt_emb = txt_emb + self.txt_token

        # Concatenate
        combined = torch.cat([img_emb, txt_emb], dim=1)

        # Add positional encoding
        seq_len = combined.shape[1]
        pos_ids = torch.arange(seq_len, device=combined.device)
        pos_enc = self.pos_embed(pos_ids).unsqueeze(0)
        combined = combined + pos_enc

        # Transformer
        out = self.transformer(combined)

        # Use first token (like BERT [CLS]) for classification
        cls_out = out[:, 0, :]
        logits = self.classifier(cls_out)

        return logits
```

**Advantages:**
✓ Unified architecture
✓ Automatic cross-modal fusion
✓ Scales well
✓ Flexible (add any modality)
✓ State-of-the-art performance

**Disadvantages:**
✗ More complex
✗ Slower inference
✗ Needs careful tuning

## 5.4 Handling Missing Modalities

**Real-world challenge:**

```
Training: All modalities present
Deployment:
  Sometimes only image available
  Sometimes only text available
  Rarely all modalities together

Example scenarios:

E-commerce system:
  Training: 1M products with image + description + reviews
  At test time:
    Product A: Image only (video unavailable)
    Product B: Text only (image not loading)
    Product C: All modalities

Medical system:
  Training: Patients with CT + MRI + blood tests
  At test time:
    Patient A: Only CT scan (MRI machine broken)
    Patient B: CT + blood (MRI not done)
    Patient C: All three
```

### Solution 1: Independent Modality Pipelines

**Approach:**
Train separate models for each modality and combinations

```python
class MultimodalClassifier:
    def __init__(self):
        self.img_only_model = train_image_classifier()
        self.txt_only_model = train_text_classifier()
        self.fusion_model = train_fusion_model()

    def predict(self, image=None, text=None):
        if image is not None and text is not None:
            # Both available: use fusion
            img_features = extract_image_features(image)
            txt_features = extract_text_features(text)
            return self.fusion_model.predict([img_features, txt_features])

        elif image is not None:
            # Only image
            return self.img_only_model.predict(image)

        elif text is not None:
            # Only text
            return self.txt_only_model.predict(text)
```

**Advantages:**
✓ Simple and modular
✓ Good performance per modality
✓ Easy to add modalities

**Disadvantages:**
✗ Requires training multiple models
✗ Duplication of effort
✗ Inconsistent predictions (models disagree)

### Solution 2: Adaptive Fusion with Modality Weighting

**Approach:**
Learn which modalities to trust based on availability

```python
class AdaptiveFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractors
        self.img_extractor = ImageEncoder()
        self.txt_extractor = TextEncoder()

        # Modality gates (learn importance)
        self.gate_img = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.gate_txt = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Fusion and classification
        self.fusion = nn.Linear(2048 + 768, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, image=None, text=None):
        features = []

        if image is not None:
            img_feat = self.img_extractor(image)
            w_img = self.gate_img(img_feat)
            img_feat = img_feat * w_img
            features.append(img_feat)

        if text is not None:
            txt_feat = self.txt_extractor(text)
            w_txt = self.gate_txt(txt_feat)
            txt_feat = txt_feat * w_txt
            features.append(txt_feat)

        # Concatenate available features
        combined = torch.cat(features, dim=1)

        # Pad if missing modalities
        if image is None:
            combined = torch.cat([torch.zeros(batch, 2048), combined])
        if text is None:
            combined = torch.cat([combined, torch.zeros(batch, 768)])

        fused = self.fusion(combined)
        logits = self.classifier(fused)

        return logits
```

**How it works:**
- Gate networks learn importance of each modality
- During training: All modalities penalize equally (gates = 1)
- Some modalities learned as less important (gates < 1)
- At test time: Missing modalities handled gracefully (gate = 0)

**Advantages:**
✓ Single model
✓ Learns to trust modalities
✓ Handles missing data

**Disadvantages:**
✗ More complex training
✗ Potential numerical issues (zeros in features)

### Solution 3: Modality Embedding/Imputation

**Approach:**
Predict missing modalities from available ones

```python
class ImputingFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_encoder = ImageEncoder()
        self.txt_encoder = TextEncoder()

        # Decoders for imputation
        self.img_to_txt_decoder = nn.Linear(2048, 768)
        self.txt_to_img_decoder = nn.Linear(768, 2048)

        # Classification
        self.classifier = nn.Linear(2048 + 768, num_classes)

    def forward(self, image=None, text=None):
        if image is not None:
            img_feat = self.img_encoder(image)
        else:
            txt_feat = self.txt_encoder(text)
            img_feat = self.txt_to_img_decoder(txt_feat)

        if text is not None:
            txt_feat = self.txt_encoder(text)
        else:
            img_feat = self.img_encoder(image)
            txt_feat = self.img_to_txt_decoder(img_feat)

        combined = torch.cat([img_feat, txt_feat], dim=1)
        logits = self.classifier(combined)

        return logits
```

**How it works:**
- If text missing: Predict from image
- If image missing: Predict from text
- Use predictions as if real

**Advantages:**
✓ Single model
✓ Predictions fill in gaps
✓ Cross-modal knowledge transfer

**Disadvantages:**
✗ Predictions may be inaccurate
✗ Error propagation
✗ Requires training decoder networks

## 5.5 Practical Fusion Examples

### Example 1: Image-Text Sentiment Analysis

**Problem:**
Determine sentiment from image + text (social media post)

```python
class SentimentFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
        self.img_encoder = models.resnet50(pretrained=True)
        self.img_encoder = nn.Sequential(*list(self.img_encoder.children())[:-1])

        self.txt_encoder = AutoModel.from_pretrained('bert-base-uncased')

        # Projections to common space (256D)
        self.img_proj = nn.Linear(2048, 256)
        self.txt_proj = nn.Linear(768, 256)

        # Fusion
        self.fusion_options = {
            'concat': FusionConcat(512, 256),
            'add': FusionAdd(),
            'mult': FusionMult(),
            'attention': FusionAttention(256)
        }

        # Classification
        self.classifier = nn.Linear(256, 3)  # 3 sentiments

    def forward(self, image, text, fusion_type='concat'):
        # Extract features
        img_feat = self.img_encoder(image).squeeze(-1).squeeze(-1)

        txt_inputs = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        )
        txt_out = self.txt_encoder(**txt_inputs)
        txt_feat = txt_out.last_hidden_state[:, 0, :]

        # Project to common space
        img_proj = self.img_proj(img_feat)
        txt_proj = self.txt_proj(txt_feat)

        # Normalize
        img_proj = F.normalize(img_proj, p=2, dim=1)
        txt_proj = F.normalize(txt_proj, p=2, dim=1)

        # Fuse
        fused = self.fusion_options[fusion_type](img_proj, txt_proj)

        # Classify
        logits = self.classifier(fused)

        return logits

# Fusion modules
class FusionConcat(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, img, txt):
        combined = torch.cat([img, txt], dim=1)
        return self.fc(combined)

class FusionAdd(nn.Module):
    def forward(self, img, txt):
        return img + txt

class FusionMult(nn.Module):
    def forward(self, img, txt):
        return img * txt

class FusionAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8)

    def forward(self, img, txt):
        # img, txt: (batch, dim)
        # Reshape for attention: (seq_len=1, batch, dim)
        img_seq = img.unsqueeze(0)
        txt_seq = txt.unsqueeze(0)

        # txt attends to img
        attended, _ = self.attention(txt_seq, img_seq, img_seq)
        return attended.squeeze(0)
```

**Training:**

```python
model = SentimentFusionModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for images, texts, labels in train_loader:
        # Forward pass with different fusion strategies
        logits = model(images, texts, fusion_type='attention')

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    accuracy = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Accuracy = {accuracy:.3f}")
```

---

### Example 2: Video Understanding with Audio-Visual Fusion

**Problem:**
Classify video action considering both visual frames and audio

```python
class AudioVisualFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Visual encoder (3D CNN for video)
        self.visual_encoder = Video3DCNN(output_dim=512)

        # Audio encoder
        self.audio_encoder = AudioCNN(output_dim=256)

        # Temporal models
        self.visual_lstm = nn.LSTM(512, 256, batch_first=True)
        self.audio_lstm = nn.LSTM(256, 256, batch_first=True)

        # Fusion
        self.fusion = nn.Linear(512, 256)

        # Classification
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, video, audio):
        batch_size = video.shape[0]
        num_frames = video.shape[1]

        # Process video frames
        visual_features = []
        for t in range(num_frames):
            frame_feat = self.visual_encoder(video[:, t])  # (batch, 512)
            visual_features.append(frame_feat)
        visual_seq = torch.stack(visual_features, dim=1)  # (batch, num_frames, 512)

        # LSTM over frames
        visual_out, _ = self.visual_lstm(visual_seq)
        visual_final = visual_out[:, -1, :]  # (batch, 256) - last frame

        # Process audio
        audio_feat = self.audio_encoder(audio)  # (batch, seq_len, 256)
        audio_out, _ = self.audio_lstm(audio_feat)
        audio_final = audio_out[:, -1, :]  # (batch, 256) - last time step

        # Fuse
        combined = torch.cat([visual_final, audio_final], dim=1)  # (batch, 512)
        fused = self.fusion(combined)

        # Classify
        logits = self.classifier(fused)

        return logits
```

**Key considerations:**
- Video: Multiple frames, visual information
- Audio: Temporal signal, semantic content
- Synchronization: Both should be aligned in time
- Late fusion: Aggregate final representations

---

## Key Takeaways

- **Early fusion:** Raw data level, high dimensionality, rarely used
- **Mid fusion:** Feature level, standard approach, recommended
- **Late fusion:** Decision level, modular, handles missing data well
- **Transformers:** Modern approach, automatic fusion
- **Missing modalities:** Solutions include independent models, adaptive weighting, imputation
- **Choose based on:** Data characteristics, modality importance, missing data handling

## Exercises

**⭐ Beginner:**
1. Implement early, mid, late fusion for simple dataset
2. Compare fusion strategies on evaluation metrics
3. Visualize combined feature space

**⭐⭐ Intermediate:**
4. Build adaptive fusion with modality gates
5. Handle missing modalities with multiple strategies
6. Compare computational costs of different approaches

**⭐⭐⭐ Advanced:**
7. Implement multimodal transformer from scratch
8. Design adaptive weighting scheme for heterogeneous data
9. Build system handling variable numbers of modalities

