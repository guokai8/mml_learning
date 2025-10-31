# Chapter 11: Practical Implementation Guide

---

**Previous**: [Chapter 10: Seminal Models and Architectures](chapter-10.md) | **Next**: [Chapter 12: Advanced Topics and Future Directions](chapter-12.md) | **Home**: [Table of Contents](index.md)

---

## Learning Objectives

After reading this chapter, you should be able to:
- Collect and preprocess multimodal datasets
- Build production-ready training pipelines
- Handle edge cases and failures
- Deploy models efficiently
- Monitor and maintain systems
- Implement best practices for MLOps

> 💡 **Note on fusion (what this revision adds):** Wherever your training code calls your model’s `forward(images, text_ids, text_mask)`, that’s where **fusion** should occur. In Section **11.2 → Multimodal Model (Fusion Point)**, we include a compact model example with **concat / gated / late / cross‑attention** fusion so you can drop it in directly.

## 11.1 Data Collection and Preprocessing

### Building Multimodal Datasets

**Data sources:**

```
Web-scale data:
  LAION (5.8B images + captions)
  Conceptual Captions (3.3M pairs)
  Wikipedia + images
  News articles + images
  Social media posts + images/video

Curated datasets:
  COCO (image captioning)
  Flickr30K (image-text)
  Visual Genome (regions + descriptions)
  ActivityNet (video + captions)

Synthetic/Generated:
  Text descriptions from writers
  AI-generated descriptions
  Rule-based generation
```

**Data quality considerations:**

```
Issue 1: Image-text mismatch
  Problem: Caption doesn't describe image
  Solution: Filter with CLIP-based similarity

Issue 2: Duplicate or near-duplicate pairs
  Problem: Same image with different captions
  Solution: Hash-based deduplication

Issue 3: Offensive or sensitive content
  Problem: Dataset contains harmful content
  Solution: Content moderation filters

Issue 4: Biases in distribution
  Problem: Skewed toward certain domains
  Solution: Stratified sampling, data augmentation

Issue 5: Missing or corrupted files
  Problem: Broken image links, corrupted videos
  Solution: Validation pipeline
```

### Preprocessing Pipeline

**Step 1: Image preprocessing**

```python
from torchvision import transforms
from PIL import Image
import torch

class ImagePreprocessor:
    def __init__(self, input_size=224):
        self.input_size = input_size

        # Training transforms (with augmentation)
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Validation transforms (no augmentation)
        self.val_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_path, is_train=True):
        \"\"\"Load and preprocess image\"\"\"
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Apply transforms
            image = self.train_transforms(image) if is_train else self.val_transforms(image)
            return image
        except Exception as e:
            print(f\"Error processing {image_path}: {e}\")
            return None

    def preprocess_batch(self, image_paths, is_train=True):
        \"\"\"Preprocess batch of images\"\"\"
        images, valid_paths = [], []
        for path in image_paths:
            img = self.preprocess_image(path, is_train)
            if img is not None:
                images.append(img); valid_paths.append(path)
        return (torch.stack(images), valid_paths) if images else (None, [])

# Example usage
preprocessor = ImagePreprocessor(input_size=224)
image_batch, valid_paths = preprocessor.preprocess_batch(
    image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    is_train=True
)
```

**Step 2: Text preprocessing**

```python
from transformers import AutoTokenizer

class TextPreprocessor:
    def __init__(self, model_name='bert-base-uncased', max_length=77):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def clean_text(self, text):
        \"\"\"Clean text\"\"\"
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters (keep basic punctuation)
        import re
        text = re.sub(r'[^\\w\\s\\.\\,\\!\\?\\-\\']', '', text)
        # Lowercase
        return text.lower()

    def tokenize(self, text):
        \"\"\"Tokenize single text\"\"\"
        cleaned = self.clean_text(text)
        return self.tokenizer(
            cleaned,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True
        )

    def tokenize_batch(self, texts):
        \"\"\"Tokenize batch of texts\"\"\"
        cleaned = [self.clean_text(text) for text in texts]
        return self.tokenizer(
            cleaned,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            batch_first=True
        )

# Example
text_proc = TextPreprocessor()
tokens = text_proc.tokenize_batch([
    \"A red cat on a wooden chair\",
    \"Two dogs playing in the park\"
])
print(tokens['input_ids'].shape)  # (2, 77)
```

**Step 3: Video preprocessing**

```python
import cv2
import numpy as np

class VideoPreprocessor:
    def __init__(self, fps=1, frame_count=8, frame_size=224):
        self.fps = fps
        self.frame_count = frame_count
        self.frame_size = frame_size

    def extract_frames(self, video_path):
        \"\"\"Extract frames from video\"\"\"
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, self.frame_count, dtype=int)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()

            return np.stack(frames) if frames else None  # (T, H, W, 3)
        except Exception as e:
            print(f\"Error processing video {video_path}: {e}\")
            return None

# Example
video_proc = VideoPreprocessor(frame_count=8)
frames = video_proc.extract_frames('video.mp4')
print(frames.shape)  # (8, 224, 224, 3)
```

### Complete Preprocessing Pipeline

```python
import torch

class MultimodalDataPreprocessor:
    \"\"\"Complete preprocessing for image-text-video data\"\"\"\n    def __init__(self, image_size=224, max_text_length=77, video_frames=8,\n                 text_model_name='bert-base-uncased'):\n        self.image_preprocessor = ImagePreprocessor(image_size)\n        self.text_preprocessor = TextPreprocessor(model_name=text_model_name, max_length=max_text_length)\n        self.video_preprocessor = VideoPreprocessor(frame_count=video_frames)\n\n    def process_sample(self, sample):\n        \"\"\"Process single multimodal sample\"\"\"\n        processed = {}\n        # Image\n        if 'image_path' in sample:\n            img = self.image_preprocessor.preprocess_image(sample['image_path'], is_train=sample.get('is_train', True))\n            if img is not None:\n                processed['image'] = img\n        # Text\n        if 'text' in sample:\n            tokens = self.text_preprocessor.tokenize(sample['text'])\n            processed['text_ids'] = tokens['input_ids'].squeeze()\n            processed['text_mask'] = tokens['attention_mask'].squeeze()\n        # Video\n        if 'video_path' in sample:\n            frames = self.video_preprocessor.extract_frames(sample['video_path'])\n            if frames is not None:\n                processed['video'] = torch.from_numpy(frames).float()\n        # Label (if available)\n        if 'label' in sample:\n            processed['label'] = torch.tensor(sample['label'])\n        return processed\n\n    def validate_sample(self, sample):\n        \"\"\"Check if raw sample lists required modalities\"\"\"\n        required_keys = sample.get('required_modalities', ['image', 'text'])\n        return all(k in sample for k in required_keys)\n\n# Usage\npreprocessor = MultimodalDataPreprocessor()\nraw = {\n    'image_path': 'cat.jpg',\n    'text': 'A cute cat on a sofa',\n    'label': 0,\n    'is_train': True,\n    'required_modalities': ['image', 'text']\n}\nif preprocessor.validate_sample(raw):\n    processed = preprocessor.process_sample(raw)\n    print(f\"Image shape: {processed['image'].shape}\")\n    print(f\"Text IDs shape: {processed['text_ids'].shape}\")\n```

## 11.2 Building Training Pipelines

### Data Loading with Multiprocessing

```python
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

class MultimodalDataset(Dataset):
    \"\"\"Efficient multimodal dataset\"\"\"\n    def __init__(self, samples, preprocessor, cache_size=1000):\n        self.samples = samples\n        self.preprocessor = preprocessor\n        self.cache = {}\n        self.cache_size = cache_size\n    def __len__(self):\n        return len(self.samples)\n    def __getitem__(self, idx):\n        if idx in self.cache:\n            return self.cache[idx]\n        sample = self.samples[idx]\n        processed = self.preprocessor.process_sample(sample)\n        if len(self.cache) < self.cache_size:\n            self.cache[idx] = processed\n        return processed\n\ndef create_dataloaders(train_samples, val_samples, batch_size=256, num_workers=8):\n    \"\"\"Create train and validation dataloaders\"\"\"\n    preprocessor = MultimodalDataPreprocessor()\n    train_dataset = MultimodalDataset(train_samples, preprocessor)\n    val_dataset = MultimodalDataset(val_samples, preprocessor)\n    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,\n                              num_workers=num_workers, pin_memory=True, drop_last=True)\n    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,\n                            num_workers=num_workers, pin_memory=True, drop_last=False)\n    return train_loader, val_loader\n\n# Usage\n# train_loader, val_loader = create_dataloaders(train_data, val_data, batch_size=256, num_workers=8)\n```

### Multimodal Model (Fusion Point)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(feat, out_dim)
    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, name='distilbert-base-uncased', out_dim=256):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.proj = nn.Linear(self.enc.config.hidden_size, out_dim)
    def forward(self, ids, mask):
        cls = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:,0]
        return F.normalize(self.proj(cls), dim=-1)

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim,1), nn.Sigmoid())
    def forward(self, a,b):
        g = self.fc(torch.cat([a,b],-1))
        return g*a + (1-g)*b

class CrossAttentionFusion(nn.Module):
    """Lightweight cross-attention: text queries attend to image features.
    For production, swap single-vector embeddings for ViT patch tokens.
    """
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.ReLU(), nn.Linear(dim*4, dim))
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
    def forward(self, img_emb, txt_emb):
        B, D = img_emb.shape
        L = 4
        img_seq = img_emb.unsqueeze(1).repeat(1, L, 1)
        txt_seq = txt_emb.unsqueeze(1).repeat(1, L, 1)
        q, k, v = self.q(txt_seq), self.k(img_seq), self.v(img_seq)
        attn, _ = self.attn(q, k, v)
        x = self.n1(txt_seq + attn)
        x = self.n2(x + self.ffn(x))
        return x.mean(1)

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=2, emb=256, fusion='concat'):
        super().__init__()
        self.fusion = fusion
        self.ie, self.te = ImageEncoder(emb), TextEncoder(out_dim=emb)
        if fusion=='concat':
            self.head = nn.Sequential(nn.Linear(emb*2, emb), nn.ReLU(), nn.Dropout(0.1), nn.Linear(emb, num_classes))
        elif fusion=='gated':
            self.gate = GatedFusion(emb)
            self.head = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Dropout(0.1), nn.Linear(emb, num_classes))
        elif fusion=='late':
            self.temp = nn.Parameter(torch.tensor(10.0)); self.head = nn.Linear(1, num_classes)
        elif fusion=='xattn':
            self.xfuse = CrossAttentionFusion(emb)
            self.head = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Dropout(0.1), nn.Linear(emb, num_classes))
        else:
            raise ValueError('fusion must be concat|gated|late|xattn')
    def forward(self, images, text_ids, text_mask):
        img, txt = self.ie(images), self.te(text_ids, text_mask)
        # ✨ FUSION HAPPENS HERE
        if self.fusion=='concat':
            z = torch.cat([img, txt], -1)
        elif self.fusion=='gated':
            z = self.gate(img, txt)
        elif self.fusion=='late':
            z = (F.cosine_similarity(img, txt, -1).unsqueeze(-1) * self.temp)
        else:  # 'xattn'
            z = self.xfuse(img, txt)
        return self.head(z)
```
### Training Loop with Best Practices

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

class MultimodalTrainer:
    \"\"\"Production-ready trainer\"\"\"\n    def __init__(self, model, device='cuda', use_wandb=True):
        self.model = model
        self.device = device
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project='multimodal-learning')

    def train_epoch(self, train_loader, optimizer, scheduler, criterion, scaler=None):
        self.model.train(); total_loss=0; n=0
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            text_ids = batch['text_ids'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(images, text_ids, text_mask)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            else:
                logits = self.model(images, text_ids, text_mask)
                loss = criterion(logits, labels); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if scaler is not None: scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad(); scheduler.step()
            total_loss += loss.item(); n += 1
            pbar.set_postfix({'loss': total_loss/n})
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({'train_loss': loss.item(), 'learning_rate': scheduler.get_last_lr()[0]})
        return total_loss/n

    @torch.no_grad()
    def evaluate(self, val_loader, criterion):
        self.model.eval(); total_loss=total_acc=n=0
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            images = batch['image'].to(self.device)
            text_ids = batch['text_ids'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            logits = self.model(images, text_ids, text_mask)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
            total_loss += loss.item(); total_acc += acc.item(); n += 1
            pbar.set_postfix({'loss': total_loss/n, 'acc': total_acc/n})
        return total_loss/n, total_acc/n

    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-4, warmup_steps=1000):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=warmup_steps/total_steps)
        scaler = torch.cuda.amp.GradScaler()
        criterion = torch.nn.CrossEntropyLoss()
        best_val=float('inf'); patience=5; wait=0
        for epoch in range(num_epochs):
            print(f\"\\n{'='*50}\\nEpoch {epoch+1}/{num_epochs}\\n{'='*50}\")
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, criterion, scaler)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            print(f\"Train Loss: {train_loss:.4f}\\nVal Loss: {val_loss:.4f}\\nVal Acc: {val_acc:.4f}\")
            if self.use_wandb:
                wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})
            if val_loss < best_val:
                best_val = val_loss; wait = 0; self.save_checkpoint(f'best_model_epoch{epoch}.pt')
            else:
                wait += 1
                if wait >= patience:
                    print(f\"Early stopping after {epoch+1} epochs\"); break
        if self.use_wandb: wandb.finish()

    def save_checkpoint(self, path):
        torch.save({'model_state_dict': self.model.state_dict(), 'model_config': getattr(self.model, 'config', None)}, path)
        print(f\"Saved checkpoint to {path}\")

# Usage
# model = MultimodalModel()
# trainer = MultimodalTrainer(model)
# trainer.train(train_loader, val_loader, num_epochs=30, lr=1e-4)
```

## 11.3 Handling Edge Cases and Failures

### Error Handling in Data Loading

```python
class RobustDataLoader:
    \"\"\"Data loader with error handling\"\"\"\n    def __init__(self, dataset, batch_size=32, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.failed_indices = []
    def load_with_retry(self, idx, max_retries=3):
        \"\"\"Load sample with retry logic\"\"\"
        for attempt in range(max_retries):
            try:
                return self.dataset[idx]
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f\"Failed to load sample {idx} after {max_retries} attempts: {e}\")
                    self.failed_indices.append(idx)
                    return None
    def get_valid_batch(self, indices):
        \"\"\"Get batch skipping failed samples\"\"\"
        batch, valid_indices = [], []
        for idx in indices:
            sample = self.load_with_retry(idx)
            if sample is not None:
                batch.append(sample); valid_indices.append(idx)
        if not batch:
            return None, []
        try:
            stacked = {k: torch.stack([s[k] for s in batch]) for k in batch[0].keys()}
            return stacked, valid_indices
        except Exception as e:
            print(f\"Error stacking batch: {e}\"); return None, []
```

### Validation and Sanity Checks

```python
import torch
from typing import Callable

class DataValidator:
    """Validate data quality"""
    @staticmethod
    def check_image_quality(image_tensor, min_entropy=0.5):
        # Histogram-entropy heuristic
        flat = image_tensor.flatten()
        flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)
        hist = torch.histc(flat, bins=256); hist = hist / hist.sum()
        entropy = -(hist * torch.log(hist + 1e-8)).sum()
        return entropy > min_entropy
    @staticmethod
    def check_text_quality(text, min_length=5, max_length=1000):
        if text is None or not isinstance(text, str):
            return False
        text = text.strip()
        if len(text) < min_length or len(text) > max_length:
            return False
        special_chars = sum(1 for c in text if not c.isalnum() and c != ' ')
        return (special_chars / max(len(text),1)) <= 0.5
    @staticmethod
    def check_alignment(image_tensor: torch.Tensor, text_ids: torch.Tensor, text_mask: torch.Tensor,
                        img_encoder: Callable[[torch.Tensor], torch.Tensor],
                        txt_encoder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                        thresh: float = 0.3) -> bool:
        img_feat = img_encoder(image_tensor.unsqueeze(0))
        txt_feat = txt_encoder(text_ids.unsqueeze(0), text_mask.unsqueeze(0))
        sim = torch.nn.functional.cosine_similarity(img_feat, txt_feat).item()
        return sim > thresh
```

> **Additional recommended safeguards (from our expansion):** dataset retries with backoff, timeout wrappers, deterministic seeding, NaN/Inf and AMP overflow guards, class-imbalance sampling, atomic checkpoints, minimal telemetry, circuit breakers, and graceful degradation. See the **Expanded 11.3 toolkit** at the end of this chapter for paste‑ready snippets.

## 11.4 Optimization for Production

### Model Quantization

```python
import torch
import torch.nn as nn

class ModelQuantizer:
    """Quantize model for faster inference"""
    @staticmethod
    def quantize_int8(model):
        model.eval()
        return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    @staticmethod
    def quantize_with_calibration(model, calib_loader):
        model.eval(); model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        with torch.no_grad():
            for batch in calib_loader:
                _ = model(batch['image'], batch['text_ids'], batch['text_mask'])
        torch.quantization.convert(model, inplace=True)
        return model
```
### Knowledge Distillation

```python
import torch

class KnowledgeDistiller:
    def __init__(self, teacher_model, student_model, temperature=3.0):
        self.teacher, self.student, self.temperature = teacher_model, student_model, temperature
    def distillation_loss(self, student_logits, teacher_logits, labels, alpha=0.7):
        kd = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        ce = torch.nn.functional.cross_entropy(student_logits, labels)
        return alpha * kd + (1 - alpha) * ce
    def train_student(self, train_loader, optimizer, num_epochs):
        self.teacher.eval(); self.student.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                images, ids, mask, labels = batch['image'], batch['text_ids'], batch['text_mask'], batch['label']
                with torch.no_grad():
                    t_logits = self.teacher(images, ids, mask)
                s_logits = self.student(images, ids, mask)
                loss = self.distillation_loss(s_logits, t_logits, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
```
### Model Serving with TorchServe

```yaml
# config.yaml
model_store: ./model_store
ncs: false

# Handler (my_handler.py)
import torch
from transformers import AutoModel, AutoTokenizer

class MultimodalHandler:
    def __init__(self):
        self.image_model = AutoModel.from_pretrained('model_name')
        self.text_model = AutoModel.from_pretrained('model_name')
        self.tokenizer = AutoTokenizer.from_pretrained('model_name')
    def preprocess(self, data):
        image = data['image']; text = data['text']
        tokens = self.tokenizer(text, return_tensors='pt')
        return image, tokens
    def inference(self, image, tokens):
        img_feat = self.image_model(image)
        txt_feat = self.text_model(tokens['input_ids'])
        similarity = torch.cosine_similarity(img_feat, txt_feat)
        return similarity
    def postprocess(self, output):
        return {'similarity': float(output)}

# Deployment
# torch-model-archiver --model-name multimodal \
#   --version 1.0 \
#   --model-file model.py \
#   --serialized-file model.pt \
#   --handler my_handler.py \
#   --export-path model_store
# torchserve --start --model-store model_store --ncs --models multimodal=multimodal.mar
```

## 11.5 Monitoring and Maintenance

### Model Performance Monitoring

```python
import logging
from datetime import datetime
import numpy as np

class ModelMonitor:
    \"\"\"Monitor model performance in production\"\"\"\n    def __init__(self, log_file='model_performance.log'):
        self.log_file = log_file; self.setup_logging()
    def setup_logging(self):
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    def check_drift(self, current_batch, reference_data):
        current_mean = current_batch.mean(); reference_mean = reference_data.mean()
        drift_score = abs(current_mean - reference_mean) / (reference_data.std() + 1e-8)
        if drift_score > 3.0:
            logging.warning(f\"Data drift detected: {drift_score:.2f}\"); return True
        return False
    def log_prediction(self, input_id, prediction, confidence, latency):
        logging.info(str({
            'timestamp': datetime.now().isoformat(),
            'input_id': input_id,
            'prediction': prediction,
            'confidence': float(confidence),
            'latency_ms': latency
        }))
    def detect_anomalies(self, predictions, threshold=2.0):
        confidences = [p['confidence'] for p in predictions]
        mean_conf = np.mean(confidences); std_conf = np.std(confidences)
        return [i for i,p in enumerate(predictions) if abs(p['confidence']-mean_conf)/(std_conf+1e-6) > threshold]
```

### A/B Testing

```python
class ABTester:
    \"\"\"A/B testing for model updates\"\"\"\n    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a, self.model_b, self.split_ratio = model_a, model_b, split_ratio
        self.results = {'a': [], 'b': []}
    def predict(self, input_data, user_id=None):
        if user_id is not None:
            use_a = hash(user_id) % 100 < (self.split_ratio * 100)
        else:
            import numpy as np
            use_a = np.random.rand() < self.split_ratio
        if use_a:
            prediction = self.model_a(input_data); self.results['a'].append(prediction); return prediction, 'a'
        else:
            prediction = self.model_b(input_data); self.results['b'].append(prediction); return prediction, 'b'
    def get_statistics(self):
        import numpy as np
        from scipy import stats
        def stats_of(xs):
            accs = [r['accuracy'] for r in xs]
            return {'mean_accuracy': np.mean(accs), 'std_accuracy': np.std(accs), 'count': len(accs)}
        sa, sb = stats_of(self.results['a']), stats_of(self.results['b'])
        t, p = stats.ttest_ind([r['accuracy'] for r in self.results['a']], [r['accuracy'] for r in self.results['b']])
        return {'model_a': sa, 'model_b': sb, 't_statistic': t, 'p_value': p, 'winner': 'b' if sb['mean_accuracy'] > sa['mean_accuracy'] else 'a'}
```

## Key Takeaways

- **Preprocessing is critical** - garbage in, garbage out
- **Robust error handling** prevents cascading failures
- **Monitoring catches issues early** - drift, anomalies, degradation
- **Optimization techniques** make models production-ready
- **A/B testing validates improvements** before full rollout
- **MLOps practices** enable reliable systems

## Exercises

**⭐ Beginner:**
1. Build image preprocessing pipeline
2. Create text tokenization pipeline
3. Implement basic data validation

**⭐⭐ Intermediate:**
4. Build multimodal dataset loader
5. Implement training loop with early stopping
6. Add logging and monitoring

**⭐⭐⭐ Advanced:**
7. Implement model quantization
8. Set up knowledge distillation
9. Deploy model with monitoring

---

### Expanded 11.3 Toolkit (paste‑ready extras)

```python
# ---- Deterministic seeding & worker init ----
import random, numpy as np, torch

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    seed_everything(42 + worker_id)

# ---- Timeout wrapper for slow I/O ----
import signal, contextlib
class TimeoutError_(Exception): ...
@contextlib.contextmanager
def time_limit(seconds: int):
    def handler(signum, frame): raise TimeoutError_()
    signal.signal(signal.SIGALRM, handler); signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0)

# ---- Atomic checkpoints ----
import os, tempfile, shutil

def atomic_save_torch(state: dict, path: str):
    d = os.path.dirname(path); os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as tmp:
        torch.save(state, tmp.name)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_path = tmp.name
    shutil.move(tmp_path, path)

# ---- Class imbalance sampler ----
from torch.utils.data import WeightedRandomSampler

def make_weighted_sampler(labels):
    from collections import Counter
    c = Counter(labels); weights = [1.0 / c[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

# ---- Minimal telemetry ----
import json, time

def log_event(name: str, **kw):
    print(json.dumps({"t": time.time(), "event": name, **kw}))

# ---- Circuit breaker ----
class CircuitBreaker:
    def __init__(self, fail_thresh=5, cooldown=30):
        self.fail_thresh, self.cooldown = fail_thresh, cooldown
        self.fails, self.block_until = 0, 0
    def allow(self, now: float) -> bool:
        return now >= self.block_until
    def report(self, ok: bool, now: float):
        if ok: self.fails = 0
        else:
            self.fails += 1
            if self.fails >= self.fail_thresh:
                self.block_until = now + self.cooldown; self.fails = 0
```

---

## 11.2.7 End‑to‑End Minimal **Runnable** Demo (Synthetic Data)

> Use this to verify your environment end‑to‑end without any images or files. It trains one tiny epoch on synthetic inputs and exercises the full model + trainer. No wandb required.

```python
# 1) Build a tiny synthetic dataset that matches our model's expected keys
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticMultimodalDataset(Dataset):
    def __init__(self, n=64, H=224, W=224, T=77, num_classes=3):
        self.n, self.H, self.W, self.T, self.C = n, H, W, T, num_classes
    def __len__(self): return self.n
    def __getitem__(self, idx):
        image = torch.randn(3, self.H, self.W)
        text_ids = torch.randint(0, 30000, (self.T,))
        text_mask = torch.ones(self.T, dtype=torch.long)
        label = torch.randint(0, self.C, (1,)).item()
        return {"image": image, "text_ids": text_ids, "text_mask": text_mask, "label": torch.tensor(label)}

def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }

# 2) Instantiate model and trainer
model = MultimodalModel(num_classes=3, emb=128, fusion='xattn')  # try 'concat'|'gated'|'late'|'xattn'
trainer = MultimodalTrainer(model, device='cpu', use_wandb=False)

# 3) DataLoaders
train_ds = SyntheticMultimodalDataset(n=32, num_classes=3)
val_ds   = SyntheticMultimodalDataset(n=16, num_classes=3)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dl   = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

# 4) Run a short training schedule (1–2 epochs)
trainer.train(train_dl, val_dl, num_epochs=2, lr=5e-4, warmup_steps=10)
```
