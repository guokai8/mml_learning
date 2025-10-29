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
        """Load and preprocess image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Apply transforms
            if is_train:
                image = self.train_transforms(image)
            else:
                image = self.val_transforms(image)

            return image

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def preprocess_batch(self, image_paths, is_train=True):
        """Preprocess batch of images"""
        images = []
        valid_paths = []

        for path in image_paths:
            img = self.preprocess_image(path, is_train)
            if img is not None:
                images.append(img)
                valid_paths.append(path)

        if images:
            images = torch.stack(images)
            return images, valid_paths
        else:
            return None, []

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
        """Clean text"""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove special characters (keep basic punctuation)
        import re
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)

        # Lowercase
        text = text.lower()

        return text

    def tokenize(self, text):
        """Tokenize single text"""
        cleaned = self.clean_text(text)

        tokens = self.tokenizer(
            cleaned,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True
        )

        return tokens

    def tokenize_batch(self, texts):
        """Tokenize batch of texts"""
        cleaned = [self.clean_text(text) for text in texts]

        tokens = self.tokenizer(
            cleaned,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            batch_first=True
        )

        return tokens

# Example
text_proc = TextPreprocessor()
tokens = text_proc.tokenize_batch([
    "A red cat on a wooden chair",
    "Two dogs playing in the park"
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
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames evenly
            frame_indices = np.linspace(
                0, total_frames - 1,
                self.frame_count,
                dtype=int
            )

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    # Resize
                    frame = cv2.resize(
                        frame,
                        (self.frame_size, self.frame_size)
                    )
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            if frames:
                return np.stack(frames)  # (frame_count, h, w, 3)
            else:
                return None

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None

# Example
video_proc = VideoPreprocessor(frame_count=8)
frames = video_proc.extract_frames('video.mp4')
print(frames.shape)  # (8, 224, 224, 3)
```

### Complete Preprocessing Pipeline

```python
class MultimodalDataPreprocessor:
    """Complete preprocessing for image-text-video data"""

    def __init__(self, image_size=224, max_text_length=77,
                 video_frames=8):
        self.image_preprocessor = ImagePreprocessor(image_size)
        self.text_preprocessor = TextPreprocessor(max_text_length)
        self.video_preprocessor = VideoPreprocessor(frame_count=video_frames)

    def process_sample(self, sample):
        """Process single multimodal sample"""
        processed = {}

        # Image
        if 'image_path' in sample:
            img = self.image_preprocessor.preprocess_image(
                sample['image_path'],
                is_train=sample.get('is_train', True)
            )
            if img is not None:
                processed['image'] = img

        # Text
        if 'text' in sample:
            tokens = self.text_preprocessor.tokenize(sample['text'])
            processed['text_ids'] = tokens['input_ids'].squeeze()
            processed['text_mask'] = tokens['attention_mask'].squeeze()

        # Video
        if 'video_path' in sample:
            frames = self.video_preprocessor.extract_frames(
                sample['video_path']
            )
            if frames is not None:
                processed['video'] = torch.from_numpy(frames).float()

        # Label (if available)
        if 'label' in sample:
            processed['label'] = torch.tensor(sample['label'])

        return processed

    def validate_sample(self, sample):
        """Check if sample is valid"""
        required_keys = sample.get('required_modalities', ['image', 'text'])

        for key in required_keys:
            if key not in sample:
                return False

        return True

# Usage
preprocessor = MultimodalDataPreprocessor()

sample = {
    'image_path': 'cat.jpg',
    'text': 'A cute cat on a sofa',
    'label': 0,
    'is_train': True,
    'required_modalities': ['image', 'text']
}

if preprocessor.validate_sample(sample):
    processed = preprocessor.process_sample(sample)
    print(f"Image shape: {processed['image'].shape}")
    print(f"Text IDs shape: {processed['text_ids'].shape}")
```

## 11.2 Building Training Pipelines

### Data Loading with Multiprocessing

```python
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

class MultimodalDataset(Dataset):
    """Efficient multimodal dataset"""

    def __init__(self, samples, preprocessor, cache_size=1000):
        self.samples = samples
        self.preprocessor = preprocessor
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Load and preprocess
        sample = self.samples[idx]
        processed = self.preprocessor.process_sample(sample)

        # Cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = processed

        return processed

def create_dataloaders(train_samples, val_samples, batch_size=256,
                      num_workers=8):
    """Create train and validation dataloaders"""

    preprocessor = MultimodalDataPreprocessor()

    train_dataset = MultimodalDataset(train_samples, preprocessor)
    val_dataset = MultimodalDataset(val_samples, preprocessor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader

# Usage
train_loader, val_loader = create_dataloaders(
    train_samples=train_data,
    val_samples=val_data,
    batch_size=256,
    num_workers=8
)
```

### Training Loop with Best Practices

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

class MultimodalTrainer:
    """Production-ready trainer"""

    def __init__(self, model, device='cuda', use_wandb=True):
        self.model = model
        self.device = device
        self.use_wandb = use_wandb

        if use_wandb:
            wandb.init(project='multimodal-learning')

    def train_epoch(self, train_loader, optimizer, scheduler,
                   criterion, scaler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc='Training')

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            text_ids = batch['text_ids'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)

            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(images, text_ids, text_mask)
                    loss = criterion(logits, batch['label'].to(self.device))

                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                logits = self.model(images, text_ids, text_mask)
                loss = criterion(logits, batch['label'].to(self.device))
                loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # Optimization step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': total_loss / num_batches})

            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                })

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, val_loader, criterion):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        pbar = tqdm(val_loader, desc='Validating')

        for batch in pbar:
            images = batch['image'].to(self.device)
            text_ids = batch['text_ids'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(images, text_ids, text_mask)
            loss = criterion(logits, labels)

            # Accuracy
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'acc': total_acc / num_batches
            })

        return total_loss / num_batches, total_acc / num_batches

    def train(self, train_loader, val_loader, num_epochs=10,
             lr=1e-4, warmup_steps=1000):
        """Full training loop"""

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)

        # Scheduler with warmup
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps
        )

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler()

        # Loss
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")

            # Train
            train_loss = self.train_epoch(
                train_loader, optimizer, scheduler,
                criterion, scaler
            )

            # Validate
            val_loss, val_acc = self.evaluate(val_loader, criterion)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save checkpoint
                self.save_checkpoint(f'best_model_epoch{epoch}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config if hasattr(self.model, 'config') else None
        }, path)
        print(f"Saved checkpoint to {path}")

# Usage
model = MultimodalModel()
trainer = MultimodalTrainer(model)

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=30,
    lr=1e-4
)
```

## 11.3 Handling Edge Cases and Failures

### Error Handling in Data Loading

```python
class RobustDataLoader:
    """Data loader with error handling"""

    def __init__(self, dataset, batch_size=32, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.failed_indices = []

    def load_with_retry(self, idx, max_retries=3):
        """Load sample with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.dataset[idx]
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to load sample {idx} after {max_retries} attempts: {e}")
                    self.failed_indices.append(idx)
                    return None

    def get_valid_batch(self, indices):
        """Get batch skipping failed samples"""
        batch = []
        valid_indices = []

        for idx in indices:
            sample = self.load_with_retry(idx)
            if sample is not None:
                batch.append(sample)
                valid_indices.append(idx)

        if not batch:
            return None, []

        # Stack samples
        try:
            stacked = {}
            for key in batch[0].keys():
                stacked[key] = torch.stack([s[key] for s in batch])
            return stacked, valid_indices
        except Exception as e:
            print(f"Error stacking batch: {e}")
            return None, []

# Usage
robust_loader = RobustDataLoader(dataset, batch_size=32)
```

### Validation and Sanity Checks

```python
class DataValidator:
    """Validate data quality"""

    @staticmethod
    def check_image_quality(image_tensor, min_entropy=0.5):
        """Check if image has meaningful content"""
        # Calculate entropy
        import torch.nn.functional as F

        # Flatten and normalize to [0, 1]
        flat = image_tensor.flatten()
        flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)

        # Histogram-based entropy
        hist = torch.histc(flat, bins=256)
        hist = hist / hist.sum()
        entropy = -(hist * torch.log(hist + 1e-8)).sum()

        return entropy > min_entropy

    @staticmethod
    def check_text_quality(text, min_length=5, max_length=1000):
        """Check if text is valid"""
        if text is None or not isinstance(text, str):
            return False

        text = text.strip()

        if len(text) < min_length or len(text) > max_length:
            return False

        # Check for too many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c != ' ')
        if special_chars / len(text) > 0.5:
            return False

        return True

    @staticmethod
    def check_alignment(image_tensor, text, similarity_fn):
        """Check if image and text are aligned"""
        # Encode both
        img_feat = image_encoder(image_tensor.unsqueeze(0))
        txt_feat = text_encoder(text)

        # Compute similarity
        sim = similarity_fn(img_feat, txt_feat)

        # Threshold (depends on model)
        return sim > 0.3

# Usage
validator = DataValidator()

# Check a sample
if validator.check_image_quality(image) and \
   validator.check_text_quality(text) and \
   validator.check_alignment(image, text, similarity_fn):
    print("Sample is valid!")
```

## 11.4 Optimization for Production

### Model Quantization

```python
class ModelQuantizer:
    """Quantize model for faster inference"""

    @staticmethod
    def quantize_int8(model, sample_input):
        """Convert to INT8 quantization"""
        model.eval()

        # Dynamic quantization (easiest)
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        return quantized

    @staticmethod
    def quantize_with_calibration(model, calibration_loader):
        """Quantization with calibration data"""
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Insert observers
        torch.quantization.prepare(model, inplace=True)

        # Calibrate with sample data
        with torch.no_grad():
            for batch in calibration_loader:
                _ = model(batch)

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

        return model

# Usage
quantizer = ModelQuantizer()

# Simple quantization
q_model = quantizer.quantize_int8(model, sample_input)

# Memory savings
print(f"Original model size: {get_model_size(model):.2f} MB")
print(f"Quantized model size: {get_model_size(q_model):.2f} MB")
```

### Knowledge Distillation

```python
class KnowledgeDistiller:
    """Distill large model to small student"""

    def __init__(self, teacher_model, student_model, temperature=3.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature

    def distillation_loss(self, student_logits, teacher_logits, labels,
                         alpha=0.7):
        """Combined distillation + task loss"""
        # KL divergence for distillation
        kd_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(
                student_logits / self.temperature,
                dim=1
            ),
            torch.nn.functional.softmax(
                teacher_logits / self.temperature,
                dim=1
            ),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Task loss
        task_loss = torch.nn.functional.cross_entropy(
            student_logits,
            labels
        )

        # Combined
        return alpha * kd_loss + (1 - alpha) * task_loss

    def train_student(self, train_loader, optimizer, num_epochs):
        """Train student model"""
        self.teacher.eval()

        for epoch in range(num_epochs):
            total_loss = 0

            for batch in train_loader:
                # Teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_logits = self.teacher(batch)

                # Student predictions
                student_logits = self.student(batch)

                # Loss
                loss = self.distillation_loss(
                    student_logits,
                    teacher_logits,
                    batch['label']
                )

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
        image = data['image']
        text = data['text']

        tokens = self.tokenizer(text, return_tensors='pt')

        return image, tokens

    def inference(self, image, tokens):
        img_feat = self.image_model(image)
        txt_feat = self.text_model(tokens['input_ids'])

        # Compute similarity
        similarity = torch.cosine_similarity(img_feat, txt_feat)

        return similarity

    def postprocess(self, output):
        return {'similarity': float(output)}

# Deployment
# torch-model-archiver --model-name multimodal \
#     --version 1.0 \
#     --model-file model.py \
#     --serialized-file model.pt \
#     --handler my_handler.py \
#     --export-path model_store
#
# torchserve --start --model-store model_store \
#     --ncs --models multimodal=multimodal.mar
```

## 11.5 Monitoring and Maintenance

### Model Performance Monitoring

```python
import logging
from datetime import datetime

class ModelMonitor:
    """Monitor model performance in production"""

    def __init__(self, log_file='model_performance.log'):
        self.log_file = log_file
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def check_drift(self, current_batch, reference_data):
        """Check for data drift"""
        # Compare statistics
        current_mean = current_batch.mean()
        reference_mean = reference_data.mean()

        # Z-test
        drift_score = abs(current_mean - reference_mean) / reference_data.std()

        if drift_score > 3.0:  # Threshold
            logging.warning(f"Data drift detected: {drift_score:.2f}")
            return True

        return False

    def log_prediction(self, input_id, prediction, confidence, latency):
        """Log prediction for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_id': input_id,
            'prediction': prediction,
            'confidence': float(confidence),
            'latency_ms': latency
        }

        logging.info(str(log_entry))

    def detect_anomalies(self, predictions, threshold=2.0):
        """Detect anomalous predictions"""
        confidences = [p['confidence'] for p in predictions]
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)

        anomalies = []
        for i, pred in enumerate(predictions):
            z_score = abs(pred['confidence'] - mean_conf) / (std_conf + 1e-6)
            if z_score > threshold:
                anomalies.append(i)

        return anomalies

# Usage
monitor = ModelMonitor()

# During inference
for batch in inference_batches:
    predictions = model(batch)

    for i, pred in enumerate(predictions):
        monitor.log_prediction(
            input_id=batch['id'][i],
            prediction=pred['class'],
            confidence=pred['confidence'],
            latency=pred['latency_ms']
        )

    # Check for issues
    if monitor.check_drift(batch, reference_batch):
        print("Model may need retraining!")

    anomalies = monitor.detect_anomalies(predictions)
    if anomalies:
        print(f"Anomalous predictions at indices: {anomalies}")
```

### A/B Testing

```python
class ABTester:
    """A/B testing for model updates"""

    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.results = {'a': [], 'b': []}

    def predict(self, input_data, user_id=None):
        """Route to model A or B"""
        # Consistent routing per user
        if user_id is not None:
            use_a = hash(user_id) % 100 < (self.split_ratio * 100)
        else:
            use_a = np.random.rand() < self.split_ratio

        if use_a:
            prediction = self.model_a(input_data)
            self.results['a'].append(prediction)
            return prediction, 'a'
        else:
            prediction = self.model_b(input_data)
            self.results['b'].append(prediction)
            return prediction, 'b'

    def get_statistics(self):
        """Compare model performance"""
        def compute_stats(results):
            accs = [r['accuracy'] for r in results]
            return {
                'mean_accuracy': np.mean(accs),
                'std_accuracy': np.std(accs),
                'count': len(accs)
            }

        stats_a = compute_stats(self.results['a'])
        stats_b = compute_stats(self.results['b'])

        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            [r['accuracy'] for r in self.results['a']],
            [r['accuracy'] for r in self.results['b']]
        )

        return {
            'model_a': stats_a,
            'model_b': stats_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'winner': 'b' if stats_b['mean_accuracy'] > stats_a['mean_accuracy'] else 'a'
        }

# Usage
ab_tester = ABTester(model_v1, model_v2, split_ratio=0.5)

# In production
for request in requests:
    prediction, model_used = ab_tester.predict(request, user_id=request['user_id'])

# After collecting data
stats = ab_tester.get_statistics()
print(f"Winner: Model {stats['winner']}")
print(f"P-value: {stats['p_value']}")
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

