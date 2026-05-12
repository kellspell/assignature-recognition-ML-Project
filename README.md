# Signature Recognition — ML Project

A binary image classifier that distinguishes **forged** from **original** handwritten signatures using transfer learning on ResNet-34.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Part 1 — Model Training Notebook](#part-1--model-training-notebook)
  - [Dataset](#dataset)
  - [Preprocessing & Augmentation](#preprocessing--augmentation)
  - [Data Splitting](#data-splitting)
  - [DataLoaders](#dataloaders)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Saving & Loading the Model](#saving--loading-the-model)

---

## Project Overview

| Property | Value |
|---|---|
| Task | Binary classification (Forged / Original) |
| Model | ResNet-34 (pretrained on ImageNet, fine-tuned) |
| Framework | PyTorch + torchvision |
| Input | Grayscale signature images resized to 224×224 |
| Output | Class label — `0: Forged`, `1: Original` |
| Saved model | `saved-model/model.pt` |

---

## Part 1 — Model Training Notebook

**File:** `notebooks/Signature-Recognition.ipynb`

### Dataset

Located at `notebooks/dataset/`. Organized into two sub-folders that `ImageFolder` uses as class labels:

```
dataset/
├── Forged/       # forged signature images
└── Original/     # genuine signature images
```

| Split | Count |
|---|---|
| Total | 842 |
| Train (60%) | 505 |
| Validation (30%) | 252 |
| Test (10%) | 85 |

Classes are loaded via `torchvision.datasets.ImageFolder`, which maps folder names to integer indices:

```
['Forged', 'Original']  →  [0, 1]
```

---

### Preprocessing & Augmentation

A single `transforms.Compose` pipeline is applied to every split:

```python
T.Compose([
    T.Resize((224, 224)),               # standardize spatial dimensions for ResNet
    T.RandomRotation(degrees=(-20, 20)),# light augmentation — signatures vary in tilt
    T.ToTensor(),                        # scale pixels to [0, 1]
    T.Normalize(
        mean=[0.485, 0.456, 0.406],     # ImageNet channel means
        std =[0.229, 0.224, 0.225]      # ImageNet channel stds
    ),
])
```

ImageNet statistics are reused because the backbone was pretrained on ImageNet — keeping the same normalization preserves the feature distribution the backbone learned.

---

### Data Splitting

`torch.utils.data.random_split` splits the full dataset at a fixed ratio with no manual seed, so each run produces a different random partition:

```python
train_count = int(0.6 * total_count)   # 505
valid_count = int(0.3 * total_count)   # 252
test_count  = total_count - train_count - valid_count  # 85
```

---

### DataLoaders

```python
trainloader      = DataLoader(train_data, batch_size=32, shuffle=True)
validationloader = DataLoader(val_data,   batch_size=32, shuffle=True)
testloader       = DataLoader(test_data,  batch_size=32, shuffle=False)
```

| Loader | Batches | Samples |
|---|---|---|
| Train | 16 | 505 |
| Validation | 8 | 252 |
| Test | 3 | 85 |

---

### Model Architecture

ResNet-34 with a custom classification head replacing the original 1000-class fully-connected layer:

```
ResNet-34 (pretrained backbone)
    └── avgpool: AdaptiveAvgPool2d → (1, 1)
    └── fc (replaced):
            Dropout(p=0.1)
            Linear(in_features=512, out_features=2)
```

The backbone feature extractor is **not frozen** — all layers are updated during fine-tuning.

---

### Training

| Hyperparameter | Value |
|---|---|
| Loss function | `CrossEntropyLoss` |
| Optimizer | `SGD` |
| Learning rate | `1e-3` |
| Momentum | `0.9` |
| Epochs | `5` |
| Batch size | `32` |
| Device | CPU (CUDA not available in this run) |

Training loop summary — loss decreases steadily across epochs:

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.0231 | 0.0205 |
| 2 | 0.0189 | 0.0201 |
| 3 | 0.0160 | 0.0159 |
| 4 | 0.0132 | 0.0153 |
| 5 | 0.0103 | 0.0099 |

Total training time on CPU: ~27 minutes.

The `train_model` function runs one full epoch: forward pass → loss → backward pass → optimizer step over the training batches, then a no-grad pass over the validation batches to compute validation loss.

---

### Evaluation

After training, the model is evaluated on the held-out test set:

```
Test Loss: 0.0032
```

The `evaluation` function mirrors the validation pass in `train_model` but runs over `testloader` and does not update weights.

---

### Inference

Single-image prediction pipeline:

```python
image = Image.open("path/to/signature.png")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),    # ensure 3-channel input even for grayscale images
    transforms.ToTensor()
])

image = preprocess(image)[:3]           # drop alpha channel if present
image = image.unsqueeze(0).to(device)   # add batch dimension

with torch.no_grad():
    logits    = model(image)
    probs     = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(probs, dim=1)

predicted_class = data.classes[pred_label.item()]  # 'Forged' or 'Original'
```

Example output for `dataset/Original/00101001.png`:

```
Predicted Label: 1
Predicted Label Names: Original
```

---

### Saving & Loading the Model

The full model object (architecture + weights) is persisted with `torch.save`:

```python
# Save
torch.save(model, 'saved-model/model.pt')

# Load
model = torch.load('saved-model/model.pt', weights_only=False)
model.eval()
```

`weights_only=False` is required because the file contains the full model object, not just a state dict.

---

*Documentation is being added incrementally. More sections will cover the rest of the stack as it is built.*
