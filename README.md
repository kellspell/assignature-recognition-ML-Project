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
- [Part 2 — Production Pipeline](#part-2--production-pipeline)
  - [Project Structure](#project-structure)
  - [How to Run](#how-to-run)
  - [Constants](#constants)
  - [Artifacts](#artifacts)
  - [Data Injection](#data-injection)
  - [Data Transformation](#data-transformation)
  - [Model Trainer](#model-trainer)
  - [Training Pipeline](#training-pipeline)
  - [Logger](#logger)
  - [Exception Handler](#exception-handler)
  - [Utilities](#utilities)

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

---

## Part 2 — Production Pipeline

**Entry point:** `main.py`

This part wraps the trained model in a modular, production-style pipeline. Each concern (data ingestion, configuration, logging, error handling) lives in its own module under `src/`.

---

### Project Structure

```
src/
├── artifacts/
│   ├── config_entry.py         # config dataclasses for all pipeline stages
│   └── artifects_entry.py      # output artifact dataclasses for all stages
├── components/
│   ├── data_injection.py       # downloads, unzips, and cleans up the dataset
│   ├── data_transformation.py  # applies transforms and splits dataset into train/valid/test
│   └── model_trainer.py        # fine-tunes ResNet-34 and saves the trained model
├── configurations/
│   └── gcloud_syncer.py        # gsutil wrapper for GCloud Storage transfers
├── constants/
│   └── __init__.py             # shared constants (paths, device, timestamp)
├── exceptions/
│   └── __init__.py             # CustomException with file/line enrichment
├── logger/
│   └── __init__.py             # file + stream logging setup
├── pipeline/
│   └── training.py             # TrainingPipeline — orchestrates all stages
└── utils/
    └── main_utils.py           # save/load objects, YAML reader, base64 helper
```

---

### How to Run

```bash
python main.py
```

Prerequisites:
- Conda environment activated (`tf`)
- Google Cloud SDK installed and authenticated (`gcloud auth login`)
- `config/config.yaml` present with `data_injection_config.bucket_name` and `data_injection_config.zip_file_name`

---

### Constants

**File:** `src/constants/__init__.py`

| Constant | Description |
|---|---|
| `CONFIG_PATH` | Path to `config/config.yaml` |
| `TIMESTAMP` | Run timestamp used to namespace artifact directories |
| `ARTIFACTS_DIR` | `artifacts/<timestamp>/` — root for all pipeline outputs |
| `DEVICE` | `torch.device` — CUDA if available, otherwise CPU |
| `DATA_INJECTION_ARTIFACTS_DIR` | Subfolder name for data injection outputs |
| `DATA_TRANSFORMATION_ARTIFACTS_DIR` | Subfolder name for data transformation outputs |
| `DATA_TRANSFORMATION_TRAIN_FILE_NAME` | Filename for the serialized train dataset (`train_transformed.pkl`) |
| `DATA_TRANSFORMATION_VALID_FILE_NAME` | Filename for the serialized validation dataset (`valid_transformed.pkl`) |
| `DATA_TRANSFORMATION_TEST_FILE_NAME` | Filename for the serialized test dataset (`test_transformed.pkl`) |
| `MODEL_TRAINER_ARTIFACTS_DIR` | Subfolder name for model trainer outputs |
| `TRAINED_MODEL_PATH` | Filename for the saved model (`model.pt`) |

---

### Artifacts

**Files:** `src/artifacts/config_entry.py`, `src/artifacts/artifects_entry.py`

Configuration and output for each pipeline stage are typed dataclasses.

**`DataInjectionConfig`** — built from `config/config.yaml`:

| Field | Description |
|---|---|
| `BUCKET_NAME` | GCloud Storage bucket name |
| `ZIP_FILE_NAME` | Name of the zip file in the bucket |
| `DATA_INJECTION_ARTIFACTS_DIR` | Local directory where the dataset is extracted |
| `ZIP_FILE_PATH` | Full local path to the downloaded zip file |

**`DataInjectionArtifacts`** — returned after injection completes:

| Field | Description |
|---|---|
| `dataset_path` | Path to the extracted dataset directory |

**`DataTransformationConfig`** — built from `config/config.yaml`:

| Field | Description |
|---|---|
| `MEAN` / `STD` | ImageNet channel means and stds for normalization |
| `IMG_SIZE` | Target image size (both dimensions) |
| `DEGREE_N` / `DEGREE_P` | Negative/positive rotation range for augmentation |
| `TRAIN_RATIO` / `VALID_RATIO` | Fractions used to split the dataset |
| `DATA_TRANSFORMATION_ARTIFACTS_DIR` | Directory where transformed splits are saved |
| `TRAIN/VALID/TEST_TRANSFORMATION_OBJECT_FILE_PATH` | Full paths to the serialized split `.pkl` files |

**`DataTransformationArtifacts`** — returned after transformation completes:

| Field | Description |
|---|---|
| `train_transformed_object` | Path to the serialized train split |
| `valid_transformed_object` | Path to the serialized validation split |
| `test_transformed_object` | Path to the serialized test split |
| `classes` | Number of classes detected from the dataset folder |

**`ModelTrainerConfig`** — built from `config/config.yaml`:

| Field | Description |
|---|---|
| `LR` | Learning rate |
| `EPOCHS` | Number of training epochs |
| `BATCH_SIZE` | Batch size for train and validation loaders |
| `NUM_WORKERS` | DataLoader worker count |
| `MODEL_TRAINER_ARTIFACTS_DIR` | Directory where the trained model is saved |
| `TRAINED_MODEL_PATH` | Full path to `model.pt` |

**`ModelTrainerArtifacts`** — returned after training completes:

| Field | Description |
|---|---|
| `trained_model_path` | Path to the saved `model.pt` file |

---

### Data Injection

**File:** `src/components/data_injection.py`

`DataInjection` handles three steps in sequence:

1. **Download** — calls `GCloudSync.sync_file_from_gcloud` to pull `dataset.zip` from GCloud Storage into the artifacts directory via `gsutil cp`
2. **Unzip** — extracts the zip into the same artifacts directory using `ZipFile`
3. **Cleanup** — deletes the zip file after extraction

Returns a `DataInjectionArtifacts` instance pointing to the extracted dataset directory.

**GCloud sync** uses `subprocess.run(check=True)` with the full resolved path to `gsutil` (resolved via `shutil.which` with fallback to `~/google-cloud-sdk/bin/gsutil`) to avoid PATH issues in subprocess environments.

---

### Data Transformation

**File:** `src/components/data_transformation.py`

`DataTransformation` prepares the dataset for training in three steps:

1. **Transform** — builds a `transforms.Compose` pipeline (resize → random rotation → to tensor → normalize with ImageNet stats)
2. **Split** — uses `torch.utils.data.random_split` to divide the full dataset into train, validation, and test subsets using the configured ratios
3. **Save** — serializes each split to disk as a `.pkl` file using `dill`

Returns a `DataTransformationArtifacts` instance with paths to the three serialized splits and the number of classes.

---

### Model Trainer

**File:** `src/components/model_trainer.py`

`ModelTrainer` fine-tunes a pretrained ResNet-34 for signature classification:

1. **Load** — deserializes the train and validation splits from disk and wraps them in `DataLoader`s
2. **Build model** — loads `ResNet34_Weights.DEFAULT` and replaces the final FC layer with `Dropout(0.1) → Linear(512, num_classes)`
3. **Train** — runs SGD with momentum over the configured number of epochs; logs train and validation loss per epoch
4. **Save** — persists the full model object to `model.pt` using `torch.save`

Training loop per epoch:

```
forward → CrossEntropyLoss → backward → optimizer.step   (train)
no_grad forward → CrossEntropyLoss                        (validation)
```

Loss is averaged over the number of batches (not samples) and printed each epoch:
```
Epoch 1 / 5
Train loss: 0.6821  Test loss: 0.5934
```

Returns a `ModelTrainerArtifacts` instance with the path to the saved model.

---

### Training Pipeline

**File:** `src/pipeline/training.py`

`TrainingPipeline` is the top-level orchestrator. `run_pipeline()` calls each stage in order and stores its artifact on `self`:

```python
tracking_pipeline = TrainingPipeline()
tracking_pipeline.run_pipeline()
```

Stages completed so far:

| Stage | Method | Status |
|---|---|---|
| Data Injection | `start_data_injection()` | Done |
| Data Transformation | `start_data_transformation()` | Done |
| Model Trainer | `start_model_trainer()` | Done |

---

### Logger

**File:** `src/logger/__init__.py`

Writes structured log lines to both a timestamped file and stdout:

```
[2026-05-14 23:15:21,115] root - INFO - <message>
```

Log files are written to `logs/<timestamp>.log`.

---

### Exception Handler

**File:** `src/exceptions/__init__.py`

`CustomException` enriches every exception with the source file name and line number:

```
Error occurred python script name [<file>] line number [<line>] error message [<msg>]
```

Usage throughout the codebase:
```python
except Exception as e:
    raise CustomException(e, sys) from e
```

---

### Utilities

**File:** `src/utils/main_utils.py`

| Function | Description |
|---|---|
| `save_objects(file_path, obj)` | Serializes any Python object to disk using `dill` |
| `load_objects(file_path)` | Deserializes a `dill`-pickled object from disk |
| `read_yaml_file(file_path)` | Reads a YAML file and returns it as a dict |
| `image_to_base64(image)` | Encodes an image file to base64 |
