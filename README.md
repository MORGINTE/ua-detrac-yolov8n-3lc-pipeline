# UA-DETRAC Multi-Vehicle Detection Pipeline (YOLOv8n + 3LC)

End-to-end, configuration-driven object detection pipeline built for the **UA-DETRAC** multi-vehicle detection benchmark using **YOLOv8n** (trained from scratch) and **3LC** for data-centric table management and experiment tracking.

## Key Engineering Highlights
- **Config-Driven Architecture:** All dataset paths, 3LC table metadata, training hyperparameters, and inference settings are centralized in [`config.yaml`](config.yaml).
- **Data-Centric Workflow (3LC):** Idempotent dataset registration (`register_tables.py`) with 3LC tables for `train` and `val` splits, plus UMAP image embedding reduction.
- **Reproducible From-Scratch Training:** Deterministic seed initialization and enforced `yolov8n.yaml` training without pretrained checkpoints (`train.py`), including an Ultralytics 8.3+ / 3LC validator compatibility shim.
- **Memory-Efficient Inference:** Chunked GPU inference pipeline (`predict.py`) designed to avoid CUDA OOM on 8 GB GPUs and generate Kaggle-ready `submission.csv`.
- **Pre-Flight Environment Verification:** Automated check (`verify_setup.py`) for Python version, CUDA/PyTorch, required packages, and dataset layout.

## Target Classes
Defined in [`dataset.yaml`](dataset.yaml):
- `0`: `truck`
- `1`: `car`
- `2`: `van`
- `3`: `bus`

---

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note:** For GPU training, install CUDA-enabled `torch` and `torchvision` before `3lc-ultralytics`.

### 2. Prepare Dataset Directory
Place the UA-DETRAC dataset under `data/` (excluded from Git via `.gitignore`):
```text
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    └── images/
```

### 3. Run the Pipeline
```bash
python verify_setup.py      # Verify Python, CUDA, packages, and data paths
3lc login YOUR_API_KEY      # One-time 3LC authentication
python register_tables.py   # Register train & val splits as 3LC tables
python train.py             # Train YOLOv8n from scratch (yolov8n.yaml)
python predict.py           # Run test inference and generate submission.csv
```

---

## Repository Structure
```text
├── config.yaml             # Centralized paths, 3LC config, and hyperparameters
├── dataset.yaml            # YOLO dataset & class definitions
├── verify_setup.py         # Pre-flight environment & dependency checker
├── register_tables.py      # Idempotent 3LC table registration script
├── train.py                # YOLOv8n from-scratch training with 3LC integration
├── predict.py              # Chunked inference & Kaggle submission generator
├── sample_submission.csv   # Official submission format reference
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes datasets, weights, and local build artifacts
└── README.md
```
