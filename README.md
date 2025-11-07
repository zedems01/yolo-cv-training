# Training YOLO models

Setup and training scripts for YOLO-based computer vision models

## Setup

1. Install dependencies:
```bash
python -m venv .venv
.venv\Script\activate
# if running locally with an NVIDIA GPU, consider installing PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# using uv
uv venv .venv
.venv\Script\activate
uv init --bare
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
```
2. For training: Run `python scripts/main.py --dataset <kaggle-handle> --nc <num-classes> --names <class-names>`
3. For inference: Run `python scripts/inference.py --model <model-path> --input <image/video/webcam>`
