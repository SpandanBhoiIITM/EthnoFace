# EthnoFace: Ethnicity Classification using Deep Learning

This project trains a ResNet-based CNN on the [UTKFace Dataset](https://susanqq.github.io/UTKFace/) to classify faces into 5 ethnic categories:
- White
- Black
- Asian
- Indian
- Others

## 🚀 Features
- Trains on UTKFace dataset with class balancing
- Achieves ~82% validation accuracy
- GPU acceleration (CUDA support)
- Supports inference on new images
- Easy-to-use scripts: `train.py`, `predict.py`, `evaluate.py`

## 📂 Project Structure
- `data_loader.py` – custom PyTorch Dataset
- `model.py` – ResNet-based classifier
- `train.py` – training with weighted loss & validation tracking
- `predict.py` – inference on single image
- `models/` – saved model checkpoints
- `requirements.txt` – dependencies

## ⚡ Usage
### Install dependencies
```bash
pip install -r requirements.txt
