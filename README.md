# 🐾 Class-Distinction — Image Classification with YOLOv8

An image classification project trained with a YOLOv8 classification model,
testable through a Streamlit-based interface, and optionally runnable as a
desktop application.

The dataset comes from [Roboflow Universe](https://universe.roboflow.com/ali-kaan/class-ayrim).

## 📁 Project Structure

```
class-distinction/
├── main.py                # Used to train the model (YOLOv8-cls)
├── test.py                # Streamlit web interface (upload an image, get a prediction)
├── app.py                 # Wrapper that opens test.py in a desktop window (pywebview)
├── requirements.txt       # Required Python packages
├── docs/
│   ├── DATASET.md         # Dataset source and license
│   └── ROBOFLOW_EXPORT.md # Roboflow export details
└── .gitignore
```

## ⚙️ Setup

```bash
git clone https://github.com/kaanklft/class-distinction.git
cd class-distinction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Dataset

This repo does not include the dataset (due to its size). To train the model:

1. Download the dataset from [Roboflow Universe - class-ayrim](https://universe.roboflow.com/ali-kaan/class-ayrim) (export in folder format).
2. Place the downloaded class folders (e.g. `animals/cat`, `animals/dog`, ...) in the project root.

See `docs/DATASET.md` and `docs/ROBOFLOW_EXPORT.md` for details.

## 🏋️ Training the Model

```bash
python main.py
```

`main.py` uses the pretrained `yolov8n-cls.pt` weights as a starting point
(automatically downloaded by Ultralytics on first run) and saves training
results under `runs/classify/`.

> **Note:** the `device=0` parameter is for GPU usage. If you don't have a GPU,
> remove this line or change it to `device='cpu'`.

## 🖥️ Testing with the Web Interface

Once training finishes, the best weights file is created under
`runs/classify/train*/weights/best.pt`. Update the `model_path` variable in
`test.py` to match your own folder name, then run:

```bash
streamlit run test.py
```

## 🖱️ Running as a Desktop App

```bash
python app.py
```

This starts the Streamlit server in the background and opens it in a desktop
window using `pywebview`.

## 📄 License

Code: MIT License (feel free to change this).
Dataset: CC BY 4.0 — see `docs/DATASET.md`.
