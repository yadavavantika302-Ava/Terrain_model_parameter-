# Terrain Lens

Terrain Lens is a FastAPI image-classification testing workspace. It keeps the bundled terrain TFLite model as the example and supports uploaded `.tflite`, `.keras`, `.h5`, and `.onnx` models when their runtime is installed. Uploaded Python files are never accepted or executed. Keras loading uses TensorFlow safe mode.

## Install and run

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional runtimes, depending on model files:
# pip install tensorflow   # Keras/H5 and TFLite fallback
# pip install onnxruntime  # ONNX
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. Model files are limited to 20 MB, images to 8 MB and 20 megapixels. The service runs models on CPU. Restart the app to clear uploaded models.

## Model setup

The bundled TFLite artifact reports a `1×224×224×3` uint8 input with quantization scale 1 and zero point 0, and a six-value uint8 output with scale 1/256 and zero point 0. The site selects and enforces raw 0–255 pixels for that input quantization; 0–1 would round almost every input pixel to 0 or 1. The TFLite graph ends in softmax followed by quantization, so the site does not apply another softmax. RGB channel order, 224×224 bilinear stretch, and the raw-pixel training convention are compatible with tensor metadata but cannot all be confirmed as the original training preprocessing because training code and preprocessing metadata are not included. The class names in `backend/model.py` are configured output labels; the training-time class-to-index mapping is not included and therefore is not verified. For other uploaded models, options are inferred only where the tensor metadata supports it; otherwise check the model card or training code and configure preprocessing, output interpretation, and labels accordingly. Auto interpretation accepts probability-like outputs only when values are nonnegative, at most one, and sum to approximately one. Raw scores are not probabilities, and probability outputs are not calibrated confidence.

## Analysis

A prediction runs image measurements, one-property-at-a-time interventions, and occlusion sensitivity against the selected model. These are perturbation-based sensitivity measurements; they do not prove the model understands terrain. Altered images may differ from the model's training distribution. Single-image explanations do not replace dataset evaluation. The configurable low-margin and low-top-score warnings are user-set heuristics; validate them on labeled validation data before using them operationally.

`backend/dataset_audit.py` evaluates labeled folders named after the configured classes, using the same website model adapter and default preprocessing. It reports accuracy, per-class precision/recall/F1, confusion matrix, and perturbation summaries. No labeled validation dataset is included in this repository, so no accuracy or threshold calibration claim is available by default.

## API

- `GET /api/info` reports loaded models and metadata.
- `POST /api/models` accepts a model and optional newline-separated class labels.
- `POST /api/predict` accepts an image, model ID, labels, preprocessing settings and output interpretation.
- `GET /api/history` returns recent in-memory predictions.

Supported image input is RGB-like 4D image tensors (`NHWC` or `NCHW`) and single-output classification tensors shaped `[batch, classes]`. Dynamic input dimensions and non-classification outputs are rejected with an error. Runtime installation and operator compatibility vary by platform.

