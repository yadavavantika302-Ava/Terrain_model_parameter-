import os, time, uuid, json, math
from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import analysis as A
from model import CLASSES, MODEL_PATH, ImageModel, decode_image

MAX_BYTES, MAX_PIXELS, MAX_MODEL_BYTES = 8 * 1024 * 1024, 20_000_000, 20 * 1024 * 1024
app = FastAPI(title="Terrain Lens")
frontend_origins = [origin.strip() for origin in os.getenv(
    "FRONTEND_ORIGINS",
    "https://terrainmodelparameter-frontend.vercel.app,https://terrainmodelparameter-frontend-izngcraf7-myth-b240.vercel.app,http://localhost:8000,http://127.0.0.1:8000",
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_origin_regex=r"https://terrainmodelparameter-frontend(?:-[a-z0-9]+-myth-b240)?\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)
history = []
UPLOAD_DIR = Path(__file__).parent / "uploaded_models"
UPLOAD_DIR.mkdir(exist_ok=True)
models = {}
default_model = None
try:
    default_model = ImageModel(MODEL_PATH, "tflite", CLASSES)
    models["terrain-default"] = default_model
except Exception:
    pass


def information(m, key, default=False):
    h, w = m.image_size()
    input_quantization = m.metadata.get("input_quantization", ())
    required_scaling = ("raw" if m.input_dtype == "uint8" and len(input_quantization) == 2
                        and float(input_quantization[0]) == 1.0 and int(input_quantization[1]) == 0 else None)
    required_interpretation = "probabilities" if m.metadata.get("output_activation") == "softmax" else None
    return {"id": key, "format": m.format, "input_shape": m.input_shape, "input_dtype": m.input_dtype,
            "output_dtype": m.output_dtype,
            "output_shape": m.output_shape, "class_count": m.class_count, "classes": m.classes,
            "metadata": m.metadata, "known_preprocessing": False,
            "recommended_scaling": required_scaling,
            "recommended_interpretation": required_interpretation,
            "preprocessing_source": "Not present in the model artifact or repository training files",
            "label_order_source": "backend/model.py configured labels" if default else "uploaded labels or generic class names",
            "label_order_verified": False}


@app.get("/api/info")
def info():
    return {"classes": default_model.classes if default_model else [],
            "model": information(default_model, "terrain-default", True) if default_model else None,
            "models": [information(m, k, k == "terrain-default") for k, m in models.items()],
            "hints": A.MEASURE_HINT, "group_names": dict(A.GROUPS)}


@app.post("/api/models")
async def upload_model(file: UploadFile = File(...), labels: str = Form("")):
    ext = Path(file.filename or "").suffix.lower()
    formats = {".tflite": "tflite", ".keras": "keras", ".h5": "h5", ".onnx": "onnx"}
    if ext not in formats:
        raise HTTPException(400, "Unsupported model extension. Upload .tflite, .keras, .h5, or .onnx.")
    data = await file.read(MAX_MODEL_BYTES + 1)
    if len(data) > MAX_MODEL_BYTES:
        raise HTTPException(413, "Model file exceeds the 20 MB limit.")
    key = uuid.uuid4().hex
    path = UPLOAD_DIR / (key + ext)
    try:
        path.write_bytes(data)
        label_list = [x.strip() for x in labels.splitlines() if x.strip()] or None
        m = await run_in_threadpool(ImageModel, str(path), formats[ext], label_list)
        m.image_size()
        if len(m.input_shape) != 4 or m.input_shape[-1] not in (1, 3, 4) and m.input_shape[1] not in (1, 3, 4):
            raise ValueError("Input tensor is not a supported one-image RGB/RGBA/grayscale tensor.")
        if m.output_shape[0] not in (1, -1):
            raise ValueError("Model must accept a single image per inference.")
        if len(models) >= 4:
            raise ValueError("At most three uploaded models can be kept alongside the bundled example. Restart the app to clear them.")
        models[key] = m
        return information(m, key)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))


def probabilities(raw, interpretation):
    x = np.asarray(raw, np.float64).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("Model returned non-finite output values.")
    if interpretation == "auto":
        if np.all(x >= 0) and np.all(x <= 1) and abs(float(x.sum()) - 1) < 0.02:
            interpretation = "probabilities"
        else:
            raise ValueError("Output is not clearly a probability distribution. Select Logits or Scores in Model settings.")
    if interpretation == "probabilities":
        if np.any(x < 0) or not np.isclose(x.sum(), 1, atol=0.03):
            raise ValueError("Selected probability output, but values are negative or do not sum to 1.")
        return x / x.sum()
    if interpretation == "logits":
        x = x - x.max()
        ex = np.exp(x)
        return ex / ex.sum()
    if interpretation == "scores":
        return x
    raise ValueError("Select how the model output should be interpreted.")


class Predictor:
    def __init__(self, model, config):
        self.model, self.config = model, config
    def _convert(self, x):
        x = x.astype(np.float32)
        if self.config["channel_order"] == "BGR":
            x = x[..., ::-1].copy()
        mode = self.config["scaling"]
        if mode == "zero_one": x = x / 255
        elif mode == "minus_one_one": x = x / 127.5 - 1
        return x
    def predict(self, batch):
        raw = self.model.predict(np.stack([self._convert(x) for x in batch]))
        return np.stack([probabilities(row, self.config["interpretation"]) for row in raw])
    def prepare(self, rgb):
        h, w = self.model.image_size()
        if self.config["resize"] == "stretch":
            return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.config["resize"] == "center_crop":
            ih, iw = rgb.shape[:2]
            scale = max(w / iw, h / ih)
            nw, nh = max(w, round(iw * scale)), max(h, round(ih * scale))
            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x, y = (nw - w) // 2, (nh - h) // 2
            return resized[y:y + h, x:x + w]
        if self.config["resize"] == "letterbox":
            ih, iw = rgb.shape[:2]
            scale = min(w / iw, h / ih)
            nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            x, y = (w - nw) // 2, (h - nh) // 2
            canvas[y:y + nh, x:x + nw] = resized
            return canvas
        raise ValueError("Select a supported image resize method.")


import cv2


def analyse(data, model, config, labels, explain_class):
    rgb = decode_image(data)
    if rgb.shape[0] * rgb.shape[1] > MAX_PIXELS: raise ValueError("Image is larger than 20 megapixels.")
    predictor = Predictor(model, config)
    x = predictor.prepare(rgb)
    probs = predictor.predict(x[None])[0]
    pred = int(probs.argmax())
    names = labels or model.classes or [f"class_{i}" for i in range(model.class_count)]
    if len(names) != model.class_count: raise ValueError("Provide exactly one label per output class.")
    order = probs.argsort()[::-1]
    margin = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else 0.
    is_probability = config["interpretation"] != "scores"
    top_scores = [{"label": names[int(i)], "score": float(probs[int(i)])} for i in order[:2]]
    if is_probability:
        status = ("Uncertain · low top-two margin" if margin < config["margin_threshold"] else
                  "Uncertain · low top score" if probs[order[0]] < config["min_score_threshold"] else
                  "No uncertainty flag · thresholds are not correctness checks")
    else:
        status = "Raw scores · uncertainty threshold not applicable"
    target = names.index(explain_class) if explain_class in names else pred
    cam, peak = A.occlusion(predictor, x, target)
    tests, groups = A.run_tests(predictor, x, probs, names)
    result = {
        "label": names[pred], "confidence": float(probs[pred]) if is_probability else None,
        "highest_score": float(probs[order[0]]), "second_highest_score": float(probs[order[1]]) if len(order) > 1 else None,
        "top_scores": top_scores, "margin": margin, "status": status,
        "score_type": "probability" if is_probability else "raw score",
        "probs": {names[i]: float(probs[i]) for i in order}, "explained": names[target],
        "heatmap": A.heat_png(cam), "occlusion_peak_drop": peak, "hot_region": A.hot_region_stats(x, cam),
        "groups": groups, "tests": tests, "summary": A.summarise(groups, tests, names[pred]),
        "measurements": A.measure(x, rgb.shape[:2]), "inference_ms": None,
        "model": {"format": model.format, "input_shape": model.input_shape, "input_dtype": model.input_dtype,
                  "output_shape": model.output_shape, "output_dtype": model.output_dtype,
                  "class_count": model.class_count, "metadata": model.metadata},
    }
    return result


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model_id: str = Form("terrain-default"),
                  labels: str = Form(""), channel_order: str = Form("RGB"), resize: str = Form("stretch"),
                  scaling: str = Form("raw"), interpretation: str = Form("auto"),
                  margin_threshold: float = Form(0.20), min_score_threshold: float = Form(0.50),
                  explain_class: str | None = Form(None), known_correct_label: str = Form("")):
    if model_id not in models: raise HTTPException(400, "Select or upload a supported model first.")
    if Path(file.filename or "").suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(400, "Image must be JPG, JPEG, or PNG.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES: raise HTTPException(413, "Image is larger than 8 MB.")
    if channel_order not in ("RGB", "BGR"): raise HTTPException(400, "Channel order must be RGB or BGR.")
    if resize not in ("stretch", "center_crop", "letterbox"): raise HTTPException(400, "Select a supported resize method.")
    if scaling not in ("raw", "zero_one", "minus_one_one"): raise HTTPException(400, "Select a supported pixel range.")
    if interpretation not in ("auto", "probabilities", "logits", "scores"): raise HTTPException(400, "Select a supported output interpretation.")
    selected_model = models[model_id]
    input_quantization = selected_model.metadata.get("input_quantization", ())
    if (selected_model.input_dtype == "uint8" and len(input_quantization) == 2
            and float(input_quantization[0]) == 1.0 and int(input_quantization[1]) == 0 and scaling != "raw"):
        raise HTTPException(400, "This model's uint8 input quantization requires raw 0–255 pixels. 0–1 scaling is incompatible.")
    if selected_model.metadata.get("output_activation") == "softmax" and interpretation == "logits":
        raise HTTPException(400, "This model already applies softmax. Select Auto, Probabilities, or Raw scores; logits would apply softmax twice.")
    if not 0 <= margin_threshold <= 1: raise HTTPException(400, "Uncertainty margin threshold must be between 0 and 1.")
    if not 0 <= min_score_threshold <= 1: raise HTTPException(400, "Minimum top-score threshold must be between 0 and 1.")
    config = {"channel_order": channel_order, "resize": resize, "scaling": scaling, "interpretation": interpretation,
              "margin_threshold": margin_threshold, "min_score_threshold": min_score_threshold}
    names = [x.strip() for x in labels.splitlines() if x.strip()] or None
    started = time.perf_counter()
    try:
        out = await run_in_threadpool(analyse, data, models[model_id], config, names, explain_class)
    except Exception as e:
        raise HTTPException(400, str(e))
    out["inference_ms"] = round((time.perf_counter() - started) * 1000, 1)
    if known_correct_label.strip():
        valid_labels = names or models[model_id].classes or [f"class_{i}" for i in range(models[model_id].class_count)]
        if known_correct_label not in valid_labels:
            raise HTTPException(400, "Known correct label must match one configured class label exactly.")
        out["validation"] = {"known_label": known_correct_label, "correct": known_correct_label == out["label"]}
    history.insert(0, {"t": time.strftime("%H:%M:%S"), "name": file.filename, "label": out["label"],
                       "top_score": out["highest_score"], "score_type": out["score_type"], "status": out["status"]})
    del history[50:]
    return out


@app.get("/api/history")
def get_history():
    return history


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"), html=True))
