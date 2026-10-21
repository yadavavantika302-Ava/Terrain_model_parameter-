"""Runtime adapters for image classification models (no uploaded code execution)."""
import os
import threading
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("TERRAIN_MODEL", os.path.join(HERE, "terrain_model_int8.tflite"))
CLASSES = ["barren", "building", "grass", "road", "vegetation", "water"]
_cf = os.path.join(HERE, "classes.txt")
if os.path.exists(_cf):
    with open(_cf, encoding="utf-8") as f:
        CLASSES = [line.strip() for line in f if line.strip()]


def decode_image(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not read that file as a JPG, JPEG, or PNG image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class ImageModel:
    def __init__(self, path, fmt, labels=None):
        self.path, self.format = path, fmt
        self.lock = threading.Lock()
        self.metadata = {}
        if fmt == "tflite":
            self._load_tflite()
        elif fmt in ("keras", "h5"):
            self._load_keras()
        elif fmt == "onnx":
            self._load_onnx()
        else:
            raise ValueError("Supported model files are .tflite, .keras, .h5, and .onnx.")
        self.classes = labels or (CLASSES.copy() if os.path.abspath(path) == os.path.abspath(MODEL_PATH) else [])
        self.class_count = self.output_shape[-1] if self.output_shape else 0
        if self.class_count <= 1 or len(self.output_shape) != 2:
            raise ValueError(f"Expected one classification output shaped [batch, classes]; found {self.output_shape}.")
        if self.classes and len(self.classes) != self.class_count:
            raise ValueError(f"{len(self.classes)} labels supplied but model output has {self.class_count} classes.")

    def _load_tflite(self):
        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    from tensorflow.lite import Interpreter
            self.engine = Interpreter(model_path=self.path)
            self.engine.allocate_tensors()
            self.in_detail = self.engine.get_input_details()[0]
            self.out_detail = self.engine.get_output_details()[0]
            self.input_shape = [int(x) for x in self.in_detail["shape"]]
            self.input_dtype = np.dtype(self.in_detail["dtype"]).name
            self.output_dtype = np.dtype(self.out_detail["dtype"]).name
            self.output_shape = [int(x) for x in self.out_detail["shape"]]
            self.metadata = {"input_quantization": list(self.in_detail.get("quantization", ())),
                             "output_quantization": list(self.out_detail.get("quantization", ()))}
            try:
                op_names = [op.get("op_name") for op in self.engine._get_ops_details() if op.get("op_name") != "DELEGATE"]
                self.metadata["output_activation"] = "softmax" if (op_names[-1:] == ["SOFTMAX"] or op_names[-2:] == ["SOFTMAX", "QUANTIZE"]) else "unknown"
            except Exception:
                self.metadata["output_activation"] = "unknown"
        except ImportError as e:
            raise ValueError("TFLite runtime is unavailable. Install ai-edge-litert or tensorflow.") from e
        except Exception as e:
            raise ValueError(f"Could not load TFLite model: {e}") from e

    def _load_keras(self):
        try:
            import tensorflow as tf
            self.engine = tf.keras.models.load_model(self.path, compile=False, safe_mode=True)
            self.input_shape = [int(x) if x is not None else -1 for x in self.engine.input_shape]
            self.output_shape = [int(x) if x is not None else -1 for x in self.engine.output_shape]
            self.input_dtype = str(self.engine.inputs[0].dtype).replace("<dtype: '", "").replace("'>", "")
            self.output_dtype = str(self.engine.outputs[0].dtype).replace("<dtype: '", "").replace("'>", "")
            self.metadata = {"keras_name": self.engine.name, "safe_mode": True}
        except ImportError as e:
            raise ValueError("Keras models require TensorFlow. Install tensorflow.") from e
        except Exception as e:
            raise ValueError(f"Could not load safe Keras model: {e}") from e

    def _load_onnx(self):
        try:
            import onnxruntime as ort
            self.engine = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
            i, o = self.engine.get_inputs()[0], self.engine.get_outputs()[0]
            self.input_name, self.output_name = i.name, o.name
            self.input_shape = [int(x) if isinstance(x, int) else -1 for x in i.shape]
            self.output_shape = [int(x) if isinstance(x, int) else -1 for x in o.shape]
            self.input_dtype = i.type.replace("tensor(", "").replace(")", "")
            self.output_dtype = o.type.replace("tensor(", "").replace(")", "")
            self.metadata = {"input_name": i.name, "output_name": o.name, "providers": self.engine.get_providers()}
        except ImportError as e:
            raise ValueError("ONNX models require onnxruntime. Install onnxruntime.") from e
        except Exception as e:
            raise ValueError(f"Could not load ONNX model: {e}") from e

    def image_size(self):
        s = self.input_shape
        if len(s) != 4:
            raise ValueError(f"Model input must be a 4D image tensor; found {s}.")
        if s[-1] in (1, 3, 4):
            h, w = s[1], s[2]
            self.layout = "NHWC"
            channels = s[-1]
        elif s[1] in (1, 3, 4):
            h, w = s[2], s[3]
            self.layout = "NCHW"
            channels = s[1]
        else:
            raise ValueError(f"Cannot identify image channels in input shape {s}.")
        if channels != 3:
            raise ValueError(f"Expected a three-channel color image input; found {channels} channels.")
        if h <= 0 or w <= 0 or h > 4096 or w > 4096:
            raise ValueError(f"Unsupported or dynamic image dimensions in {s}.")
        return h, w

    def predict(self, batch):
        """Return dequantized output scores; caller selects score interpretation."""
        out = []
        with self.lock:
            for x in batch:
                dtype = np.dtype(self.input_dtype)
                if dtype.kind in "iu":
                    scale, zero = self.metadata.get("input_quantization", (0, 0))
                    x = np.rint(x / scale + zero) if scale else np.rint(x)
                    x = np.clip(x, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
                else:
                    x = x.astype(dtype)
                inp = x[None]
                if self.layout == "NCHW":
                    inp = np.transpose(inp, (0, 3, 1, 2))
                if self.format == "tflite":
                    self.engine.set_tensor(self.in_detail["index"], inp)
                    self.engine.invoke()
                    r = self.engine.get_tensor(self.out_detail["index"])
                    scale, zero = self.metadata.get("output_quantization", (0, 0))
                    if scale:
                        r = scale * (r.astype(np.float32) - zero)
                elif self.format == "onnx":
                    r = self.engine.run([self.output_name], {self.input_name: inp})[0]
                else:
                    r = self.engine(inp, training=False).numpy()
                r = np.asarray(r).reshape(-1)
                if r.size != self.class_count:
                    raise ValueError("Model output changed shape during inference.")
                out.append(r.astype(np.float32))
        return np.stack(out)

    def prepare(self, rgb, config):
        h, w = self.image_size()
        methods = {"bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST,
                   "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC}
        method = config.get("resize", "bilinear")
        if method not in methods:
            raise ValueError("Unsupported resize method.")
        x = cv2.resize(rgb, (w, h), interpolation=methods[method])
        if config.get("channel_order", "RGB") == "BGR":
            x = x[..., ::-1].copy()
        scale = config.get("scaling", "raw")
        if scale == "zero_one":
            x = x.astype(np.float32) / 255
        elif scale == "minus_one_one":
            x = x.astype(np.float32) / 127.5 - 1
        elif scale == "raw":
            x = x.astype(np.float32)
        else:
            raise ValueError("Choose a supported input scaling method.")
        return x
