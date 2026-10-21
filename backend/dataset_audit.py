"""Evaluate the bundled model on labeled folders using the website's inference path.

Usage: python dataset_audit.py path/to/val [--per-class 40] [--out audit]
Expected folders: path/to/val/<class>/*.jpg (folder names must match model.py CLASSES).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

import analysis as A
from app import Predictor
from model import CLASSES, MODEL_PATH, ImageModel, decode_image

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--per-class", type=int, default=None, help="Optional maximum images per class (default: all)")
parser.add_argument("--out", default="audit")
args = parser.parse_args()

model = ImageModel(MODEL_PATH, "tflite", CLASSES)
config = {"channel_order": "RGB", "resize": "stretch", "scaling": "raw", "interpretation": "auto",
          "margin_threshold": 0.20, "min_score_threshold": 0.50}
predictor = Predictor(model, config)
rows, probs_all, meas_all = [], [], []

for cls in CLASSES:
    files = sorted(sum((glob.glob(os.path.join(args.root, cls, e)) for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")), []))
    if args.per_class is not None:
        files = files[:args.per_class]
    if not files:
        print(f"[skip] no images for '{cls}' in {args.root}/{cls}")
        continue
    for filename in files:
        try:
            rgb = decode_image(open(filename, "rb").read())
            x = predictor.prepare(rgb)
            p = predictor.predict(x[None])[0]
            _, groups = A.run_tests(predictor, x, p, CLASSES, thumbs=False)
            measurements = A.flat(A.measure(x, rgb.shape[:2]))
        except Exception as exc:
            print(f"[skip] {filename}: {exc}")
            continue
        row = {"file": filename, "true": cls, "pred": CLASSES[int(p.argmax())], "top_score": float(p.max())}
        row.update({f"sens.{g['id']}": g["sensitivity"] for g in groups})
        row.update({f"flips.{g['id']}": g["flips"] for g in groups})
        row.update({f"m.{key}": value for key, value in measurements.items()})
        rows.append(row)
        probs_all.append(p)
        meas_all.append(measurements)
    print(f"{cls}: {sum(row['true'] == cls for row in rows)} images")

if not rows:
    raise SystemExit("No labeled images found. Expected a class-named subfolder for at least one configured class.")

confusion = {true: {pred: 0 for pred in CLASSES} for true in CLASSES}
for row in rows:
    confusion[row["true"]][row["pred"]] += 1

metrics = {}
for cls in CLASSES:
    tp = confusion[cls][cls]
    support = sum(confusion[cls].values())
    predicted = sum(confusion[true][cls] for true in CLASSES)
    precision = tp / predicted if predicted else 0.0
    recall = tp / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics[cls] = {"support": support, "precision": precision, "recall": recall, "f1": f1}

P = np.asarray(probs_all)
keys = list(meas_all[0])
M = np.asarray([[measurement[key] for key in keys] for measurement in meas_all])
summary = {"n": len(rows), "accuracy": float(np.mean([row["true"] == row["pred"] for row in rows])),
           "metrics": metrics, "confusion_matrix": confusion, "classes": {}}
print(f"\nOverall accuracy: {summary['accuracy']:.1%} on {len(rows)} images")
for cls in CLASSES:
    print(f"{cls}: precision={metrics[cls]['precision']:.3f} recall={metrics[cls]['recall']:.3f} "
          f"f1={metrics[cls]['f1']:.3f} n={metrics[cls]['support']}")
    class_rows = [row for row in rows if row["true"] == cls]
    if not class_rows:
        continue
    sensitivity = {group: float(np.mean([row[f"sens.{group}"] for row in class_rows])) for group, _ in A.GROUPS}
    class_index = CLASSES.index(cls)
    correlations = []
    for index, key in enumerate(keys):
        if np.std(M[:, index]) > 1e-9 and np.std(P[:, class_index]) > 1e-9:
            rho = spearmanr(M[:, index], P[:, class_index])[0]
            if np.isfinite(rho):
                correlations.append((key, float(rho)))
    correlations.sort(key=lambda pair: -abs(pair[1]))
    confused_with = {pred: count for pred, count in confusion[cls].items() if pred != cls and count}
    summary["classes"][cls] = {**metrics[cls], "confused_with": confused_with, "sensitivity": sensitivity,
                               "top_correlated": correlations[:6]}

with open(args.out + "_images.csv", "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with open(args.out + "_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print(f"Wrote {args.out}_images.csv and {args.out}_summary.json")
