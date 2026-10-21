"""Black-box explanation for a TFLite classifier (no gradients available).

Three independent views of "what is the model using?":
  1. measure()      - plain image measurements (colour, texture, edges, shape, size). Descriptive only.
  2. run_tests()    - perturbation sensitivity: change ONE property, re-run the model, and measure score changes.
                      'remove' tests break a property; 'keep' tests retain only it. These are not proof of understanding.
  3. occlusion()    - which image regions matter (cover a patch, see the score drop).
"""
import io, base64
import numpy as np, cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

GROUPS = [("colour", "Colour"), ("lighting", "Brightness / contrast"), ("texture", "Texture"),
          ("edges", "Edges / sharpness"), ("layout", "Shape / layout"), ("scale", "Size / scale")]
GROUP_NAME = dict(GROUPS)

MEASURE_HINT = {
    "hue_deg": "Average hue angle, saturation-weighted (0 red, 120 green, 240 blue)",
    "greenness_exg": "Excess-green index (2G-R-B)/255. Higher = greener, typical for grass/vegetation",
    "blue_dominance": "How much blue exceeds red/green. Higher = water/sky-like",
    "colourfulness": "Hasler-Susstrunk colourfulness. Low = grey/brown/barren look",
    "laplacian_var": "Variance of Laplacian: fine detail / sharpness",
    "lbp_entropy": "Local-binary-pattern entropy: how varied the micro-texture is",
    "glcm_contrast": "Grey-level co-occurrence contrast: strong local intensity jumps",
    "glcm_homogeneity": "Grey-level co-occurrence homogeneity: smooth/uniform areas",
    "canny_density": "Share of pixels that are Canny edges",
    "mean_gradient": "Average Sobel gradient strength",
    "vertical_edge_share": "Share of gradient energy from vertical edges (buildings high, roads/horizons low)",
    "hough_lines": "Count of long straight line segments (man-made structure)",
    "n_regions": "Number of colour regions larger than 2% of the image",
    "largest_region_share": "Area share of the biggest colour region",
    "largest_region_elongation": "Long/short side of the biggest region (roads are elongated)",
    "largest_region_solidity": "Region area / convex hull area (1 = compact blob)",
    "stretch_distortion": "Original aspect ratio. The model stretches to a square, so >1 distorts shapes",
    "mean_spatial_freq": "Power-weighted spatial frequency (cycles/pixel). Higher = finer features",
    "high_freq_share": "Share of spectral power above 0.25 cycles/pixel",
}


# ---------------------------------------------------------------- 1. measurements
def _circ_hue(h_deg, w):
    a = np.deg2rad(h_deg)
    return float(np.rad2deg(np.arctan2((np.sin(a) * w).sum(), (np.cos(a) * w).sum())) % 360)


def measure(x: np.ndarray, orig_hw):
    """x: model-sized RGB uint8. Returns {group: {name: value}} (+ colour.dominant_colours)."""
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    f = x.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B

    px = f.reshape(-1, 3)[:: max(1, len(f.reshape(-1, 3)) // 2048)]
    _, lab, cen = cv2.kmeans(px, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                             3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab.ravel(), minlength=3) / len(lab)
    dom = [{"hex": "#%02x%02x%02x" % tuple(int(v) for v in cen[i]), "share": float(share[i])}
           for i in np.argsort(-share)]
    colour = {
        "mean_red": float(R.mean()), "mean_green": float(G.mean()), "mean_blue": float(B.mean()),
        "hue_deg": _circ_hue(hsv[..., 0].astype(np.float32) * 2, hsv[..., 1].astype(np.float32) + 1e-3),
        "saturation": float(hsv[..., 1].mean() / 255), "brightness": float(hsv[..., 2].mean() / 255),
        "greenness_exg": float((2 * G - R - B).mean() / 255),
        "blue_dominance": float((B - np.maximum(R, G)).mean() / 255),
        "colourfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
        "dominant_colours": dom,
    }
    lighting = {"contrast_std": float(gray.std() / 255)}

    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    hist = np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size
    glcm = graycomatrix((gray // 8).astype(np.uint8), [1], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    texture = {
        "grey_std": float(gray.std()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "lbp_entropy": float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum()),
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
    }

    canny = cv2.Canny(gray, 100, 200)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=5)
    edges = {
        "canny_density": float((canny > 0).mean()),
        "mean_gradient": float(np.hypot(gx, gy).mean() / 255),
        "vertical_edge_share": float(np.abs(gx).sum() / (np.abs(gx).sum() + np.abs(gy).sum() + 1e-6)),
        "hough_lines": int(0 if lines is None else len(lines)),
    }

    small = cv2.resize(x, (64, 64), interpolation=cv2.INTER_AREA)
    _, lbl, _ = cv2.kmeans(cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32), 4, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(64, 64).astype(np.uint8)
    n_reg, best, best_mask = 0, 0, None
    for k in range(4):
        n, cc, st, _ = cv2.connectedComponentsWithStats((lbl == k).astype(np.uint8), connectivity=8)
        for j in range(1, n):
            a = st[j, cv2.CC_STAT_AREA]
            if a > 0.02 * 4096:
                n_reg += 1
                if a > best:
                    best, best_mask = a, (cc == j).astype(np.uint8)
    elong, solid = 1.0, 1.0
    if best_mask is not None:
        cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        elong = float(max(rw, rh) / max(min(rw, rh), 1.0))
        hull = cv2.contourArea(cv2.convexHull(c))
        solid = float(cv2.contourArea(c) / hull) if hull > 0 else 1.0
    layout = {"n_regions": n_reg, "largest_region_share": float(best / 4096),
              "largest_region_elongation": elong, "largest_region_solidity": solid}

    P = np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1])))))**2
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(gray.shape[0])), np.fft.fftshift(np.fft.fftfreq(gray.shape[1])), indexing="ij")
    r = np.hypot(fx, fy)
    tot = P.sum() + 1e-9
    oh, ow = orig_hw
    scale = {
        "orig_width": int(ow), "orig_height": int(oh), "megapixels": float(oh * ow / 1e6),
        "stretch_distortion": float(max(ow / oh, oh / ow)),
        "mean_spatial_freq": float((P * r).sum() / tot), "high_freq_share": float(P[r > 0.25].sum() / tot),
    }
    return {"colour": colour, "lighting": lighting, "texture": texture, "edges": edges, "layout": layout, "scale": scale}


def flat(m):
    return {f"{g}.{k}": v for g, d in m.items() for k, v in d.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------- 2. interventions
def _tile_shuffle(x, n, seed=0):
    h, w = x.shape[0] // n, x.shape[1] // n
    tiles = [x[i * h:(i + 1) * h, j * w:(j + 1) * w] for i in range(n) for j in range(n)]
    order = np.random.default_rng(seed).permutation(len(tiles))
    rows = [np.concatenate([tiles[order[i * n + j]] for j in range(n)], 1) for i in range(n)]
    return np.concatenate(rows, 0)


def _gray3(g):
    return np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, 2)


def interventions(x):
    """Returns [(id, group, kind, label, image)]. kind: 'remove' (break a property) or 'keep' (keep only it)."""
    H, W = x.shape[:2]
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    T = []
    add = lambda i, g, k, n, im: T.append((i, g, k, n, np.ascontiguousarray(im)))

    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    hs = hsv.copy(); hs[..., 0] = (hs[..., 0].astype(int) + 60) % 180
    add("grayscale", "colour", "remove", "Grayscale (no colour)", _gray3(gray))
    add("hue_shift", "colour", "remove", "Hue rotated 120 degrees", cv2.cvtColor(hs, cv2.COLOR_HSV2RGB))
    add("rgb_swap", "colour", "remove", "Red and blue swapped", x[..., ::-1])
    for c, nm in enumerate(["Red", "Green", "Blue"]):
        t = x.copy(); t[..., c] = int(x[..., c].mean())
        add(f"flat_{nm.lower()}", "colour", "remove", f"{nm} channel flattened", t)
    add("colour_only", "colour", "keep", "Colour only (heavily blurred)", cv2.GaussianBlur(x, (0, 0), 12))

    add("darker", "lighting", "remove", "Darker (x0.6)", cv2.convertScaleAbs(x, alpha=0.6))
    add("brighter", "lighting", "remove", "Brighter (x1.5)", cv2.convertScaleAbs(x, alpha=1.5))
    m = x.mean()
    add("low_contrast", "lighting", "remove", "Low contrast", np.clip((x.astype(np.float32) - m) * 0.4 + m, 0, 255).astype(np.uint8))

    sm = x
    for _ in range(3):
        sm = cv2.bilateralFilter(sm, 9, 60, 7)
    add("smooth", "texture", "remove", "Fine texture smoothed (edges kept)", sm)
    add("noise", "texture", "remove", "Heavy noise added", np.clip(x + np.random.default_rng(1).normal(0, 25, x.shape), 0, 255).astype(np.uint8))
    g = gray.astype(np.float32)
    add("texture_only", "texture", "keep", "Texture only (grey high-pass)", _gray3((g - cv2.GaussianBlur(g, (0, 0), 2)) * 4 + 128))

    add("blur", "edges", "remove", "Blurred (edges and detail lost)", cv2.GaussianBlur(x, (0, 0), 4))
    add("sharpen", "edges", "remove", "Over-sharpened", np.clip(cv2.addWeighted(x, 2.5, cv2.GaussianBlur(x, (0, 0), 3), -1.5, 0), 0, 255).astype(np.uint8))
    e = cv2.dilate(cv2.Canny(gray, 80, 160), np.ones((2, 2), np.uint8))
    add("edges_only", "edges", "keep", "Edges only", _gray3(e))

    add("shuffle4", "layout", "remove", "Tiles shuffled 4x4 (layout broken)", _tile_shuffle(x, 4))
    add("shuffle8", "layout", "remove", "Tiles shuffled 8x8 (finer)", _tile_shuffle(x, 8))
    add("rot90", "layout", "remove", "Rotated 90 degrees", np.rot90(x))
    add("flip_v", "layout", "remove", "Flipped upside-down", x[::-1])

    q = H // 4
    add("zoom_in", "scale", "remove", "Zoomed in 2x (centre crop)", cv2.resize(x[q:H - q, q:W - q], (W, H)))
    add("half_scale", "scale", "remove", "Everything half size (tiled 2x2)", np.tile(cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA), (2, 2, 1)))
    add("pixelate", "scale", "remove", "Pixelated to 32x32", cv2.resize(cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_NEAREST))
    return T


def _thumb(im, s=96):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def run_tests(model, x, base_probs, classes, thumbs=True):
    pred = int(base_probs.argmax()); p0 = float(base_probs[pred])
    T = interventions(x)
    P = model.predict(np.stack([t[4] for t in T]))
    tests = []
    for (tid, grp, kind, name, im), p in zip(T, P):
        new = int(p.argmax())
        modified_score = float(p[pred])
        tests.append({"id": tid, "group": grp, "kind": kind, "name": name,
                      "original_score": p0, "modified_score": modified_score,
                      "score_delta": modified_score - p0, "drop": p0 - modified_score,
                      "p_orig": modified_score,
                      "label": classes[new], "conf": float(p[new]), "flipped": new != pred,
                      "thumb": _thumb(im) if thumbs else None})
    groups = []
    for gid, gname in GROUPS:
        rm = [t for t in tests if t["group"] == gid and t["kind"] == "remove"]
        drops = np.clip([t["drop"] for t in rm], 0, None)
        groups.append({"id": gid, "name": gname, "sensitivity": float(drops.mean()), "worst_drop": float(drops.max()),
                       "flips": int(sum(t["flipped"] for t in rm)), "n": len(rm),
                       "keep_ok": [t["flipped"] is False for t in tests if t["group"] == gid and t["kind"] == "keep"]})
    return tests, groups


def summarise(groups, tests, label):
    g = sorted(groups, key=lambda d: -d["sensitivity"])
    lines = []
    if g[0]["sensitivity"] < 0.05:
        lines.append(f"No single property dominates: changing colour, texture, edges, layout or scale barely moved the '{label}' score.")
    else:
        top = [d for d in g if d["sensitivity"] >= 0.05][:3]
        lines.append(f"Largest measured perturbation sensitivity for '{label}': " + ", then ".join(f"{d['name'].lower()} ({d['sensitivity']*100:.0f} pts mean score drop)" for d in top) + ".")
        low = [d["name"].lower() for d in g if d["sensitivity"] < 0.05]
        if low:
            lines.append("Little effect from: " + ", ".join(low) + ".")
    kept = {t["group"]: not t["flipped"] for t in tests if t["kind"] == "keep"}
    ok = [GROUP_NAME[k].lower() for k, v in kept.items() if v]
    if ok:
        lines.append("Still predicts " + f"'{label}' with only " + " / ".join(ok) + " left, so that property alone is enough.")
    else:
        lines.append("Colour-only, texture-only and edges-only versions all change the answer: the model combines several cues.")
    return lines


# ---------------------------------------------------------------- 3. occlusion
def occlusion(model, x, cls, patch=56, stride=28):
    H, W = x.shape[:2]
    base = float(model.predict(x[None])[0][cls])
    fill = x.reshape(-1, 3).mean(0).astype(np.uint8)
    boxes, batch = [], []
    for y in range(0, H - patch + 1, stride):
        for xx in range(0, W - patch + 1, stride):
            t = x.copy(); t[y:y + patch, xx:xx + patch] = fill
            boxes.append((y, xx)); batch.append(t)
    drops = base - model.predict(np.stack(batch))[:, cls]
    acc, cnt = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for (y, xx), d in zip(boxes, drops):
        acc[y:y + patch, xx:xx + patch] += d; cnt[y:y + patch, xx:xx + patch] += 1
    cam = np.clip(acc / np.maximum(cnt, 1), 0, None)
    peak = float(cam.max())
    return (cam / peak if peak > 1e-6 else cam), peak


def hot_region_stats(x, cam):
    """Compare the most influential region with the rest of the image."""
    if cam.max() <= 0:
        return None
    hot = cam >= max(np.quantile(cam, 0.75), 1e-6)
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edge = cv2.Canny(gray, 100, 200) > 0
    f = x.astype(np.float32)
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / 255
    def s(m):
        return {"mean_rgb": [float(f[..., c][m].mean()) for c in range(3)], "greenness_exg": float(exg[m].mean()),
                "edge_density": float(edge[m].mean()), "detail": float(lap[m].mean()), "brightness": float(gray[m].mean() / 255)}
    return {"hot": s(hot), "rest": s(~hot), "area_share": float(hot.mean()),
            "spread": float((cam >= 0.5).mean())}


def heat_png(cam):
    rgba = np.zeros((*cam.shape, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (255 * (1 - cam)).astype(np.uint8)
    rgba[..., 3] = (200 * cam).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return "data:image/png;base64," + base64.b64encode(buf).decode()
