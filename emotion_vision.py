#!/usr/bin/env python3
"""
EMOTION VISION v2  
"""
import argparse
import csv
import math
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
SHOTS = ROOT / "shots"
WIN = "EMOTION VISION"
FONT = cv2.FONT_HERSHEY_DUPLEX
SFONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

EMO = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

COL = {   # BGR
    "Anger": (70, 70, 255),
    "Contempt": (200, 110, 220),
    "Disgust": (80, 200, 110),
    "Fear": (210, 90, 150),
    "Happiness": (60, 215, 255),
    "Neutral": (230, 215, 150),
    "Sadness": (240, 130, 50),
    "Surprise": (255, 225, 90),
}
VA = {    # valence / arousal anchors (circumplex, approximate)
    "Anger": (-0.60, 0.70), "Contempt": (-0.50, 0.20), "Disgust": (-0.65, 0.35),
    "Fear": (-0.65, 0.65), "Happiness": (0.80, 0.50), "Neutral": (0.00, 0.00),
    "Sadness": (-0.70, -0.40), "Surprise": (0.30, 0.80),
}
HUD = (255, 190, 60)        # electric blue / cyan (BGR)
HUD2 = (255, 238, 195)      # bright ice
HUDD = (150, 100, 40)       # dim blue
WHITE = (240, 240, 240)
DIM = (150, 150, 150)

BOOT_T = 4.0


# ----------------------------------------------------------------------------- models
class Detector:
    def __init__(self, det=640, pack="buffalo_l", mesh=True):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in avail]
        prov.append("CPUExecutionProvider")
        mods = ["detection"] + (["landmark_2d_106"] if mesh else [])
        self.app = FaceAnalysis(name=pack, providers=prov, allowed_modules=mods)
        cuda = "CUDAExecutionProvider" in prov
        self.app.prepare(ctx_id=0 if cuda else -1, det_size=(det, det))
        self.device = "CUDA" if cuda else ("CoreML" if "CoreMLExecutionProvider" in prov else "CPU")

    def __call__(self, frame):
        return self.app.get(frame)


class EmotionNet:
    def __init__(self, name="enet_b0_8_best_afew"):
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        except ImportError:
            raise SystemExit("Missing package. Run:  pip install hsemotion-onnx")
        self.m = HSEmotionRecognizer(model_name=name)
        idx = self.m.idx_to_class
        names = [idx[i] for i in sorted(idx)]
        self.order = [names.index(e) if e in names else -1 for e in EMO]

    def __call__(self, rgb):
        _, s = self.m.predict_emotions(rgb, logits=False)
        s = np.asarray(s, dtype=np.float64).ravel()
        if s.min() < 0 or abs(s.sum() - 1.0) > 1e-2:
            e = np.exp(s - s.max())
            s = e / e.sum()
        out = np.zeros(len(EMO))
        for i, k in enumerate(self.order):
            if k >= 0:
                out[i] = s[k]
        return out / max(out.sum(), 1e-9)


def crop_face(frame, box, margin=0.12):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mw)), int(max(0, y1 - mh))
    x2, y2 = int(min(W, x2 + mw)), int(min(H, y2 + mh))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def valence_arousal(p):
    v = sum(p[i] * VA[e][0] for i, e in enumerate(EMO))
    a = sum(p[i] * VA[e][1] for i, e in enumerate(EMO))
    return float(v), float(a)


def mood_label(v, a):
    if v > 0.25 and a > 0.30:
        return "EXCITED"
    if v > 0.25:
        return "CONTENT"
    if v < -0.25 and a > 0.25:
        return "TENSE"
    if v < -0.25:
        return "DOWN"
    if a > 0.35:
        return "ALERT"
    return "BALANCED"


def clarity(p):
    h = -float(np.sum(p * np.log(p + 1e-9))) / math.log(len(p))
    return float(np.clip(1.0 - h, 0, 1))


# face mesh topology
_MESH = {"E": np.zeros((0, 2), int), "n": 0}


def mesh_build(lm):
    """Delaunay edges over the 106 landmarks, built once (indices keep their meaning)."""
    if _MESH["n"] == len(lm):
        return
    _MESH["n"] = len(lm)
    try:
        from scipy.spatial import Delaunay
    except Exception:
        return
    p = lm - lm.min(0)
    p = p / max(float(p.max()), 1e-6)
    edges = set()
    for a, b, c in Delaunay(p).simplices:
        d = (np.linalg.norm(p[a] - p[b]), np.linalg.norm(p[b] - p[c]), np.linalg.norm(p[c] - p[a]))
        if max(d) < 0.34:       # drop long hull-filling triangles
            for u, w in ((a, b), (b, c), (c, a)):
                edges.add((min(u, w), max(u, w)))
    _MESH["E"] = np.array(sorted(edges), int).reshape(-1, 2)


#  primitives
def txt(img, s, org, sc=0.5, color=WHITE, th=1, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, sc, (0, 0, 0), th + 1, AA)
    cv2.putText(img, s, org, FONT, sc, color, th, AA)


def tiny(img, s, org, sc=0.3, color=HUD2):
    cv2.putText(img, s, (int(org[0]), int(org[1])), SFONT, sc, color, 1, AA)


def panel(img, x1, y1, x2, y2, color=(26, 20, 16), alpha=0.55, border=None):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1 - alpha, np.full_like(sub, color), alpha, 0)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), border, 1, AA)


def brackets(img, box, color, t=2, frac=0.22):
    x1, y1, x2, y2 = box
    L = max(10, int(min(x2 - x1, y2 - y1) * frac))
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * L, py), color, t, AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, t, AA)


def bar(img, x, y, w, h, value, color, bg=(60, 60, 60)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, value))), y + h), color, -1)


def make_fx(h, w):
    """Vignette + faint scan-lines + cool tint, as a uint8 multiplier (applied with cv2.multiply)."""
    y, x = np.ogrid[:h, :w]
    d = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
    v = 1.0 - 0.55 * np.clip(d - 0.30, 0, 1)
    scan = np.ones((h, 1), np.float32)
    scan[::3] = 0.90
    fx = (v * scan)[..., None] * np.array([1.0, 0.94, 0.82], np.float32)
    return np.clip(fx * 255, 0, 255).astype(np.uint8)


def add_glow(frame, glow):
    h, w = glow.shape[:2]
    sm = cv2.resize(glow, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 3)
    sm = cv2.convertScaleAbs(sm, alpha=2.2)
    return cv2.add(frame, cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR))


def dashed(img, c, r, col, rot, n=48, duty=0.55, th=1):
    step = 360.0 / n
    for i in range(n):
        a = rot + i * step
        cv2.ellipse(img, c, (r, r), 0, a, a + step * duty, col, th, AA)


def ticks(img, c, r, col, n=90, rot=0.0, th=1, long_every=6, l1=5, l2=11):
    for i in range(n):
        a = math.radians(rot + i * 360.0 / n)
        l = l2 if i % long_every == 0 else l1
        ca, sa = math.cos(a), math.sin(a)
        cv2.line(img, (int(c[0] + r * ca), int(c[1] + r * sa)),
                 (int(c[0] + (r + l) * ca), int(c[1] + (r + l) * sa)), col, th, AA)


def draw_rings(frame, glow, c, R, acc, now, seed):
    """Rotating neon dial around a centre point (frame = sharp layer, glow = bloom layer)."""
    rot = now * 38.0 + seed * 57.0
    for img in (glow, frame):
        m = 3 if img is glow else 1
        cv2.circle(img, c, R, HUD, m, AA)
        cv2.circle(img, c, int(R * 0.88), HUDD, m, AA)
        dashed(img, c, int(R * 1.10), HUD, -rot * 1.4, 48, 0.55, m)
        ticks(img, c, int(R * 1.24), HUD, 90, rot * 0.5, m)
        for j in range(3):                               # bright arcs
            a = rot * 1.6 + j * 120
            cv2.ellipse(img, c, (R, R), 0, a, a + 55, acc, 2 + (2 if img is frame else 4), AA)
        for j in range(4):                               # chunky bars on the outer dial
            a = -rot * 0.9 + j * 90 + 20
            cv2.ellipse(img, c, (int(R * 1.24), int(R * 1.24)), 0, a, a + 7,
                        HUD2 if img is frame else HUD, 5, AA)
    cv2.putText(frame, "<<<<<<<<", (c[0] - 38, c[1] + int(R * 1.24) + 34), SFONT, 0.45, HUD, 1, AA)


def eye_reticle(frame, glow, p, r, col, now, sign):
    p = (int(p[0]), int(p[1]))
    for img in (glow, frame):
        m = 3 if img is glow else 1
        cv2.circle(img, p, r, col, m, AA)
        cv2.circle(img, p, max(2, int(r * 0.4)), HUD2, m, AA)
        a = (now * 150 * sign) % 360
        cv2.ellipse(img, p, (int(r * 1.5), int(r * 1.5)), 0, a, a + 80, HUD, 1 + m, AA)
        cv2.ellipse(img, p, (int(r * 1.5), int(r * 1.5)), 0, a + 180, a + 260, HUD, 1 + m, AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(frame, (p[0] + dx * int(r * 1.2), p[1] + dy * int(r * 1.2)),
                 (p[0] + dx * int(r * 2.1), p[1] + dy * int(r * 2.1)), HUD2, 1, AA)


def draw_mesh(frame, lm, alpha=0.55):
    H, W = frame.shape[:2]
    mesh_build(lm)
    E = _MESH["E"]
    if len(E) == 0:
        return
    x0, x1 = int(max(0, lm[:, 0].min() - 4)), int(min(W, lm[:, 0].max() + 4))
    y0, y1 = int(max(0, lm[:, 1].min() - 4)), int(min(H, lm[:, 1].max() + 4))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    roi = frame[y0:y1, x0:x1]
    ov = roi.copy()
    off = np.array([x0, y0], np.float32)
    cv2.polylines(ov, list((lm[E] - off).astype(np.int32)), False, HUD, 1, AA)
    for p in (lm - off).astype(np.int32):
        cv2.circle(ov, (int(p[0]), int(p[1])), 1, HUD2, -1, AA)
    cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)


#  emotion glyph
def glyph(img, cx, cy, r, emo, col):
    """Tiny vector face that mirrors the detected emotion."""
    i = lambda v: int(round(v))
    cv2.circle(img, (cx, cy), r, col, 2, AA)
    ex, ey = i(r * 0.38), i(r * 0.25)
    er = max(2, i(r * (0.17 if emo in ("Surprise", "Fear") else 0.11)))
    for sx in (-1, 1):
        cv2.circle(img, (cx + sx * ex, cy - ey), er, col, -1, AA)
    my = cy + i(r * 0.38)
    if emo == "Happiness":
        cv2.ellipse(img, (cx, cy + i(r * 0.08)), (i(r * 0.52), i(r * 0.46)), 0, 15, 165, col, 2, AA)
    elif emo == "Sadness":
        cv2.ellipse(img, (cx, cy + i(r * 0.72)), (i(r * 0.38), i(r * 0.30)), 0, 200, 340, col, 2, AA)
    elif emo == "Surprise":
        cv2.ellipse(img, (cx, my), (i(r * 0.14), i(r * 0.21)), 0, 0, 360, col, 2, AA)
        for sx in (-1, 1):
            cv2.line(img, (cx + sx * ex - i(r * .12), cy - ey - i(r * .30)),
                     (cx + sx * ex + i(r * .12), cy - ey - i(r * .30)), col, 2, AA)
    elif emo == "Fear":
        cv2.ellipse(img, (cx, my), (i(r * 0.26), i(r * 0.11)), 0, 0, 360, col, 2, AA)
        for sx in (-1, 1):
            cv2.line(img, (cx + sx * ex - sx * i(r * .15), cy - ey - i(r * .12)),
                     (cx + sx * ex + sx * i(r * .15), cy - ey - i(r * .30)), col, 2, AA)
    elif emo == "Anger":
        for sx in (-1, 1):
            cv2.line(img, (cx + sx * (ex + i(r * .18)), cy - ey - i(r * .24)),
                     (cx + sx * (ex - i(r * .18)), cy - ey - i(r * .06)), col, 2, AA)
        cv2.ellipse(img, (cx, cy + i(r * 0.68)), (i(r * 0.33), i(r * 0.20)), 0, 200, 340, col, 2, AA)
    elif emo == "Disgust":
        pts = np.array([(cx - r * .4, my), (cx - r * .2, my - r * .09), (cx, my + r * .06),
                        (cx + r * .2, my - r * .09), (cx + r * .4, my)], np.int32)
        cv2.polylines(img, [pts], False, col, 2, AA)
    elif emo == "Contempt":
        cv2.line(img, (cx - i(r * .3), my), (cx + i(r * .3), my), col, 2, AA)
        cv2.line(img, (cx + i(r * .3), my), (cx + i(r * .44), my - i(r * .11)), col, 2, AA)
    else:
        cv2.line(img, (cx - i(r * .3), my), (cx + i(r * .3), my), col, 2, AA)


# tracking
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    _n = 0

    def __init__(self, box, kps, probs, lm=None):
        Track._n += 1
        self.id = Track._n
        self.box = np.array(box, float)
        self.kps = kps
        self.lm = None if lm is None else np.asarray(lm, np.float32)
        self.p = np.array(probs, float)
        self.cur = int(self.p.argmax())
        self.hits = 1
        self.missed = 0
        self.t_ready = None

    def update(self, box, kps, probs, lm=None, alpha=0.35):
        self.box = 0.55 * self.box + 0.45 * np.array(box, float)
        self.kps = kps
        if lm is not None:
            lm = np.asarray(lm, np.float32)
            self.lm = lm if self.lm is None else 0.5 * self.lm + 0.5 * lm
        self.p = (1 - alpha) * self.p + alpha * np.array(probs, float)
        top = int(self.p.argmax())
        if top != self.cur and self.p[top] > self.p[self.cur] + 0.08:   # hysteresis
            self.cur = top
        self.hits += 1
        self.missed = 0
        if self.t_ready is None and self.hits >= 2:
            self.t_ready = time.time()

    @property
    def emo(self):
        return EMO[self.cur]

    @property
    def conf(self):
        return float(self.p[self.cur])

    @property
    def ready(self):
        return self.hits >= 2

    @property
    def locked(self):
        return self.hits >= 10

    @property
    def area(self):
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


def update_tracks(tracks, dets):
    free = list(range(len(dets)))
    for t in tracks:
        best, bi = 0.3, -1
        for i in free:
            v = iou(t.box, dets[i]["box"])
            if v > best:
                best, bi = v, i
        if bi >= 0:
            d = dets[bi]
            free.remove(bi)
            t.update(d["box"], d["kps"], d["probs"], d.get("lm"))
        else:
            t.missed += 1
    for i in free:
        d = dets[i]
        tracks.append(Track(d["box"], d["kps"], d["probs"], d.get("lm")))
    tracks[:] = [t for t in tracks if t.missed <= 6]


#  per-face HUD
def draw_face(frame, glow, t, now, o):
    H, W = frame.shape[:2]
    acc = COL[t.emo] if t.ready else HUD2
    bx1, by1, bx2, by2 = [float(v) for v in t.box]
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 10 or bh < 10:
        return
    cx, cy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
    R = int(0.80 * max(bw, bh))

    if o["rings"]:
        draw_rings(frame, glow, (cx, cy), R, acc, now, t.id)

    if o["mesh"] and t.lm is not None:
        draw_mesh(frame, t.lm)

    if t.kps is not None and len(t.kps) >= 2:
        eyes = np.asarray(t.kps[:2], float)
        r = max(7, int(0.17 * float(np.linalg.norm(eyes[0] - eyes[1]))))
        eye_reticle(frame, glow, eyes[0], r, acc, now, 1)
        eye_reticle(frame, glow, eyes[1], r, acc, now, -1)
        if o["kps"]:
            for px, py in t.kps:
                cv2.circle(frame, (int(px), int(py)), 3, HUD2, -1, AA)
                cv2.circle(glow, (int(px), int(py)), 6, acc, -1, AA)

    # vertical scan band
    x1, y1, x2, y2 = max(0, int(bx1)), max(0, int(by1)), min(W - 1, int(bx2)), min(H - 1, int(by2))
    if x2 - x1 > 10 and y2 - y1 > 10:
        sy = int(y1 + ((now * 0.8 + t.id * 0.37) % 1.0) * (y2 - y1))
        strip = frame[sy:sy + 2, x1:x2]
        if strip.size:
            strip[:] = cv2.addWeighted(strip, 0.35, np.full_like(strip, acc), 0.65, 0)
        cv2.line(glow, (x1, sy), (x2, sy), acc, 3, AA)
        brackets(frame, (x1, y1, x2, y2), HUD2, 1, 0.12)

        # label card
        title = t.emo.upper() if t.ready else "SCANNING..."
        cw = max(220, cv2.getTextSize(title, FONT, 0.7, 1)[0][0] + 90)
        ch = 62
        cyy = y1 - ch - 10 if y1 > ch + 50 else y2 + 10
        panel(frame, x1, cyy, x1 + cw, cyy + ch, alpha=0.62, border=acc)
        cv2.rectangle(frame, (x1, cyy), (x1 + 4, cyy + ch), acc, -1)
        if t.ready:
            glyph(frame, x1 + 36, cyy + ch // 2, 20, t.emo, acc)
            txt(frame, title, (x1 + 68, cyy + 26), 0.7)
            txt(frame, "ID#%d  %.0f%%" % (t.id, t.conf * 100), (x1 + 68, cyy + 46), 0.45, acc)
            bar(frame, x1 + 68, cyy + 52, cw - 82, 3, t.conf, acc)
        else:
            txt(frame, title, (x1 + 16, cyy + 26), 0.7)
            txt(frame, "ID#%d  ANALYSING" % t.id, (x1 + 16, cyy + 46), 0.45, acc)


#  holo panel cluster
CW, CH = 470, 150


def make_cluster(t, v, a, now):
    cv = np.zeros((CH, CW, 3), np.uint8)
    bg = np.zeros((CH, CW), np.uint8)
    ec = COL[t.emo]

    def box(x1, y1, x2, y2, col=HUD):
        cv2.rectangle(bg, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(cv, (x1, y1), (x2, y2), col, 1, AA)
        for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(cv, (px, py), (px + dx * 6, py), HUD2, 1, AA)
            cv2.line(cv, (px, py), (px, py + dy * 6), HUD2, 1, AA)

    # --- eye tracker
    box(0, 0, 100, 68)
    ex, ey = 50, 32
    cv2.ellipse(cv, (ex, ey), (34, 15), 0, 200, 340, HUD2, 2, AA)
    cv2.ellipse(cv, (ex, ey), (34, 15), 0, 20, 160, HUD2, 2, AA)
    cv2.circle(cv, (ex, ey), 10, HUD, 2, AA)
    cv2.circle(cv, (ex, ey), 3, HUD2, -1, AA)
    ar = (now * 140) % 360
    cv2.ellipse(cv, (ex, ey), (16, 16), 0, ar, ar + 80, ec, 2, AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(cv, (ex + dx * 19, ey + dy * 19), (ex + dx * 25, ey + dy * 25), HUD, 1, AA)
    tiny(cv, "EYE TRACK", (6, 62))

    # --- biometric scan
    box(100, 0, 165, 68)
    fx, fy = 132, 32
    for k in range(6):
        r = 4 + k * 4
        cv2.ellipse(cv, (fx, fy), (r, int(r * 1.3)), 0, 195 + k * 7, 345 - k * 5, HUD, 1, AA)
        cv2.ellipse(cv, (fx, fy), (r, int(r * 1.3)), 0, 15 + k * 5, 165 - k * 7, HUD, 1, AA)
    sy = int(6 + ((now * 0.9) % 1.0) * 50)
    cv2.line(cv, (104, sy), (161, sy), HUD2, 1, AA)
    tiny(cv, "BIOMETRIC", (104, 62))

    # emotion bars
    box(0, 68, 165, 128)
    for i, k in enumerate(np.argsort(-t.p)[:3]):
        y = 80 + i * 18
        pv = float(t.p[k])
        c = COL[EMO[k]]
        tiny(cv, "%2.0f%%" % (pv * 100), (6, y + 7), 0.32)
        tiny(cv, EMO[k][:3].upper(), (34, y + 7), 0.32, c)
        n = int(round(pv * 10))
        for s in range(10):
            x = 60 + s * 10
            if s < n:
                cv2.rectangle(cv, (x, y), (x + 7, y + 8), c if i == 0 else HUD2, -1)
            else:
                cv2.rectangle(cv, (x, y), (x + 7, y + 8), HUDD, 1)

    # mini face mesh
    box(165, 0, 255, 128)
    E = _MESH["E"]
    if t.lm is not None and len(E):
        p = t.lm - t.lm.min(0)
        s2 = min(80.0 / max(float(p[:, 0].max()), 1), 112.0 / max(float(p[:, 1].max()), 1))
        p = p * s2
        off = np.array([165 + (90 - p[:, 0].max()) / 2, 8 + (112 - p[:, 1].max()) / 2])
        q = p + off
        cv2.polylines(cv, list(q[E].astype(np.int32)), False, HUD2, 1, AA)
        for pt in q.astype(np.int32):
            cv2.circle(cv, (int(pt[0]), int(pt[1])), 1, HUD, -1, AA)
    elif t.kps is not None:
        k5 = np.asarray(t.kps, float)
        k5 = (k5 - k5.min(0)) / max(float((k5.max(0) - k5.min(0)).max()), 1) * 70 + np.array([175, 30])
        cv2.polylines(cv, [k5.astype(np.int32)], True, HUD2, 1, AA)
        for pt in k5.astype(np.int32):
            cv2.circle(cv, (int(pt[0]), int(pt[1])), 2, HUD, -1, AA)

    # --- lock
    box(262, 0, 322, 50)
    lx, ly = 292, 27
    cv2.rectangle(cv, (lx - 11, ly - 3), (lx + 11, ly + 14), HUD2, 2)
    if t.locked:
        cv2.ellipse(cv, (lx, ly - 3), (7, 9), 0, 180, 360, HUD2, 2, AA)
    else:
        cv2.ellipse(cv, (lx + 6, ly - 7), (7, 9), 0, 180, 330, HUD2, 2, AA)
    cv2.circle(cv, (lx, ly + 6), 2, ec if t.locked else HUD, -1, AA)
    tiny(cv, "LOCKED" if t.locked else "SCAN", (270, 46), 0.3, ec if t.locked else HUD)

    # confidence dial
    box(262, 54, 322, 128)
    gx, gy, n = 292, 88, 36
    fill = int(round(t.conf * n))
    for i in range(n):
        ang = math.radians(-90 + i * 360.0 / n)
        pt = (int(gx + 25 * math.cos(ang)), int(gy + 25 * math.sin(ang)))
        if i < fill:
            cv2.circle(cv, pt, 2, ec, -1, AA)
        else:
            cv2.circle(cv, pt, 1, HUDD, -1, AA)
    s = "%.0f%%" % (t.conf * 100)
    tw = cv2.getTextSize(s, SFONT, 0.55, 1)[0][0]
    cv2.putText(cv, s, (gx - tw // 2, gy + 5), SFONT, 0.55, WHITE, 1, AA)
    tiny(cv, "CONFIDENCE", (265, 123), 0.27)

    # shield + emotion name
    box(330, 0, 470, 74)
    sx, sy0 = 360, 36
    pts = np.array([(sx - 18, sy0 - 26), (sx + 18, sy0 - 26), (sx + 18, sy0 + 2),
                    (sx, sy0 + 26), (sx - 18, sy0 + 2)], np.int32)
    cv2.fillPoly(cv, [pts], tuple(int(c * 0.25) for c in ec))
    cv2.polylines(cv, [pts], True, ec, 2, AA)
    cv2.circle(cv, (sx, sy0 - 8), 4, HUD2, -1, AA)
    cv2.line(cv, (sx, sy0 - 4), (sx, sy0 + 8), HUD2, 3, AA)
    tiny(cv, "SUBJECT STATE", (390, 14), 0.3, HUD)
    name = t.emo.upper()
    sc = 0.6
    while cv2.getTextSize(name, SFONT, sc, 1)[0][0] > 76 and sc > 0.3:
        sc -= 0.03
    cv2.putText(cv, name, (390, 38), SFONT, sc, ec, 1, AA)
    tiny(cv, "MOOD " + mood_label(v, a), (390, 58), 0.3, HUD2)

    # --- mini gauges
    box(330, 78, 470, 128)
    for i, (lab, val) in enumerate((("VAL", (v + 1) / 2), ("ARO", (a + 1) / 2), ("CLR", clarity(t.p)))):
        mx, my = 358 + i * 46, 100
        val = float(np.clip(val, 0, 1))
        cv2.ellipse(cv, (mx, my), (13, 13), 0, 135, 405, HUDD, 2, AA)
        cv2.ellipse(cv, (mx, my), (13, 13), 0, 135, 135 + 270 * val, ec, 2, AA)
        s = "%d" % round(val * 100)
        tw = cv2.getTextSize(s, SFONT, 0.3, 1)[0][0]
        cv2.putText(cv, s, (mx - tw // 2, my + 4), SFONT, 0.3, WHITE, 1, AA)
        tiny(cv, lab, (mx - 8, 124), 0.28)

    # footer
    tiny(cv, "ID#%d  %s  V%+.2f A%+.2f" % (t.id, "TRACK LOCKED" if t.locked else "ACQUIRING", v, a),
         (4, 143), 0.34, HUD)
    tiny(cv, "SIG 0x%06X" % ((int(now * 6) * 2654435761) % 0xFFFFFF), (366, 143), 0.3, HUDD)

    cv = cv2.add(cv, cv2.GaussianBlur(cv, (0, 0), 2))
    k = 1.0 if t.t_ready is None else float(np.clip((now - t.t_ready) / 0.7, 0, 1))
    if k < 1.0:
        cv = (cv * k).astype(np.uint8)
        bg = (bg * k).astype(np.uint8)
    return cv, bg


def blit(frame, cv, bg, x, y, s):
    if abs(s - 1.0) > 1e-3:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        cv = cv2.resize(cv, None, fx=s, fy=s, interpolation=interp)
        bg = cv2.resize(bg, None, fx=s, fy=s, interpolation=interp)
    h, w = cv.shape[:2]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    cx0, cy0 = x0 - x, y0 - y
    roi = frame[y0:y1, x0:x1]
    c = cv[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)]
    b = bg[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)]
    a = (b.astype(np.float32) / 255.0 * 0.62)[..., None]
    roi[:] = (roi * (1 - a) + np.array((22, 14, 8), np.float32) * a).astype(np.uint8)
    roi[:] = cv2.add(roi, c)


def place_cluster(frame, t, vis_w, now, va):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in t.box]
    bw, bh = x2 - x1, y2 - y1
    s = float(np.clip(bh / 240.0, 0.6, 1.25))
    room_r, room_l = vis_w - (x2 + 14), x1 - 14
    side, room = (1, room_r) if room_r >= room_l else (-1, room_l)
    s = min(s, max(room, 1) / CW)
    if s < 0.5:
        return
    cv, bg = make_cluster(t, va[0], va[1], now)
    w, h = int(CW * s), int(CH * s)
    x = int(x2 + 14) if side == 1 else int(x1 - 14 - w)
    y = int(np.clip(y1 + 0.10 * bh, 46, max(46, H - h - 10)))
    blit(frame, cv, bg, x, y, s)
    # connector from face edge to the cluster
    edge = (int(x2), int(y + h * 0.5)) if side == 1 else (int(x1), int(y + h * 0.5))
    tip = (x, int(y + h * 0.5)) if side == 1 else (x + w, int(y + h * 0.5))
    cv2.line(frame, edge, tip, HUD, 1, AA)
    cv2.circle(frame, edge, 3, HUD2, -1, AA)


# screen level HUD
def draw_idle(frame, glow, now, vis_w):
    H = frame.shape[0]
    c = (vis_w // 2, H // 2)
    R = int(min(vis_w, H) * 0.26)
    draw_rings(frame, glow, c, R, HUD2, now, 0)
    for img in (glow, frame):
        m = 3 if img is glow else 1
        for i in range(36):
            a = i * 10 + now * 20
            cv2.ellipse(img, c, (int(R * 0.62), int(R * 0.82)), 0, a, a + 5, HUD, m, AA)
    pulse = 0.5 + 0.5 * math.sin(now * 3)
    col = tuple(int(v * (0.45 + 0.55 * pulse)) for v in HUD2)
    s = "SEARCHING FOR FACE"
    w = cv2.getTextSize(s, FONT, 0.7, 1)[0][0]
    txt(frame, s, (c[0] - w // 2, c[1] + int(R * 1.24) + 60), 0.7, col)
    s2 = "ALIGN YOUR FACE WITH THE CAMERA"
    w2 = cv2.getTextSize(s2, FONT, 0.42, 1)[0][0]
    txt(frame, s2, (c[0] - w2 // 2, c[1] + int(R * 1.24) + 86), 0.42, DIM)


def topbar(frame, st, now, vis_w, nfaces):
    H, W = frame.shape[:2]
    panel(frame, 0, 0, vis_w, 34, alpha=0.5)
    cv2.line(frame, (0, 34), (vis_w, 34), HUD, 1, AA)
    txt(frame, "EMOTION VISION  //  NEURAL FACIAL ANALYSIS", (16, 23), 0.5, HUD)
    right = "%s   FPS %.0f   FACES %d" % (datetime.now().strftime("%H:%M:%S"), st["fps"], nfaces)
    tw = cv2.getTextSize(right, FONT, 0.45, 1)[0][0]
    txt(frame, right, (vis_w - tw - 16, 23), 0.45, HUD2)
    brackets(frame, (8, 42, vis_w - 8, H - 8), HUD, 2, 0.05)
    pulse = 0.5 + 0.5 * math.sin(now * 4)
    cv2.circle(frame, (24, H - 24), 5, tuple(int(c * (0.4 + 0.6 * pulse)) for c in (120, 255, 160)), -1, AA)
    txt(frame, "SYSTEM ONLINE", (38, H - 19), 0.42, (120, 255, 160))
    if st["rec"]:
        if int(now * 2) % 2 == 0:
            cv2.circle(frame, (vis_w - 90, H - 24), 6, (60, 60, 255), -1, AA)
        txt(frame, "REC", (vis_w - 76, H - 19), 0.45, (60, 60, 255))


def sidebar(frame, st, now):
    H, W = frame.shape[:2]
    pw = st["pw"]
    x0 = W - pw
    pad = 18
    iw = pw - 2 * pad
    panel(frame, x0, 0, W, H, alpha=0.58, border=(90, 70, 40))
    txt(frame, "EMOTION VISION", (x0 + pad, 36), 0.78, HUD)
    cv2.line(frame, (x0 + pad, 48), (W - pad, 48), HUD, 1, AA)
    txt(frame, "SCRFD + HSEMOTION | %s | %.0f FPS" % (st["device"], st["fps"]), (x0 + pad, 68), 0.38, DIM)

    t = st["primary"]
    y = 84
    if t is None or not t.ready:
        txt(frame, "NO FACE LOCKED", (x0 + pad, y + 60), 0.7, DIM)
        txt(frame, "look at the camera", (x0 + pad, y + 86), 0.45, DIM)
    else:
        col = COL[t.emo]
        glyph(frame, x0 + pad + 46, y + 56, 40, t.emo, col)
        txt(frame, t.emo.upper(), (x0 + pad + 108, y + 46), 0.85, col, 1)
        txt(frame, "CONFIDENCE %.1f%%" % (t.conf * 100), (x0 + pad + 108, y + 72), 0.45, WHITE)
        bar(frame, x0 + pad + 108, y + 80, iw - 108, 5, t.conf, col)
        y += 118
        txt(frame, "EXPRESSION SPECTRUM", (x0 + pad, y), 0.5, HUD)
        y += 14
        for i, e in enumerate(EMO):
            top = i == t.cur
            txt(frame, e[:9], (x0 + pad, y + 12), 0.4, COL[e] if top else (175, 175, 175))
            bar(frame, x0 + pad + 84, y + 4, iw - 84 - 44, 8, float(t.p[i]), COL[e])
            txt(frame, "%3.0f%%" % (t.p[i] * 100), (W - pad - 38, y + 12), 0.38, WHITE if top else DIM)
            y += 19
        y += 12
        txt(frame, "VALENCE / AROUSAL", (x0 + pad, y), 0.5, HUD)
        y += 10
        S = 112
        px, py = x0 + pad, y
        cv2.rectangle(frame, (px, py), (px + S, py + S), (110, 90, 60), 1, AA)
        cv2.line(frame, (px + S // 2, py), (px + S // 2, py + S), (70, 60, 50), 1)
        cv2.line(frame, (px, py + S // 2), (px + S, py + S // 2), (70, 60, 50), 1)
        hist = list(st["hist"])[-40:]
        for k, (hv, ha, hi) in enumerate(hist):
            hx = px + int((np.clip(hv, -1, 1) + 1) / 2 * S)
            hy = py + int((1 - (np.clip(ha, -1, 1) + 1) / 2) * S)
            f = (k + 1) / max(1, len(hist))
            cv2.circle(frame, (hx, hy), 2, tuple(int(c * f) for c in COL[EMO[hi]]), -1, AA)
        v, a = st["va"]
        dx = px + int((np.clip(v, -1, 1) + 1) / 2 * S)
        dy = py + int((1 - (np.clip(a, -1, 1) + 1) / 2) * S)
        cv2.circle(frame, (dx, dy), 7, col, 2, AA)
        cv2.circle(frame, (dx, dy), 3, col, -1, AA)
        tx = px + S + 16
        txt(frame, "VALENCE %+.2f" % v, (tx, py + 22), 0.45, WHITE)
        txt(frame, "AROUSAL %+.2f" % a, (tx, py + 44), 0.45, WHITE)
        txt(frame, "MOOD", (tx, py + 74), 0.4, DIM)
        txt(frame, mood_label(v, a), (tx, py + 98), 0.7, col)
        y += S + 22
        txt(frame, "TIMELINE", (x0 + pad, y), 0.5, HUD)
        y += 10
        th_ = 56
        panel(frame, x0 + pad, y, x0 + pad + iw, y + th_ + 10, alpha=0.35, border=(80, 65, 45))
        hist = list(st["hist"])
        n = st["hist"].maxlen
        cw_ = iw / n
        cv2.line(frame, (x0 + pad, y + th_ // 2), (x0 + pad + iw, y + th_ // 2), (70, 60, 50), 1)
        pts = []
        for k, (hv, ha, hi) in enumerate(hist):
            xx = x0 + pad + int((n - len(hist) + k) * cw_)
            yy = y + int((1 - (np.clip(hv, -1, 1) + 1) / 2) * (th_ - 4)) + 2
            pts.append((xx, yy))
            cv2.rectangle(frame, (xx, y + th_ + 2), (xx + max(1, int(cw_) + 1), y + th_ + 8), COL[EMO[hi]], -1)
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, np.int32)], False, WHITE, 1, AA)
        y += th_ + 28
        txt(frame, "SESSION", (x0 + pad, y), 0.5, HUD)
        y += 10
        tot = sum(st["stats"].values())
        if tot > 0:
            xx = x0 + pad
            for e in EMO:
                w_ = int(iw * st["stats"][e] / tot)
                if w_ > 0:
                    cv2.rectangle(frame, (xx, y), (xx + w_, y + 10), COL[e], -1)
                    xx += w_
            xx = x0 + pad
            for e, sec in st["stats"].most_common(3):
                txt(frame, "%s %.0f%%" % (e[:4].upper(), 100 * sec / tot), (xx, y + 28), 0.38, COL[e])
                xx += 96
    txt(frame, "q quit s shot h bar p panels m mesh k pts g glow c rings r rec f full",
        (x0 + 10, H - 12), 0.3, DIM)


#boot sequence
def draw_boot(frame, p, now, device):
    H, W = frame.shape[:2]
    sweep = int(H * min(1.0, p / 0.85))
    if sweep < H:
        below = frame[sweep:]
        below[:] = (below * 0.22).astype(np.uint8)
        cv2.line(frame, (0, sweep), (W, sweep), HUD2, 2, AA)
        cv2.line(frame, (0, sweep + 3), (W, sweep + 3), HUD, 1, AA)

    e = 1 - (1 - min(1.0, p / 0.55)) ** 3
    fade = 1.0 if p < 0.88 else max(0.0, 1 - (p - 0.88) / 0.12)
    R = int(min(W, H) * 0.27 * e * (1.0 if p < 0.88 else 1 + (p - 0.88) * 4))
    lay = np.zeros_like(frame)
    glow = np.zeros_like(frame)
    if R > 10:
        draw_rings(lay, glow, (W // 2, H // 2), R, HUD2, now * 2.2, 0)
    lay = add_glow(lay, glow)
    frame[:] = cv2.add(frame, (lay * fade).astype(np.uint8))

    if p < 0.97:
        title = "EMOTION VISION"
        tw = cv2.getTextSize(title, FONT, 1.3, 2)[0][0]
        a = min(1.0, p / 0.3) * fade
        txt(frame, title, (W // 2 - tw // 2, H // 2 + 8), 1.3, tuple(int(c * a) for c in HUD2), 2)
        lines = [(0.06, "BOOTING NEURAL CORE ................ OK"),
                 (0.20, "LOADING SCRFD FACE DETECTOR ........ OK"),
                 (0.34, "LOADING HSEMOTION NETWORK .......... OK"),
                 (0.48, "CALIBRATING OPTICAL SENSOR ......... OK"),
                 (0.62, "ACCELERATOR  %-6s ................. OK" % device),
                 (0.78, "SYSTEM ONLINE")]
        y0 = H - 40 - 20 * len(lines)
        shown = [l for l in lines if p >= l[0]]
        for i, (_, s) in enumerate(shown):
            last = i == len(shown) - 1
            col = (120, 255, 160) if (last and s == "SYSTEM ONLINE") else HUD2
            if last and int(now * 6) % 2 == 0:
                s += " _"
            txt(frame, s, (36, y0 + i * 20), 0.45, col)
        bw = min(420, W - 80)
        bx, by = W // 2 - bw // 2, H - 56
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 6), HUDD, 1, AA)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * p), by + 6), HUD, -1)
        tiny(frame, "%d%%" % int(p * 100), (bx + bw + 10, by + 7), 0.4, HUD2)


#  render + main
def render(frame, tracks, st, now, o, fx8):
    H, W = frame.shape[:2]
    frame = cv2.multiply(frame, fx8, scale=1 / 255.0)
    pw = st["pw"] if (o["hud"] and W > 900) else 0
    vis = W - pw
    glow = np.zeros_like(frame)
    live = [t for t in tracks if t.missed <= 2]
    for t in live:
        draw_face(frame, glow, t, now, o)
    if not live:
        draw_idle(frame, glow, now, vis)
    if o["glow"]:
        frame = add_glow(frame, glow)
    prim = st["primary"]
    if o["panels"] and prim is not None and prim.ready and prim in live:
        place_cluster(frame, prim, vis, now, st["va"])
    if pw:
        sidebar(frame, st, now)
    topbar(frame, st, now, vis, len(live))
    return frame


def main():
    ap = argparse.ArgumentParser(description="EMOTION VISION v2 - holographic emotion recognition")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--det", type=int, default=640, help="detector size (320 = faster)")
    ap.add_argument("--pack", default="buffalo_l", help="insightface pack: buffalo_l / buffalo_s")
    ap.add_argument("--emo-model", default="enet_b0_8_best_afew",
                    help="enet_b0_8_best_afew | enet_b0_8_best_vgaf")
    ap.add_argument("--skip", type=int, default=2, help="run the networks every N frames")
    ap.add_argument("--max-faces", type=int, default=6)
    ap.add_argument("--no-mesh", action="store_true", help="skip the 106-point landmark model (faster)")
    ap.add_argument("--no-boot", action="store_true", help="skip the boot animation")
    args = ap.parse_args()

    print("Loading models (first run downloads weights) ...")
    det = Detector(args.det, args.pack, mesh=not args.no_mesh)
    net = EmotionNet(args.emo_model)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera %d" % args.cam)

    LOGS.mkdir(exist_ok=True)
    SHOTS.mkdir(exist_ok=True)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, args.width, args.height)

    tracks = []
    hist = deque(maxlen=180)
    stats = Counter()
    st = dict(pw=350, device=det.device, fps=0.0, rec=False, primary=None, va=(0.0, 0.0),
              hist=hist, stats=stats)
    o = dict(hud=True, kps=True, glow=True, mesh=not args.no_mesh, panels=True, rings=True)
    full = False
    rec_f = rec_w = None
    last_hist = last_rec = 0.0
    fps, tprev, i, fx8 = 0.0, time.time(), 0, None
    boot_t0 = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        if fx8 is None or fx8.shape[:2] != (H, W):
            fx8 = make_fx(H, W)
        if boot_t0 is None:
            boot_t0 = time.time()

        if i % args.skip == 0:
            faces = sorted(det(frame), key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            dets = []
            for f in faces[:args.max_faces]:
                rgb = crop_face(frame, f.bbox)
                if rgb is None:
                    continue
                dets.append(dict(box=f.bbox, kps=f.kps, probs=net(rgb),
                                 lm=getattr(f, "landmark_2d_106", None)))
            update_tracks(tracks, dets)
        i += 1

        now = time.time()
        dt = now - tprev
        tprev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt

        live = [t for t in tracks if t.missed <= 2]
        prim = max(live, key=lambda t: t.area) if live else None
        st["primary"] = prim
        if prim is not None and prim.ready:
            v, a = valence_arousal(prim.p)
            st["va"] = (v, a)
            stats[prim.emo] += dt
            if now - last_hist > 0.1:
                hist.append((v, a, prim.cur))
                last_hist = now
            if rec_w and now - last_rec > 0.5:
                rec_w.writerow([datetime.now().isoformat(timespec="milliseconds"), prim.id, prim.emo,
                                "%.3f" % prim.conf, "%.3f" % v, "%.3f" % a] + ["%.3f" % x for x in prim.p])
                rec_f.flush()
                last_rec = now
        st["fps"] = fps
        st["rec"] = rec_w is not None

        frame = render(frame, tracks, st, now, o, fx8)
        if not args.no_boot:
            bp = (now - boot_t0) / BOOT_T
            if bp < 1.0:
                draw_boot(frame, bp, now, det.device)

        cv2.imshow(WIN, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            p = SHOTS / ("emotion_%s.png" % datetime.now().strftime("%Y%m%d_%H%M%S"))
            cv2.imwrite(str(p), frame)
            print("saved", p)
        elif k == ord("h"):
            o["hud"] = not o["hud"]
        elif k == ord("k"):
            o["kps"] = not o["kps"]
        elif k == ord("g"):
            o["glow"] = not o["glow"]
        elif k == ord("m"):
            o["mesh"] = not o["mesh"]
        elif k == ord("p"):
            o["panels"] = not o["panels"]
        elif k == ord("c"):
            o["rings"] = not o["rings"]
        elif k == ord("f"):
            full = not full
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        elif k == ord("r"):
            if rec_w is None:
                p = LOGS / ("emotions_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S"))
                rec_f = open(p, "w", newline="")
                rec_w = csv.writer(rec_f)
                rec_w.writerow(["timestamp", "track", "emotion", "confidence", "valence", "arousal"] + EMO)
                print("recording ->", p)
            else:
                rec_f.close()
                rec_f = rec_w = None
                print("recording stopped")

    cap.release()
    if rec_f:
        rec_f.close()
    cv2.destroyAllWindows()
    tot = sum(stats.values())
    if tot > 0:
        print("\nSession summary (%.0fs with a face):" % tot)
        for e, s in stats.most_common():
            print("  %-10s %5.1f%%  (%.1fs)" % (e, 100 * s / tot, s))


if __name__ == "__main__":
    main()