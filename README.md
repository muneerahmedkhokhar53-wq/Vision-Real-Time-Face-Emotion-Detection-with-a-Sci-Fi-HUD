<div align="center">

# 👁️ EMOTION VISION

**Real-time, holographic facial emotion recognition, built with computer vision and deep learning.**

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-005CED?logo=onnx&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

![Emotion Vision demo](assets/demo.gif)

[Full demo video](assets/demo.mp4)

</div>

## What it does

Emotion Vision reads your webcam, finds every face, classifies each expression and shows the result in a sci-fi HUD. Everything runs locally on your machine. No frames are uploaded anywhere.

| Sadness | Happiness | Fear |
|:---:|:---:|:---:|
| ![Sadness](assets/sadness.jpg) | ![Happiness](assets/happiness.jpg) | ![Fear](assets/fear.jpg) |

## Features

- **Multi-face tracking** with stable IDs across frames (IoU tracker) and smoothed predictions
- **8 emotions** with live probabilities: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise
- **Valence / arousal** estimate plotted on an emotion map, with a mood label (e.g. DOWN, ALERT, EXCITED, TENSE)
- **106-point face mesh** drawn as a Delaunay wireframe, with glowing eye reticles
- **Live dashboard**: expression spectrum, rolling timeline and session totals
- **Cinematic boot sequence** and animated HUD rings
- **CSV recording** of emotions, confidence and valence/arousal for later analysis
- **GPU acceleration** (CUDA or CoreML) when available, with automatic CPU fallback

## How it works

```
Webcam frame ──► SCRFD face detector (InsightFace) ──► face crop
                       │                                  │
                106-point landmarks                HSEmotion (EfficientNet-B0, ONNX)
                       │                                  │
                 face mesh overlay            8-class emotion probabilities
                                                          │
                              IoU tracker + smoothing ──► valence/arousal ──► HUD
```

- **Detection and landmarks:** [InsightFace](https://github.com/deepinsight/insightface) (SCRFD, `buffalo_l` pack)
- **Emotion classification:** [HSEmotion](https://github.com/av-savchenko/hsemotion-onnx) EfficientNet-B0 trained on AffectNet and AFEW
- **Valence / arousal:** computed by mapping the emotion probabilities onto circumplex-model anchor points. It is an approximation, not a direct measurement.

## Quick start

```bash
git clone https://github.com/muneerahmedkhokhar53-wq/Vision-Real-Time-Face-Emotion-Detection-with-a-Sci-Fi-HUD.git
cd Vision-Real-Time-Face-Emotion-Detection-with-a-Sci-Fi-HUD

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
python emotion_vision.py
```

The first run downloads the model weights automatically (needs internet once).

## Controls

| Key | Action | Key | Action |
|:---:|---|:---:|---|
| `q` | Quit | `h` | Toggle HUD |
| `s` | Save screenshot to `shots/` | `k` | Toggle keypoints |
| `r` | Start/stop CSV recording to `logs/` | `g` | Toggle glow |
| `f` | Fullscreen | `m` | Toggle face mesh |
| `p` | Toggle side panels | `c` | Toggle rings |

## Options

```bash
python emotion_vision.py --cam 1 --det 320 --no-mesh --no-boot
```

| Flag | Default | Description |
|---|---|---|
| `--cam` | `0` | Camera index |
| `--width`, `--height` | `1280`, `720` | Capture resolution |
| `--det` | `640` | Detector input size (`320` is faster) |
| `--pack` | `buffalo_l` | InsightFace pack (`buffalo_l` or `buffalo_s`) |
| `--emo-model` | `enet_b0_8_best_afew` | Emotion model (`enet_b0_8_best_afew` or `enet_b0_8_best_vgaf`) |
| `--skip` | `2` | Run the networks every N frames |
| `--max-faces` | `6` | Maximum faces tracked |
| `--no-mesh` | off | Skip the 106-point landmark model (faster) |
| `--no-boot` | off | Skip the boot animation |

**Slow on a CPU-only laptop?** Try `--det 320 --no-mesh --skip 3`.

## Recorded data

Press `r` to write `logs/emotions_<timestamp>.csv` with these columns:
`timestamp, track, emotion, confidence, valence, arousal` plus one probability column per emotion. A session summary prints when you quit.

## Project structure

```
emotion-vision/
├── emotion_vision.py     # the whole application
├── requirements.txt
├── assets/               # screenshots, demo GIF and video
├── logs/                 # CSV recordings (created at runtime, git-ignored)
├── shots/                # screenshots (created at runtime, git-ignored)
└── .github/              # CI and issue templates
```

## Limitations and responsible use

- Facial expressions do not reliably reveal what someone actually feels. Treat the output as a visual demo of an expression classifier, not a psychological assessment.
- Accuracy depends on lighting, camera angle, occlusion and the person's face. Models can be biased across demographics.
- Do not use it to monitor, judge or make decisions about people without their informed consent.

## Roadmap

- [ ] Split the single file into a package (`detector`, `emotion`, `hud`, `tracking`)
- [ ] Video-file and image input
- [ ] Exportable session report (charts and summary)
- [ ] Optional web dashboard

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)

## Acknowledgements

[InsightFace](https://github.com/deepinsight/insightface), [HSEmotion](https://github.com/av-savchenko/hsemotion-onnx), [OpenCV](https://opencv.org/), [ONNX Runtime](https://onnxruntime.ai/)
