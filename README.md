# SunSight: leakage-aware hotspot detection in UAV thermography of PV arrays

Spatial-block split manifest, trained detector and core library for the article

> M. N. Sheik, *Leakage-aware evaluation of deep-learning hotspot detection and review prioritisation in UAV
> thermography of photovoltaic arrays*.

## Contents

| Path | Content |
|---|---|
| `data/splits/split_manifest.csv` | Subset (training, validation, test), repeated-shot group, 50 m spatial block and session of every frame |
| `data/splits/split_report.json` | Leakage checks of the split |
| `models/sunsight_hotspot_best.pt` | YOLO11s detector trained at 960 px for one class (`hot_spot`), with its model card and test evaluation summary |
| `src/` | Image and drone metadata, ground view-point estimation, detection evaluation (greedy matching at IoU 0.5, all-point AP) and Review Priority Index |
| `config/settings.yaml` | Default settings used by `src/` |

## Data

The images are not redistributed here. The dataset is public (Roboflow Universe, Public Domain):
<https://universe.roboflow.com/misaka-cpn7n/-yolde/dataset/1>. The file names in the split manifest match the
dataset's file names.

## Use

Python 3.11 or later.

```bash
python -m pip install -r requirements.txt
```

```python
from ultralytics import YOLO

model = YOLO("models/sunsight_hotspot_best.pt")
result = model.predict("frame.jpg", imgsz=960, conf=0.30)  # 0.30: operating threshold selected on validation data
```
