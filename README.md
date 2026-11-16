# Live Object Detection with YOLO

Live object detection from your webcam using a pre-trained YOLO11 model and OpenCV. It captures video from your webcam, detects objects, and displays the results with bounding boxes and class labels.

Two equivalent versions are provided:

- **`live_object_detection.py`** — the script. Best for actually running the live detector.
- **`live_object_detection.ipynb`** — the notebook. Same code, handy for stepping through it or as a demo.


## Setup

```bash
pip install -r requirements.txt
```

On the first run, the `yolo11s.pt` model weights are downloaded automatically into a `yolo-Weights` subdirectory.

Only detections at or above a confidence threshold (`CONF_THRESHOLD`, default `0.5`) are shown, and each label includes its confidence score. Raise the threshold to suppress shaky guesses, or lower it if real objects are being missed.


## Usage

**Script:**

```bash
python live_object_detection.py
```

**Notebook:** open `live_object_detection.ipynb` in Jupyter Notebook or JupyterLab and run the cells sequentially.

Either way, a window opens showing the live webcam feed with object detections. **Press 'q' to quit.**
