```markdown
# Cosmic Coders – Vision Detections (Flask + YOLO)

Tiny Flask app that runs multiple YOLO-based demos (fire, weapons, people, fall, distance) and streams results to the browser via MJPEG.

## Structure
```

.
├─ app.py                 # Flask server + detector threads + MJPEG routes
├─ templates/
│  ├─ index.html          # Landing page (cards → open live views)
│  └─ live-detection.html # Live stream page (auto-starts selected detector)
├─ static/
│  └─ images/             # Card preview images (fire.jpg, weapon.jpg, people.jpg, fall.jpg, distance.jpg)
├─ models/                # Your YOLO weights (subfolders per demo)
└─ videos/                # Demo input videos (e.g., fire.mp4, weapons.mp4, distance.mp4)

````

## Quickstart
```bash
# 1) Create venv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install flask ultralytics opencv-python numpy simpleaudio

# 3) Run
python app.py
# Open: http://127.0.0.1:5000
````

## Usage

* Click a card on the homepage → opens `/live/<kind>` and auto-starts that detector.
* Streams are available at `/stream/<kind>`.
* Navigating away calls `/api/stop/<kind>` to stop the running detector.

## Notes

* Put images in `static/images/` and reference with `{{ url_for('static', filename='images/<file>') }}`.
* Place model weights under `models/...` and sample videos under `videos/...`.
* On Windows, beeps use `winsound`; on other OS, `simpleaudio` or a terminal bell fallback.

## Demos

* `fire`, `weapons`, `people`, `fall`, `distance` (each wired in `app.py`).

```
```
