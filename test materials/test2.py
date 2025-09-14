import cv2
from ultralytics import YOLO
import numpy as np
import time
import sys

VIDEO_PATH = "videos//distance.mp4"
MODEL_WEIGHTS = "models//fall//fall.pt"
H_REAL = 1.70
FOCAL_MM = 4.25
SENSOR_WIDTH_MM = 6.40
CONF_THRESHOLD = 0.25
SMOOTH_ALPHA = 0.2
DRAW_STRATEGY = "tallest"
DEBUG = False

SHOW_WINDOW = True
WINDOW_NAME = "Distance with box (ESC to quit)"

# ---- Proximity alert config ----
MIN_SAFE_DISTANCE_M = 1.0      # threshold (meters)
BEEP_ENABLED = True            # turn sound on/off
BEEP_COOLDOWN_S = 0.40         # minimum time between beeps when too close
BEEP_FREQ_HZ = 1000            # tone frequency (if winsound/simpleaudio available)
BEEP_DURATION_MS = 180         # tone length (ms) on Windows; ~duration on others
# --------------------------------


def smooth(prev, new, alpha=0.2):
    return new if prev is None else (1 - alpha) * prev + alpha * new

def infer_person_class_id(model):
    names = getattr(model, "names", None)
    if names is None: return None
    if isinstance(names, dict):
        for k, v in names.items():
            if isinstance(v, str) and "person" in v.lower():
                return int(k)
    elif isinstance(names, (list, tuple)):
        for idx, name in enumerate(names):
            if isinstance(name, str) and "person" in name.lower():
                return int(idx)
    return None

def gather_candidates(results, f_px, H_REAL, accept_class_id=None):
    out = []
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        boxes = r.boxes
        xyxy = boxes.xyxy
        confs = boxes.conf
        clss  = boxes.cls
        if xyxy is None or confs is None or clss is None:
            continue
        xyxy = xyxy.cpu().numpy()
        confs = confs.cpu().numpy()
        clss  = clss.cpu().numpy()
        for b, conf, cls_id in zip(xyxy, confs, clss):
            if accept_class_id is not None and int(cls_id) != int(accept_class_id):
                continue
            x1, y1, x2, y2 = b.astype(int)
            h_px = max(1.0, float(y2 - y1))
            D = (f_px * H_REAL) / h_px
            out.append((D, float(conf), (x1, y1, x2, y2), h_px))
    return out

# ---- Cross-platform beep helper ----
def _make_beeper():
    # Try Windows winsound first
    try:
        import winsound
        def _beep(freq=BEEP_FREQ_HZ, duration_ms=BEEP_DURATION_MS):
            try:
                winsound.Beep(int(freq), int(duration_ms))
            except RuntimeError:
                pass
        return _beep
    except Exception:
        pass

    # Try simpleaudio (pip install simpleaudio)
    try:
        import simpleaudio as sa
        def _beep(freq=BEEP_FREQ_HZ, duration_ms=BEEP_DURATION_MS):
            fs = 44100
            t = duration_ms / 1000.0
            samples = (np.sin(2 * np.pi * np.arange(int(fs * t)) * (freq / fs))).astype(np.float32)
            audio = (samples * 32767).astype(np.int16)
            try:
                sa.play_buffer(audio, 1, 2, fs)  # non-blocking
            except Exception:
                pass
        return _beep
    except Exception:
        pass

    # Fallback: terminal bell (may or may not make a sound depending on terminal/OS)
    def _beep(freq=BEEP_FREQ_HZ, duration_ms=BEEP_DURATION_MS):
        try:
            print('\a', end='', flush=True)
        except Exception:
            pass
    return _beep

BEEP = _make_beeper()

def main():
    model = YOLO(MODEL_WEIGHTS)
    person_class_id = infer_person_class_id(model)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video/camera")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame")
    img_h, img_w = frame.shape[:2]

    if SENSOR_WIDTH_MM <= 0:
        raise ValueError("SENSOR_WIDTH_MM must be > 0")
    f_px = FOCAL_MM * (img_w / SENSOR_WIDTH_MM)

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Taller window:
        cv2.resizeWindow(WINDOW_NAME, 960, 900)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    distance_smooth = None
    last_beep_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if person_class_id is not None:
            results = model(frame, verbose=False, conf=CONF_THRESHOLD, classes=[person_class_id])
            candidates = gather_candidates(results, f_px, H_REAL, accept_class_id=person_class_id)
            filtered = True
        else:
            results = model(frame, verbose=False, conf=CONF_THRESHOLD)
            candidates = gather_candidates(results, f_px, H_REAL, accept_class_id=None)
            filtered = False

        if not candidates and filtered:
            results = model(frame, verbose=False, conf=CONF_THRESHOLD)
            candidates = gather_candidates(results, f_px, H_REAL, accept_class_id=None)

        choice = None
        if candidates:
            if DRAW_STRATEGY == "confident":
                candidates.sort(key=lambda t: t[1], reverse=True)
            else:
                candidates.sort(key=lambda t: t[3], reverse=True)
            choice = candidates[0]

        if choice:
            D, conf, (x1, y1, x2, y2), h_px = choice
            distance_smooth = smooth(distance_smooth, D, SMOOTH_ALPHA)

            # Proximity logic
            too_close = (distance_smooth is not None and distance_smooth < MIN_SAFE_DISTANCE_M)

            # Box color: red if too close, else green
            color = (0, 0, 255) if too_close else (0, 255, 0)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"D: {distance_smooth:.2f} m"
            y_text = y1 - 10 if y1 - 10 > 20 else y1 + 25
            cv2.putText(frame, label, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            # Console log
            print(f"{distance_smooth:.3f}")

            # Beep if too close (rate-limited)
            if BEEP_ENABLED and too_close:
                now = time.time()
                if now - last_beep_time >= BEEP_COOLDOWN_S:
                    BEEP(BEEP_FREQ_HZ, BEEP_DURATION_MS)
                    last_beep_time = now
        else:
            if SHOW_WINDOW:
                cv2.putText(frame, "No person detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        if SHOW_WINDOW:
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
        else:
            time.sleep(0.001)

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
