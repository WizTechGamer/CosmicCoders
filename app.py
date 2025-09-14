# app.py
from flask import Flask, render_template, Response, jsonify, request
import threading, time, os, cv2, numpy as np
from ultralytics import YOLO

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static"
)

# Fixed frame size to match frontend .frame CSS
FRAME_WIDTH, FRAME_HEIGHT = 1200, 700


# ---------- Per-stream state ----------
class StreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.running = False
        self.error = None
        self.viewers = 0          # <— how many clients currently connected to /stream/<kind>
        self.last_seen = 0.0      # last time a viewer was present

STREAMS = {k: StreamState() for k in ("fire","weapons","people","fall","distance")}

# ---------- Paths (edit as needed) ----------
MODELS = {
    "fire":     "models/fire/fire.pt",
    "weapons":  "models/gun/gun.pt",
    "people":   "models/people/people.pt",
    "fall":     "models/fall/fall.pt",
    "distance": "models/fall/fall.pt",
}
VIDEOS = {
    "fire":     "videos/fire3.mp4",
    "weapons":  "videos/rpg.mp4",
    "people":   "videos/people.mp4",
    "fall":     "videos/fall.mp4",
    "distance": "videos/distance.mp4",
}

# ---------- Beep ----------
def _make_beeper(freq=1000, duration_ms=180):
    try:
        import winsound
        def _beep():
            try: winsound.Beep(int(freq), int(duration_ms))
            except RuntimeError: pass
        return _beep
    except Exception:
        pass
    try:
        import simpleaudio as sa
        def _beep():
            fs=44100; t=duration_ms/1000.0
            samples=(np.sin(2*np.pi*np.arange(int(fs*t))*(freq/fs))).astype(np.float32)
            audio=(samples*32767).astype(np.int16)
            try: sa.play_buffer(audio,1,2,fs)
            except Exception: pass
        return _beep
    except Exception:
        pass
    def _beep():
        try: print('\a', end='', flush=True)
        except Exception: pass
    return _beep
BEEP = _make_beeper()

# ---------- Model cache ----------
_model_cache, _model_lock = {}, threading.Lock()
def get_model(kind):
    path = MODELS.get(kind)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model missing for '{kind}': {path}")
    with _model_lock:
        if path not in _model_cache:
            _model_cache[path] = YOLO(path)
        return _model_cache[path]

# ---------- JPEG helpers ----------
def jpg_frame(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:  # extremely rare
        return (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n")
    return (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

def text_frame(msg, w=960, h=540, color=(50,50,50)):
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(img, msg, (30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return jpg_frame(img)

# ---------- Base detector ----------
class BaseDetector(threading.Thread):
    def __init__(self, kind, video_path, conf=0.5):
        super().__init__(daemon=True)
        self.kind = kind
        self.video_path = video_path
        self.conf = conf
        self.stop_flag = False

    def annotate(self, frame):
        model = get_model(self.kind)
        res = model(frame, conf=self.conf)
        return res[0].plot(), res

    def run(self):
        st = STREAMS[self.kind]
        st.running, st.error = True, None
        st.last_seen = time.time()

        if not self.video_path or not os.path.exists(self.video_path):
            st.error = f"Missing video: {self.video_path}"
            while not self.stop_flag:
                with st.lock: st.latest_jpeg = text_frame(st.error, color=(0,0,255))
                time.sleep(0.3)
            st.running = False
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            st.error = f"Could not open video: {self.video_path}"
            while not self.stop_flag:
                with st.lock: st.latest_jpeg = text_frame(st.error, color=(0,0,255))
                time.sleep(0.3)
            st.running = False
            return

        try:
            while not self.stop_flag:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                annotated, _ = self.annotate(frame)

                annotated = cv2.resize(annotated, (FRAME_WIDTH, FRAME_HEIGHT))

                with st.lock:
                    st.latest_jpeg = jpg_frame(annotated)

                time.sleep(0.02)
        finally:
            cap.release()
            st.running = False


class FireDetector(BaseDetector):
    def __init__(self, video_path, det_threshold=3, conf=0.5):
        super().__init__("fire", video_path, conf)
        self.det_threshold = det_threshold
    def annotate(self, frame):
        model = get_model("fire")
        res = model(frame, conf=self.conf)
        n = len(res[0].boxes) if getattr(res[0], "boxes", None) is not None else 0
        if n >= self.det_threshold:
            BEEP()
        return res[0].plot(), res

class WeaponsDetector(BaseDetector):
    def __init__(self, video_path, conf=0.6):
        super().__init__("weapons", video_path, conf)

class PeopleDetector(BaseDetector):
    def __init__(self, video_path, conf=0.4):
        super().__init__("people", video_path, conf)
    def annotate(self, frame):
        model = get_model("people")
        names = getattr(model, "names", {})
        person_ids = [int(k) for k,v in (names.items() if isinstance(names,dict) else []) if isinstance(v,str) and "person" in v.lower()]
        res = model(frame, conf=self.conf, classes=person_ids or None)
        return res[0].plot(), res

class FallDetector(BaseDetector):
    def __init__(self, video_path, conf=0.4):
        super().__init__("fall", video_path, conf)

class DistancePeopleDetector(BaseDetector):
    def __init__(self, video_path, conf=0.25, H_REAL=1.70, FOCAL_MM=4.25, SENSOR_WIDTH_MM=6.40,
                 smooth_alpha=0.2, draw_strategy="tallest", min_safe_distance_m=1.0, beep_cooldown_s=0.4):
        super().__init__("distance", video_path, conf)
        self.H_REAL = H_REAL; self.FOCAL_MM = FOCAL_MM; self.SENSOR_WIDTH_MM = SENSOR_WIDTH_MM
        self.smooth_alpha = smooth_alpha; self.draw_strategy = draw_strategy
        self.min_safe_distance_m = min_safe_distance_m; self.beep_cooldown_s = beep_cooldown_s
        self._ds = None; self._last_beep = 0.0
    def _smooth(self, p, n): a=self.smooth_alpha; return n if p is None else (1-a)*p + a*n
    def annotate(self, frame):
        model = get_model("distance")
        h,w = frame.shape[:2]
        f_px = self.FOCAL_MM * (w / max(1e-6, self.SENSOR_WIDTH_MM))
        res = model(frame, verbose=False, conf=self.conf)
        boxes = getattr(res[0], "boxes", None)
        annotated = frame.copy()
        choice = None
        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
            idx = int(np.argmax(confs)) if self.draw_strategy=="confident" else int(np.argmax(xyxy[:,3]-xyxy[:,1]))
            x1,y1,x2,y2 = xyxy[idx].astype(int)
            h_px = max(1.0, float(y2-y1))
            D = (f_px * self.H_REAL) / h_px
            choice = (D,(x1,y1,x2,y2))
        if choice:
            D,(x1,y1,x2,y2) = choice
            self._ds = self._smooth(self._ds, D)
            too_close = (self._ds is not None and self._ds < self.min_safe_distance_m)
            color = (0,0,255) if too_close else (0,255,0)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            label = f"D: {self._ds:.2f} m"
            y_text = y1-10 if y1-10>20 else y1+25
            cv2.putText(annotated,label,(x1,y_text),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2,cv2.LINE_AA)
            if too_close and (time.time()-self._last_beep>=self.beep_cooldown_s):
                BEEP(); self._last_beep = time.time()
        else:
            cv2.putText(annotated,"No person detected",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2,cv2.LINE_AA)
        return annotated, res

# ---------- Runner control ----------
RUNNERS = {}

def start_detector(kind):
    if kind in RUNNERS and RUNNERS[kind].is_alive():
        return
    video = VIDEOS.get(kind, "")
    if kind == "fire":
        runner = FireDetector(video_path=video, det_threshold=3, conf=0.5)
    elif kind == "weapons":
        runner = WeaponsDetector(video_path=video, conf=0.6)
    elif kind == "people":
        runner = PeopleDetector(video_path=video, conf=0.4)
    elif kind == "fall":
        runner = FallDetector(video_path=video, conf=0.4)
    elif kind == "distance":
        runner = DistancePeopleDetector(video_path=video)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    STREAMS[kind].running = True
    STREAMS[kind].error = None
    RUNNERS[kind] = runner
    runner.start()

def stop_detector(kind):
    """Stop a detector thread if running."""
    th = RUNNERS.get(kind)
    if th and th.is_alive():
        th.stop_flag = True
        th.join(timeout=2.0)
    # clean stream state
    st = STREAMS.get(kind)
    if st:
        st.running = False
        st.latest_jpeg = None
        st.error = None
    if kind in RUNNERS:
        del RUNNERS[kind]

# ---------- Routes ----------
@app.route("/")
def route_index():
    return render_template("index.html")

@app.route("/live/<kind>")
def route_live(kind):
    return render_template("live-detection.html", kind=kind)

@app.route("/api/start/<kind>", methods=["POST"])
def route_start(kind):
    # Verify model exists first
    try:
        _ = get_model(kind)
    except Exception as e:
        STREAMS[kind].error = str(e)
        return jsonify(ok=False, error=str(e)), 400
    start_detector(kind)
    return jsonify(ok=True)

@app.route("/api/stop/<kind>", methods=["POST"])
def route_stop(kind):
    stop_detector(kind)
    return jsonify(ok=True)

@app.route("/stream/<kind>")
def route_stream(kind):
    st = STREAMS.get(kind)
    if st is None:
        def unknown():
            while True:
                yield text_frame(f"Unknown stream: {kind}", color=(0,0,255))
                time.sleep(0.4)
        return Response(unknown(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # viewer enters
    with st.lock:
        st.viewers += 1
        st.last_seen = time.time()

    def gen():
        try:
            # show placeholders until first frame or error
            while st.latest_jpeg is None:
                msg = st.error or ("Starting…" if st.running else "Waiting for /api/start")
                yield text_frame(msg, color=((0,0,255) if st.error else (80,80,80)))
                time.sleep(0.25)

            # stream frames
            while True:
                with st.lock:
                    frame = st.latest_jpeg
                    st.last_seen = time.time()
                yield frame if frame else text_frame("No frame")
                time.sleep(0.02)
        finally:
            # viewer leaves
            with st.lock:
                st.viewers = max(0, st.viewers - 1)
            # auto-stop when last viewer leaves
            if st.viewers == 0:
                stop_detector(kind)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
