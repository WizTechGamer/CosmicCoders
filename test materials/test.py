import cv2
from ultralytics import YOLO
import sys

# --- Beep helper ---
try:
    import winsound
    def beep():
        winsound.Beep(1000, 200)  # 1000 Hz for 200 ms
except ImportError:
    def beep():
        sys.stdout.write('\a')
        sys.stdout.flush()

# --- Settings ---
model_path = 'models/fire/fire.pt'
video_path = 'videos/fire.mp4'
DETECTION_THRESHOLD = 3     # ✅ Minimum number of detections to trigger beep

# --- Load model ---
model = YOLO(model_path)

# --- Open video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit("❌ Could not open video file.")

cv2.namedWindow("YOLOv11 Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv11 Object Detection", 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    annotated = results[0].plot()

    # ✅ Beep only if detections >= threshold
    num_detections = len(results[0].boxes)
    if num_detections >= DETECTION_THRESHOLD:
        beep()

    cv2.imshow("YOLOv11 Object Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
