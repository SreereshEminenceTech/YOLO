from ultralytics import YOLO
import cv2
import time

# Load trained YOLO model
model = YOLO("best.pt")
# OR:
# model = YOLO(r"C:\path\to\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ Webcam started")
print("Press 'q' to quit")

# Colors
COLORS = [
    (255, 56, 56),
    (56, 255, 56),
    (56, 56, 255),
    (255, 200, 56),
    (255, 56, 200),
    (56, 255, 200)
]

while True:
    start_time = time.time()

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    # YOLO inference
    results = model.predict(frame, conf=0.4, verbose=False)[0]

    # Draw detections
    for box in results.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{model.names[cls_id]} {conf:.2f}"

        color = COLORS[cls_id % len(COLORS)]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - th - 10),
            (x1 + tw + 10, y1),
            color,
            -1
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # FPS
    fps = 1 / (time.time() - start_time)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Show frame
    cv2.imshow("Live Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()