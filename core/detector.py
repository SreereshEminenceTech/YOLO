import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


# ─── Detection Result ─────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single face detection result."""
    bbox: tuple           
    confidence: float     
    class_name: str       
    class_id: int         


# ─── Color Palette for Bounding Boxes ─────────────────────────────────────────

COLORS = [
    (124, 107, 255),  
    (0, 229, 255),    
    (255, 200, 56),   
    (255, 56, 200),   
    (56, 255, 56),     
    (255, 56, 56),   
]


# ─── Face Detector Class ─────────────────────────────────────────────────────

class FaceDetector:

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        input_size: int = 480,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to .onnx or .pt model file.
                        If None, downloads yolov8n.pt from Ultralytics.
            confidence_threshold: Minimum confidence to keep a detection.
            input_size: Resize input frames to this height before inference.
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self._model = None
        self._model_path = model_path
        self._model_type = None  # "onnx" or "ultralytics"
        self._class_names = {}

        self._load_model()

    def _load_model(self):
        """Load the YOLO model (ONNX preferred, .pt fallback)."""

        # ── Strategy 1: Try ONNX model ────────────────────────────────────────
        onnx_paths = [
            self._model_path,
            os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n_face.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "models", "best.onnx"),
        ]

        for path in onnx_paths:
            if path and os.path.isfile(path) and path.endswith(".onnx"):
                try:
                    import onnxruntime as ort
                    self._model = ort.InferenceSession(
                        path,
                        providers=["CPUExecutionProvider"],
                    )
                    self._model_type = "onnx"
                    print(f"✅ Loaded ONNX model: {path}")
                    return
                except Exception as e:
                    print(f"⚠️  ONNX load failed ({path}): {e}")

        # ── Strategy 2: Try .pt model with Ultralytics ────────────────────────
        pt_paths = [
            self._model_path,
            os.path.join(os.path.dirname(__file__), "..", "models", "best_v2.pt"),
        ]

        for path in pt_paths:
            if path and os.path.isfile(path) and path.endswith(".pt"):
                try:
                    from ultralytics import YOLO
                    self._model = YOLO(path)
                    self._model_type = "ultralytics"
                    self._class_names = self._model.names
                    print(f"✅ Loaded Ultralytics model: {path}")
                    return
                except Exception as e:
                    print(f"⚠️  .pt load failed ({path}): {e}")

        raise RuntimeError(
            "Could not load custom YOLO face model. Ensure best.pt is in the models/ directory."
        )

    @property
    def model_info(self) -> str:
        """Human-readable model description."""
        if self._model_type == "onnx":
            return "YOLOv8n Face (ONNX)"
        elif self._model_type == "ultralytics":
            return "YOLOv8n (Ultralytics)"
        return "Unknown"

    # ─── Core Detection ───────────────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> List[Detection]:
        """
        Run face detection on a single BGR frame.

        Args:
            frame: OpenCV BGR image (numpy array, shape HxWx3)

        Returns:
            List of Detection objects for all faces found.
        """
        if self._model is None:
            return []

        if self._model_type == "ultralytics":
            return self._detect_ultralytics(frame)
        elif self._model_type == "onnx":
            return self._detect_onnx(frame)
        return []

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        """Run inference using Ultralytics YOLO."""
        # Resize for speed
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        if scale < 1.0:
            resized = cv2.resize(frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame
            scale = 1.0

        results = self._model.predict(
            resized,
            conf=self.confidence_threshold,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Scale coordinates back to original frame size
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            class_name = self._class_names.get(cls_id, f"class_{cls_id}")

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                class_name=class_name,
                class_id=cls_id,
            ))

        return detections

    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        """Run inference using ONNX Runtime."""
        # Preprocess: resize, normalize, transpose to NCHW
        h, w = frame.shape[:2]
        target = self.input_size
        scale = target / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        # Normalize and transpose
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

        # Run inference
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: blob})
        predictions = outputs[0]  # shape: (1, num_detections, 5+num_classes)

        # Parse YOLO output format
        detections = []
        if predictions.ndim == 3:
            preds = predictions[0].T  # (num_detections, 5+num_classes)
        else:
            preds = predictions

        for pred in preds:
            # YOLO format: cx, cy, w, h, conf, class_scores...
            if len(pred) < 5:
                continue

            cx, cy, bw, bh = pred[:4]
            scores = pred[4:]

            if len(scores) == 1:
                # Single class (face)
                conf = float(scores[0])
                cls_id = 0
            else:
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])

            if conf < self.confidence_threshold:
                continue

            # Convert from normalized center coords to pixel coords
            x1 = int((cx - bw / 2) / scale)
            y1 = int((cy - bh / 2) / scale)
            x2 = int((cx + bw / 2) / scale)
            y2 = int((cy + bh / 2) / scale)

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_name="face",
                class_id=cls_id,
            ))

        # NMS (Non-Maximum Suppression) to remove duplicate boxes
        if detections:
            detections = self._nms(detections, iou_threshold=0.45)

        return detections

    @staticmethod
    def _nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
        """Apply Non-Maximum Suppression to filter overlapping boxes."""
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            bboxes=[(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])) for b in boxes],
            scores=scores.tolist(),
            score_threshold=0.1,
            nms_threshold=iou_threshold,
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        return [detections[i] for i in indices]

    # ─── Frame Annotation ─────────────────────────────────────────────────────

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        privacy_mode: bool = False,
        show_fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw detection results onto the frame.

        Args:
            frame: Original BGR frame.
            detections: List of Detection objects.
            privacy_mode: If True, blur face regions instead of showing them.
            show_fps: If > 0, overlay FPS counter on the frame.

        Returns:
            Annotated frame (copy of original).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            color = COLORS[det.class_id % len(COLORS)]
            conf_pct = f"{det.confidence:.0%}"
            label = f"{det.class_name} {conf_pct}"

            # ── Privacy blur ──────────────────────────────────────────────────
            if privacy_mode:
                face_roi = annotated[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if face_roi.size > 0:
                    blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    annotated[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = blurred

            # ── Bounding box ──────────────────────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # ── Confidence bar (small bar above the box) ──────────────────────
            bar_width = x2 - x1
            bar_filled = int(bar_width * det.confidence)
            bar_y = max(0, y1 - 8)
            cv2.rectangle(annotated, (x1, bar_y), (x1 + bar_width, bar_y + 5),
                          (40, 40, 60), -1)
            cv2.rectangle(annotated, (x1, bar_y), (x1 + bar_filled, bar_y + 5),
                          color, -1)

            # ── Label background ──────────────────────────────────────────────
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = max(0, bar_y - 4)
            cv2.rectangle(annotated, (x1, label_y - th - 6), (x1 + tw + 10, label_y),
                          color, -1)
            cv2.putText(annotated, label, (x1 + 5, label_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # ── Face count overlay (top-left) ─────────────────────────────────────
        count_text = f"Faces: {len(detections)}"
        cv2.putText(annotated, count_text, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 229, 255), 2, cv2.LINE_AA)

        # ── FPS overlay (top-right) ───────────────────────────────────────────
        if show_fps > 0:
            fps_text = f"FPS: {show_fps:.1f}"
            (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(annotated, fps_text, (w - tw - 15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 107, 255), 2,
                        cv2.LINE_AA)

        return annotated
