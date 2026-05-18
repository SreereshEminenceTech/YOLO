"""
FPS Tracker & Session Logger.

Provides real-time performance metrics and session-level data logging
for the YOLO face detection Streamlit app.
"""

import time
import threading
from collections import deque
from datetime import datetime
from typing import List, Optional

import pandas as pd



class FPSTracker:
    """
    Thread-safe rolling-window FPS calculator.

    Records timestamps of the last N frames and computes a smoothed FPS.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of recent frames to average over.
        """
        self._timestamps = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def tick(self):
        """Record a new frame timestamp."""
        with self._lock:
            self._timestamps.append(time.monotonic())

    def get_fps(self) -> float:
        """
        Calculate the current FPS based on the rolling window.

        Returns:
            Smoothed FPS value. Returns 0.0 if fewer than 2 frames recorded.
        """
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / elapsed

    def reset(self):
        """Clear all recorded timestamps."""
        with self._lock:
            self._timestamps.clear()



class SessionLogger:
    """
    Thread-safe session data logger.

    Logs detection events at a throttled rate (max 1 entry per second)
    to avoid flooding the log with per-frame data.
    """

    def __init__(self, throttle_interval: float = 1.0):
        """
        Args:
            throttle_interval: Minimum seconds between log entries.
        """
        self._entries: List[dict] = []
        self._lock = threading.Lock()
        self._last_log_time: float = 0.0
        self._throttle = throttle_interval
        self._peak_faces: int = 0
        self._total_confidence: float = 0.0
        self._total_detections: int = 0
        self._session_start = time.monotonic()

    def log(self, face_count: int, avg_confidence: float, fps: float):
        """
        Log a detection event (throttled).

        Args:
            face_count: Number of faces detected in the current frame.
            avg_confidence: Average confidence score across all faces.
            fps: Current FPS at the time of logging.
        """
        now = time.monotonic()

        with self._lock:
            # Update peak even if throttled
            self._peak_faces = max(self._peak_faces, face_count)

            if face_count > 0:
                self._total_confidence += avg_confidence * face_count
                self._total_detections += face_count

            # Throttle log entries
            if now - self._last_log_time < self._throttle:
                return

            self._last_log_time = now

            self._entries.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "faces": face_count,
                "avg_confidence": round(avg_confidence * 100, 1) if face_count > 0 else 0.0,
                "fps": round(fps, 1),
            })

    @property
    def peak_faces(self) -> int:
        """Maximum number of faces detected in a single frame this session."""
        with self._lock:
            return self._peak_faces

    @property
    def overall_avg_confidence(self) -> float:
        """Weighted average confidence across all detections this session (0-100)."""
        with self._lock:
            if self._total_detections == 0:
                return 0.0
            return round((self._total_confidence / self._total_detections) * 100, 1)

    @property
    def total_entries(self) -> int:
        """Number of log entries recorded."""
        with self._lock:
            return len(self._entries)

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the session started."""
        return time.monotonic() - self._session_start

    @property
    def uptime_display(self) -> str:
        """Human-readable uptime string (MM:SS)."""
        secs = int(self.uptime_seconds)
        mins = secs // 60
        secs = secs % 60
        return f"{mins:02d}:{secs:02d}"

    def get_recent(self, n: int = 20) -> List[dict]:
        """Get the last N log entries."""
        with self._lock:
            return list(self._entries[-n:])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the full session log to a Pandas DataFrame."""
        with self._lock:
            if not self._entries:
                return pd.DataFrame(columns=["timestamp", "faces", "avg_confidence", "fps"])
            return pd.DataFrame(self._entries)

    def to_csv(self) -> bytes:
        """Export the session log as CSV bytes (for download)."""
        df = self.to_dataframe()
        return df.to_csv(index=False).encode("utf-8")

    def get_confidence_history(self, n: int = 50) -> List[float]:
        """Get the last N average confidence values for charting."""
        with self._lock:
            entries = self._entries[-n:]
            return [e["avg_confidence"] for e in entries]

    def get_fps_history(self, n: int = 50) -> List[float]:
        """Get the last N FPS values for charting."""
        with self._lock:
            entries = self._entries[-n:]
            return [e["fps"] for e in entries]

    def reset(self):
        """Clear all session data."""
        with self._lock:
            self._entries.clear()
            self._peak_faces = 0
            self._total_confidence = 0.0
            self._total_detections = 0
            self._last_log_time = 0.0
            self._session_start = time.monotonic()
