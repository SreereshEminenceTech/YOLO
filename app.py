"""
YOLO Face Detection — Streamlit App

Real-time face detection via webcam using YOLOv8 + streamlit-webrtc.
Features: multi-face detection, FPS tracking, privacy blur, session logging,
and downloadable CSV reports.

Usage:
    streamlit run app.py
"""

import av
import cv2
import threading
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
)

from core.detector import FaceDetector
from core.tracker import FPSTracker, SessionLogger
from ui.styles import inject_styles
from ui.components import render_sidebar, render_status_bar, render_empty_state


# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Face Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Real-time face detection powered by YOLOv8 & Streamlit WebRTC.",
    },
)

# ─── Inject Custom CSS ───────────────────────────────────────────────────────
inject_styles()


# ─── Module-Level Shared Objects ──────────────────────────────────────────────
# These are shared between the Streamlit main thread and the WebRTC processor
# thread. They are thread-safe by design (use internal locks).
# Using module-level singletons avoids st.session_state access issues in the
# video processor thread.

@st.cache_resource
def get_shared_fps_tracker():
    return FPSTracker(window_size=30)

@st.cache_resource
def get_shared_logger():
    return SessionLogger(throttle_interval=1.0)

@st.cache_resource(show_spinner=False)
def load_detector():
    """Load and cache the YOLO face detector model."""
    return FaceDetector(
        model_path=None,
        confidence_threshold=0.4,
        input_size=480,
    )


# Initialize shared objects
_fps_tracker = get_shared_fps_tracker()
_session_logger = get_shared_logger()
_detector = load_detector()


# ─── Session State ────────────────────────────────────────────────────────────

if "is_active" not in st.session_state:
    st.session_state["is_active"] = False


# ─── Video Processor ─────────────────────────────────────────────────────────

class YOLOVideoProcessor(VideoProcessorBase):
    """
    WebRTC video processor that runs YOLO face detection on every frame.

    Uses module-level shared objects (not st.session_state) because this
    class is instantiated in a separate thread by streamlit-webrtc.
    """

    def __init__(self):
        self.detector = _detector
        self.fps_tracker = _fps_tracker
        self.logger = _session_logger
        self._privacy_mode: bool = False
        self._lock = threading.Lock()

    @property
    def privacy_mode(self) -> bool:
        with self._lock:
            return self._privacy_mode

    @privacy_mode.setter
    def privacy_mode(self, value: bool):
        with self._lock:
            self._privacy_mode = value

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process a single video frame.

        Called automatically by streamlit-webrtc for every frame.
        Runs YOLO detection, annotates the frame, and updates metrics.
        """
        # ── Convert to OpenCV format ──────────────────────────────────────────
        img = frame.to_ndarray(format="bgr24")

        # ── Record frame timing ──────────────────────────────────────────────
        self.fps_tracker.tick()
        current_fps = self.fps_tracker.get_fps()

        # ── Run detection ────────────────────────────────────────────────────
        detections = self.detector.detect_faces(img)

        # ── Calculate average confidence ─────────────────────────────────────
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)
        else:
            avg_conf = 0.0

        # ── Log data (throttled internally) ──────────────────────────────────
        self.logger.log(
            face_count=len(detections),
            avg_confidence=avg_conf,
            fps=current_fps,
        )

        # ── Annotate frame ───────────────────────────────────────────────────
        annotated = self.detector.annotate_frame(
            frame=img,
            detections=detections,
            privacy_mode=self.privacy_mode,
            show_fps=current_fps,
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─── App Header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>👁️ YOLO Face Detection</h1>
        <p>Real-time multi-face detection powered by YOLOv8 · Streamlit WebRTC</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar Dashboard ───────────────────────────────────────────────────────

privacy_mode = render_sidebar(
    fps_tracker=_fps_tracker,
    logger=_session_logger,
    model_info=_detector.model_info,
    is_active=st.session_state.get("is_active", False),
)


# ─── WebRTC Configuration ────────────────────────────────────────────────────

# RTC config for Streamlit Cloud (uses public STUN servers)
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}


# ─── Main Video Feed ─────────────────────────────────────────────────────────

webrtc_ctx = webrtc_streamer(
    key="yolo-face-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15, "max": 20},
        },
        "audio": False,
    },
    async_processing=True,
)


# ─── Sync Privacy Mode Toggle ────────────────────────────────────────────────

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.privacy_mode = privacy_mode
    st.session_state["is_active"] = True
else:
    st.session_state["is_active"] = False


# ─── Status Bar & Empty State ─────────────────────────────────────────────────

render_status_bar(
    model_info=_detector.model_info,
    resolution="640×480",
    uptime=_session_logger.uptime_display,
    is_active=st.session_state.get("is_active", False),
)

if not st.session_state.get("is_active", False):
    render_empty_state()


# ─── Auto-Refresh Hint ───────────────────────────────────────────────────────

if st.session_state.get("is_active", False):
    st.markdown(
        '<p style="color:#8892B0;font-size:0.72rem;text-align:center;margin-top:1rem;">'
        '💡 Sidebar metrics update when you interact with the page. '
        'Toggle Privacy Mode or click anywhere to refresh stats.</p>',
        unsafe_allow_html=True,
    )
