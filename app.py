"""
YOLO Face Detection — Streamlit App

Real-time face detection via webcam using YOLOv8 + streamlit-webrtc.
Features: multi-face detection, FPS tracking, privacy blur, session logging,
and a clean 3-column professional UI.

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
from ui.components import render_left_panel, render_right_panel, render_metrics_row, render_empty_state


# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniSight Face Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Real-time face detection powered by YOLOv8 & Streamlit WebRTC.",
    },
)

# ─── Inject Custom CSS ───────────────────────────────────────────────────────
inject_styles()


# ─── Module-Level Shared Objects ──────────────────────────────────────────────
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
        # Convert to OpenCV format
        img = frame.to_ndarray(format="bgr24")

        # Record frame timing
        self.fps_tracker.tick()
        current_fps = self.fps_tracker.get_fps()

        # Run detection
        detections = self.detector.detect_faces(img)

        # Calculate average confidence
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)
        else:
            avg_conf = 0.0

        # Log data
        self.logger.log(
            face_count=len(detections),
            avg_confidence=avg_conf,
            fps=current_fps,
        )

        # Annotate frame
        annotated = self.detector.annotate_frame(
            frame=img,
            detections=detections,
            privacy_mode=self.privacy_mode,
            show_fps=current_fps,
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─── WebRTC Configuration ────────────────────────────────────────────────────
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}


# ─── App Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>OmniSight Face Detection</h1>
        <p>Enterprise Real-time Face Detection Platform</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Top Navigation Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Real-time Detection", "Analytics", "Settings"])

with tab2:
    # ─── 3-Column Layout ────────────────────────────────────────────────────────
    # Proportions roughly match Sample 2: left panel (1), main video (2.5), right panel (1)
    col_left, col_main, col_right = st.columns([1, 2.5, 1], gap="medium")

    # ── Left Column ──
    with col_left:
        render_left_panel(_session_logger)

    # ── Main Column ──
    with col_main:
        # We store webrtc_ctx to use below, outside the context manager
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

        # Top-right status badge indicating stream state
        if webrtc_ctx.state.playing:
            st.markdown('<div style="text-align:right; margin-top: -1.5rem;"><span class="status-badge live">LIVE</span></div>', unsafe_allow_html=True)
            st.session_state["is_active"] = True
        else:
            st.session_state["is_active"] = False

        # If not active, show empty state inside the main col
        if not st.session_state["is_active"]:
            render_empty_state()

        # Metrics Row placed under the video player
        render_metrics_row(_fps_tracker, _session_logger)
        
        # Download button
        if _session_logger.total_entries > 0:
            st.download_button(
                label="Download Session Logs (CSV)",
                data=_session_logger.to_csv(),
                file_name="omnisight_logs.csv",
                mime="text/csv",
                key="download_csv",
            )

    # ── Right Column ──
    with col_right:
        privacy_mode = render_right_panel(_detector.model_info)


# ─── Sync Settings ───────────────────────────────────────────────────────────
if webrtc_ctx and webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.privacy_mode = privacy_mode

with tab1:
    st.info("Dashboard overview (Placeholder)")

with tab3:
    st.info("Advanced analytics (Placeholder)")

with tab4:
    st.info("System settings (Placeholder)")
