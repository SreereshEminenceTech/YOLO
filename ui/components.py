"""
Streamlit UI components for the YOLO Face Detection dashboard.

Provides reusable renderers for the sidebar metrics, status bar,
detection log, and empty state placeholder.
"""

import streamlit as st
import pandas as pd
from typing import Optional

from core.tracker import FPSTracker, SessionLogger


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(
    fps_tracker: FPSTracker,
    logger: SessionLogger,
    model_info: str = "YOLOv8n",
    is_active: bool = False,
):
    """
    Render the full sidebar dashboard with metrics, chart, log, and download.

    Args:
        fps_tracker: FPSTracker instance for FPS data.
        logger: SessionLogger instance for detection data.
        model_info: Human-readable model description.
        is_active: Whether the detection stream is currently active.
    """
    with st.sidebar:
        # ── Header ────────────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-family:Space Grotesk;font-size:1.15rem;font-weight:700;'
            'color:#7C6BFF;margin-bottom:0.2rem;">⚡ Detection Dashboard</p>',
            unsafe_allow_html=True,
        )

        # ── Status ────────────────────────────────────────────────────────────
        if is_active:
            st.markdown(
                '<div class="status-badge live">● LIVE</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge idle">○ IDLE</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Live Metrics ──────────────────────────────────────────────────────
        fps = fps_tracker.get_fps()
        peak = logger.peak_faces
        avg_conf = logger.overall_avg_confidence
        uptime = logger.uptime_display

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f'''<div class="metric-card">
                    <div class="label">FPS</div>
                    <div class="value {'green' if fps > 10 else 'gold' if fps > 5 else 'red'}">{fps:.1f}</div>
                </div>''',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f'''<div class="metric-card">
                    <div class="label">Peak Faces</div>
                    <div class="value cyan">{peak}</div>
                </div>''',
                unsafe_allow_html=True,
            )

        col3, col4 = st.columns(2)

        with col3:
            st.markdown(
                f'''<div class="metric-card">
                    <div class="label">Avg Confidence</div>
                    <div class="value purple">{avg_conf:.0f}%</div>
                </div>''',
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f'''<div class="metric-card">
                    <div class="label">Uptime</div>
                    <div class="value gold">{uptime}</div>
                </div>''',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Privacy Mode Toggle ───────────────────────────────────────────────
        st.markdown(
            '''<div class="privacy-card">
                <div class="title">🔒 Privacy Mode</div>
                <div class="desc">Blur detected faces while still counting them</div>
            </div>''',
            unsafe_allow_html=True,
        )
        privacy_mode = st.toggle("Enable Face Blur", value=False, key="privacy_toggle")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Confidence Chart ──────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.8rem;font-weight:600;color:#8892B0;'
            'text-transform:uppercase;letter-spacing:0.06em;">📊 Confidence Over Time</p>',
            unsafe_allow_html=True,
        )

        conf_history = logger.get_confidence_history(50)
        if conf_history and len(conf_history) > 1:
            chart_df = pd.DataFrame({"confidence": conf_history})
            st.line_chart(chart_df, height=120, use_container_width=True)
        else:
            st.markdown(
                '<p style="color:#8892B0;font-size:0.75rem;text-align:center;'
                'padding:1rem 0;">Waiting for detection data...</p>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Detection Log ─────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.8rem;font-weight:600;color:#8892B0;'
            'text-transform:uppercase;letter-spacing:0.06em;">📋 Detection Log</p>',
            unsafe_allow_html=True,
        )

        recent = logger.get_recent(15)
        if recent:
            # Build HTML table
            rows = ""
            for entry in reversed(recent):
                conf_color = (
                    "#00E676" if entry["avg_confidence"] >= 80
                    else "#FFC107" if entry["avg_confidence"] >= 50
                    else "#FF5252"
                )
                rows += (
                    f'<tr>'
                    f'<td>{entry["timestamp"]}</td>'
                    f'<td style="text-align:center;">{entry["faces"]}</td>'
                    f'<td style="color:{conf_color};text-align:center;">'
                    f'{entry["avg_confidence"]}%</td>'
                    f'<td style="text-align:center;">{entry["fps"]}</td>'
                    f'</tr>'
                )

            st.markdown(
                f'''<div class="log-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Faces</th>
                                <th>Conf</th>
                                <th>FPS</th>
                            </tr>
                        </thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>''',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="color:#8892B0;font-size:0.75rem;text-align:center;'
                'padding:1rem 0;">No detections logged yet</p>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Download CSV ──────────────────────────────────────────────────────
        if logger.total_entries > 0:
            csv_data = logger.to_csv()
            st.download_button(
                label="⬇️  Download Session Log (CSV)",
                data=csv_data,
                file_name=f"yolo_detection_log.csv",
                mime="text/csv",
                key="download_csv",
            )

        # ── Model Info Footer ─────────────────────────────────────────────────
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#8892B0;font-size:0.68rem;text-align:center;">'
            f'Model: {model_info}<br>Resolution: 480p optimized</p>',
            unsafe_allow_html=True,
        )

    return privacy_mode


# ─── Status Bar ───────────────────────────────────────────────────────────────

def render_status_bar(
    model_info: str = "YOLOv8n",
    resolution: str = "480p",
    uptime: str = "00:00",
    is_active: bool = False,
):
    """Render a compact status bar below the video feed."""
    status_dot = "🟢" if is_active else "🔴"
    status_text = "Live" if is_active else "Stopped"

    st.markdown(
        f'''<div class="status-bar">
            <div class="item">{status_dot} <span>{status_text}</span></div>
            <div class="item">Model: <span>{model_info}</span></div>
            <div class="item">Resolution: <span>{resolution}</span></div>
            <div class="item">Uptime: <span>{uptime}</span></div>
        </div>''',
        unsafe_allow_html=True,
    )


# ─── Empty State ──────────────────────────────────────────────────────────────

def render_empty_state():
    """Render a placeholder when the camera is not yet active."""
    st.markdown(
        '''<div class="empty-state">
            <div class="icon">👁️</div>
            <div class="title">YOLO Face Detection Ready</div>
            <div class="subtitle">
                Press <strong>START</strong> on the video component above to begin real-time detection.
                <br>Your webcam feed will be processed frame-by-frame with YOLOv8.
            </div>
        </div>''',
        unsafe_allow_html=True,
    )
