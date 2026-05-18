"""
Streamlit UI components for the YOLO Face Detection dashboard.

Provides reusable renderers for the left panel, right panel, 
metrics row, and empty state placeholder.
"""

import streamlit as st
import pandas as pd
from core.tracker import FPSTracker, SessionLogger


def render_left_panel(logger: SessionLogger):
    """
    Render the left panel with Detection Activity and Logs.
    """
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Detection Activity</div>', unsafe_allow_html=True)
    
    peak = logger.peak_faces
    st.markdown(f'''
    <div class="metric-item">
        <div class="label">Peak Live Faces</div>
        <div class="value blue">{peak}</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 1.5rem;" class="card-title">Detection Log</div>', unsafe_allow_html=True)
    
    recent = logger.get_recent(10)
    if recent:
        rows = ""
        for entry in reversed(recent):
            rows += (
                f'<tr>'
                f'<td>{entry["timestamp"]}</td>'
                f'<td style="text-align:center;">{entry["faces"]}</td>'
                f'<td style="text-align:center;">{entry["avg_confidence"]}%</td>'
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
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color:var(--text-muted);font-size:0.8rem;text-align:center;'
            'padding:1rem 0;">No detections logged yet</p>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_right_panel(model_info: str):
    """
    Render the right panel with Settings and configuration.
    """
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Configuration</div>', unsafe_allow_html=True)
    
    st.markdown(f'''
    <table style="width:100%; font-size:0.85rem; color:var(--text-primary); margin-bottom: 1.5rem;">
        <tr><td style="padding-bottom: 8px;">Model</td><td style="text-align:right; font-weight:500;">{model_info}</td></tr>
        <tr><td style="padding-bottom: 8px;">Sensitivity</td><td style="text-align:right; font-weight:500;">Medium</td></tr>
        <tr><td style="padding-bottom: 8px;">Resolution</td><td style="text-align:right; font-weight:500;">480p</td></tr>
    </table>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="card-title">Privacy Settings</div>', unsafe_allow_html=True)
    
    st.markdown(
        '<p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">'
        'Blur detected faces in the live stream.</p>',
        unsafe_allow_html=True
    )
    privacy_mode = st.toggle("Enable Face Blur", value=False, key="privacy_toggle")

    st.markdown('</div>', unsafe_allow_html=True)
    return privacy_mode


def render_metrics_row(fps_tracker: FPSTracker, logger: SessionLogger):
    """
    Render the bottom metrics row showing overall stats.
    """
    fps = fps_tracker.get_fps()
    avg_conf = logger.overall_avg_confidence
    uptime = logger.uptime_display
    total_faces = logger.total_entries # using total entries as proxy for 'faces detected today'

    st.markdown(
        f'''<div class="metrics-row">
            <div class="metric">
                <div class="label">Faces Logged</div>
                <div class="value">{total_faces}</div>
            </div>
            <div class="metric">
                <div class="label">Avg Confidence</div>
                <div class="value">{avg_conf:.1f}%</div>
            </div>
            <div class="metric">
                <div class="label">Current FPS</div>
                <div class="value">{fps:.1f}</div>
            </div>
            <div class="metric">
                <div class="label">Session Uptime</div>
                <div class="value">{uptime}</div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )


def render_empty_state():
    """Render a placeholder when the camera is not yet active."""
    st.markdown(
        '''<div class="empty-state">
            <div class="title">WebRTC Stream Inactive</div>
            <div class="subtitle">
                Click <strong>START</strong> to initialize the webcam and YOLO model.
            </div>
        </div>''',
        unsafe_allow_html=True,
    )
