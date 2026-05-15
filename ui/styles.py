"""
Premium dark-theme CSS styles for the YOLO Face Detection app.

Injects custom CSS via st.markdown for a polished, data-dashboard look
with glassmorphism cards, custom fonts, and smooth animations.
"""

import streamlit as st


def inject_styles():
    """Inject the full custom CSS theme into the Streamlit page."""
    st.markdown(_MAIN_CSS, unsafe_allow_html=True)


_MAIN_CSS = """
<style>
/* ─── Google Fonts ────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root Variables ──────────────────────────────────────────────────────── */
:root {
    --bg-primary:    #0A0E1A;
    --bg-secondary:  #131929;
    --bg-card:       rgba(19, 25, 41, 0.85);
    --accent:        #7C6BFF;
    --accent-glow:   rgba(124, 107, 255, 0.25);
    --cyan:          #00E5FF;
    --green:         #00E676;
    --red:           #FF5252;
    --gold:          #FFC107;
    --text-primary:  #F0F0FF;
    --text-muted:    #8892B0;
    --border:        rgba(124, 107, 255, 0.15);
    --glass-bg:      rgba(19, 25, 41, 0.65);
    --glass-border:  rgba(124, 107, 255, 0.12);
    --font-body:     'Space Grotesk', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
}

/* ─── Global Overrides ────────────────────────────────────────────────────── */
.stApp {
    font-family: var(--font-body) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

/* ─── App Header ──────────────────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 1.8rem 1rem 1rem;
    margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(124,107,255,0.08) 0%, rgba(0,229,255,0.04) 100%);
    border-radius: 16px;
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(12px);
    animation: fadeSlideDown 0.6s ease-out;
}

.app-header h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--accent) 0%, var(--cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0 !important;
    letter-spacing: -0.02em;
}

.app-header p {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin: 0;
}

/* ─── Metric Cards ────────────────────────────────────────────────────────── */
.metric-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-glow);
}

.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}

.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: var(--font-mono);
}

.metric-card .value.purple { color: var(--accent); }
.metric-card .value.cyan   { color: var(--cyan); }
.metric-card .value.green  { color: var(--green); }
.metric-card .value.gold   { color: var(--gold); }
.metric-card .value.red    { color: var(--red); }

/* ─── Status Badge ────────────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.status-badge.live {
    background: rgba(0, 230, 118, 0.12);
    color: var(--green);
    border: 1px solid rgba(0, 230, 118, 0.25);
    animation: pulse-glow 2s infinite;
}

.status-badge.idle {
    background: rgba(136, 146, 176, 0.12);
    color: var(--text-muted);
    border: 1px solid rgba(136, 146, 176, 0.2);
}

/* ─── Status Bar ──────────────────────────────────────────────────────────── */
.status-bar {
    display: flex;
    gap: 1.5rem;
    align-items: center;
    padding: 0.7rem 1.2rem;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    margin-top: 0.8rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    backdrop-filter: blur(8px);
}

.status-bar .item {
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.status-bar .item span {
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-weight: 500;
}

/* ─── Section Divider ─────────────────────────────────────────────────────── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--border) 50%, transparent 100%);
    margin: 1.2rem 0;
}

/* ─── Log Table ───────────────────────────────────────────────────────────── */
.log-container {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 0.8rem;
    max-height: 250px;
    overflow-y: auto;
    backdrop-filter: blur(8px);
}

.log-container table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.75rem;
    font-family: var(--font-mono);
}

.log-container th {
    text-align: left;
    padding: 0.4rem 0.5rem;
    color: var(--accent);
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
}

.log-container td {
    padding: 0.3rem 0.5rem;
    color: var(--text-primary);
    border-bottom: 1px solid rgba(255,255,255,0.03);
}

/* ─── Empty State ─────────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    background: var(--glass-bg);
    border: 1px dashed var(--glass-border);
    border-radius: 16px;
    backdrop-filter: blur(8px);
}

.empty-state .icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    animation: float 3s ease-in-out infinite;
}

.empty-state .title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 0.5rem;
}

.empty-state .subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* ─── Privacy Toggle ──────────────────────────────────────────────────────── */
.privacy-card {
    background: rgba(255, 82, 82, 0.06);
    border: 1px solid rgba(255, 82, 82, 0.15);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
}

.privacy-card .title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--red);
    margin-bottom: 0.2rem;
}

.privacy-card .desc {
    font-size: 0.72rem;
    color: var(--text-muted);
}

/* ─── Download Button ─────────────────────────────────────────────────────── */
.stDownloadButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--accent) 0%, #5B4FD9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton > button:hover {
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* ─── WebRTC Container ────────────────────────────────────────────────────── */
.stVideo, iframe {
    border-radius: 14px !important;
    border: 2px solid var(--glass-border) !important;
    overflow: hidden;
}

/* ─── Animations ──────────────────────────────────────────────────────────── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(0, 230, 118, 0.2); }
    50%      { box-shadow: 0 0 15px rgba(0, 230, 118, 0.4); }
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-8px); }
}

/* ─── Sidebar Tweaks ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* ─── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--accent);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: #9B8FFF; }
</style>
"""
