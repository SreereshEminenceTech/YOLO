"""
Light-theme CSS styles for the YOLO Face Detection app.

Injects custom CSS via st.markdown for a polished, clean dashboard look
with white cards, subtle shadows, and modern sans-serif typography.
"""

import streamlit as st


def inject_styles():
    """Inject the full custom CSS theme into the Streamlit page."""
    st.markdown(_MAIN_CSS, unsafe_allow_html=True)


_MAIN_CSS = """
<style>
/* ─── Google Fonts ────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root Variables ──────────────────────────────────────────────────────── */
:root {
    --bg-primary:    #F3F4F6;
    --bg-card:       #FFFFFF;
    --accent:        #2563EB;
    --accent-glow:   rgba(37, 99, 235, 0.15);
    --blue-light:    #DBEAFE;
    --green:         #10B981;
    --red:           #EF4444;
    --gold:          #F59E0B;
    --text-primary:  #1F2937;
    --text-muted:    #6B7280;
    --border:        #E5E7EB;
    --font-body:     'Inter', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
    --shadow-sm:     0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md:     0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --radius:        12px;
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
    padding: 1.5rem 1rem 1rem;
    margin-bottom: 1.2rem;
    background: var(--bg-card);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}

.app-header h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 0 0.3rem 0 !important;
    letter-spacing: -0.02em;
}

.app-header p {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin: 0;
}

/* ─── Cards & Containers ──────────────────────────────────────────────────── */
.dashboard-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1rem;
    height: 100%;
}

.card-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

/* ─── Metric Item ─────────────────────────────────────────────────────────── */
.metric-item {
    display: flex;
    flex-direction: column;
    margin-bottom: 1rem;
}

.metric-item .label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}

.metric-item .value {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-primary);
}

.metric-item .value.blue  { color: var(--accent); }
.metric-item .value.green { color: var(--green); }
.metric-item .value.gold  { color: var(--gold); }
.metric-item .value.red   { color: var(--red); }

/* ─── Status Badge ────────────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.status-badge.live {
    background: #D1FAE5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}

.status-badge.idle {
    background: #F3F4F6;
    color: var(--text-muted);
    border: 1px solid var(--border);
}

.status-badge.live::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--green);
    animation: blink 1.5s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ─── Metrics Row (Bottom) ────────────────────────────────────────────────── */
.metrics-row {
    display: flex;
    justify-content: space-between;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    box-shadow: var(--shadow-sm);
}

.metrics-row .metric {
    text-align: center;
}

.metrics-row .metric .label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}

.metrics-row .metric .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ─── Log Table ───────────────────────────────────────────────────────────── */
.log-container {
    max-height: 250px;
    overflow-y: auto;
    margin-top: 0.5rem;
}

.log-container table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}

.log-container th {
    text-align: left;
    padding: 0.5rem;
    color: var(--text-muted);
    font-weight: 500;
    border-bottom: 1px solid var(--border);
}

.log-container td {
    padding: 0.5rem;
    color: var(--text-primary);
    border-bottom: 1px solid #F9FAFB;
}

/* ─── Empty State ─────────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: #F9FAFB;
    border: 1px dashed #D1D5DB;
    border-radius: var(--radius);
}

.empty-state .title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}

.empty-state .subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* ─── WebRTC Container ────────────────────────────────────────────────────── */
.stVideo, iframe {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

/* ─── Download Button ─────────────────────────────────────────────────────── */
.stDownloadButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
    font-weight: 500 !important;
}

.stDownloadButton > button:hover {
    background: #1D4ED8 !important; /* darker blue */
}

/* ─── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}
.stTabs [data-baseweb="tab"] {
    padding-top: 0;
    padding-bottom: 1rem;
    color: var(--text-muted);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent);
}

</style>
"""
