"""CSS partage minimal et sans selecteurs risqués."""
import streamlit as st

_CSS = """
<style>
.main .block-container {
    max-width: 1350px !important;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="stSidebarNav"],
[data-testid="collapsedControl"] {
    display: none !important;
}

[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label,
[data-testid="stButton"] button {
    font-size: 0.95rem !important;
}

[data-testid="stButton"] button {
    min-height: 48px !important;
}

/* ── Boutons Reel / Simule ── */
.mode-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    padding: 18px 10px;
    cursor: pointer;
    transition: transform 0.12s, box-shadow 0.15s;
    text-align: center;
    min-height: 140px;
    border: 2px solid transparent;
}
.mode-btn:hover { transform: scale(1.03); }
.mode-btn.reel {
    background: linear-gradient(135deg, #1a4a7a 0%, #2d6db5 100%);
    border-color: #3b82f6;
}
.mode-btn.sim {
    background: linear-gradient(135deg, #7a5a00 0%, #c49a1a 100%);
    border-color: #eab308;
}
.mode-btn.disabled {
    background: #2a2a2a !important;
    border-color: #444 !important;
    opacity: 0.45;
    cursor: not-allowed;
    pointer-events: none;
}
.mode-btn .icon { font-size: 3rem; margin-bottom: 6px; }
.mode-btn .label {
    font-size: 1.15rem;
    font-weight: 700;
    color: #fff;
}
.mode-btn.disabled .label { color: #888; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
