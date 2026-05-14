# IntelliTrade AI - Stock Price Prediction & Trading Strategy System
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '/app/streamlit_app')

from data.sp500_companies import get_sp500_list, get_company_name
from utils.data_loader import StockDataLoader
from utils.indicators import TechnicalIndicators
from utils.backtesting import Backtester
from utils.demo_data import generate_demo_stock_data, get_demo_real_time_data

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="IntelliTrade AI", page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ───────────────────────────────────────────────────────
# [CHANGED] Completely rewritten load_css() to fix all dark mode text
# visibility issues. Every Streamlit widget is now explicitly styled
# for both Dark and Light themes.
def load_css(theme):

    if theme == "Dark":
        bg        = "#0E1117"       # Main background
        sidebar   = "#111827"       # Sidebar background
        card      = "#1A1F2E"       # Card / secondary background
        card_border = "#2D3348"     # Subtle card border
        text      = "#FFFFFF"       # Primary text
        subtext   = "#D1D5DB"       # Secondary / muted text
        accent    = "#2563EB"       # Accent blue
        accent_hover = "#3B82F6"    # Accent blue hover
        input_bg  = "#1E2536"       # Input / select background
        input_border = "#374151"    # Input borders
        hover_bg  = "#2A3144"       # Hover state background
        success   = "#10B981"       # Success green
        danger    = "#EF4444"       # Danger red
        cyan      = "#00E5FF"       # Metric accent cyan
        tab_active = "#FF6B6B"      # Active tab color
        btn_text  = "#FFFFFF"       # White button text for dark mode

    else:
        # [CHANGED] Updated light theme palette per user spec for
        # maximum readability — darker text, proper white backgrounds
        bg        = "#F9FAFB"       # Soft white main background
        sidebar   = "#FFFFFF"       # Pure white sidebar
        card      = "#FFFFFF"       # Pure white cards
        card_border = "#D1D5DB"     # Light gray borders
        text      = "#111827"       # Dark slate primary text
        subtext   = "#374151"       # Slate gray secondary text
        accent    = "#3B82F6"       # Blue buttons
        accent_hover = "#2563EB"    # Darker blue hover
        input_bg  = "#FFFFFF"       # White input backgrounds
        input_border = "#D1D5DB"    # Light gray input borders
        hover_bg  = "#F3F4F6"       # Light hover background
        success   = "#059669"       # Success green
        danger    = "#DC2626"       # Danger red
        cyan      = "#0369A1"       # Dark blue metric accent
        tab_active = "#2563EB"      # Active tab blue
        btn_text  = "#111111"       # Near-black button text

    st.markdown(f"""
    <style>
    /* ── [CHANGED] Import Inter font for modern typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── [CHANGED] Global app background & base text ── */
    .stApp {{
        background-color: {bg} !important;
        color: {text} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* ── [CHANGED] All heading levels ── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {text} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }}

    /* ── [CHANGED] Paragraph, label, span – using subtext for readability ── */
    p, label, .stMarkdown p, .stMarkdown li {{
        color: {subtext} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* ── [CHANGED] Bold text inside markdown should be brighter ── */
    strong, b, .stMarkdown strong {{
        color: {text} !important;
    }}

    /* ── [CHANGED] Sidebar styling ── */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar} !important;
        border-right: 1px solid {card_border} !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {subtext} !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {text} !important;
    }}
    /* Sidebar collapse button (inside open sidebar) */
    section[data-testid="stSidebar"] button[kind="header"] {{
        color: {text} !important;
    }}
    section[data-testid="stSidebar"] button[kind="header"] svg {{
        fill: {text} !important;
        stroke: {text} !important;
    }}
    /* [CHANGED] Sidebar expand/collapse arrow — comprehensive selectors
       for ALL Streamlit versions. The arrow button uses various test IDs
       and sits both inside and outside the sidebar element. */
    div[data-testid="collapsedControl"],
    div[data-testid="stSidebarCollapsedControl"],
    div[data-testid*="collapse"],
    div[data-testid*="Sidebar"] > button {{
        color: {text} !important;
    }}
    div[data-testid="collapsedControl"] button,
    div[data-testid="stSidebarCollapsedControl"] button,
    div[data-testid*="collapse"] button {{
        color: {text} !important;
    }}
    div[data-testid="collapsedControl"] svg,
    div[data-testid="stSidebarCollapsedControl"] svg,
    div[data-testid*="collapse"] svg,
    button[kind="header"] svg {{
        fill: {text} !important;
        stroke: {text} !important;
        color: {text} !important;
    }}
    /* Catch-all: any header-type button (collapse/expand arrows) */
    button[kind="header"],
    button[data-testid="baseButton-header"] {{
        color: {text} !important;
    }}
    button[kind="header"] svg,
    button[data-testid="baseButton-header"] svg {{
        fill: {text} !important;
        stroke: {text} !important;
    }}

    /* ── [CHANGED] Streamlit header / top bar ── */
    header[data-testid="stHeader"] {{
        background-color: {bg} !important;
        border-bottom: 1px solid {card_border} !important;
    }}
    /* Deploy button and hamburger menu */
    header[data-testid="stHeader"] button {{
        color: {subtext} !important;
    }}

    /* ── [CHANGED] Selectbox / dropdown – fixes white-on-white text ── */
    div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
        border-radius: 8px !important;
        color: {text} !important;
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div[data-testid="stMarkdownContainer"] {{
        color: {text} !important;
    }}
    /* Dropdown arrow icon */
    div[data-baseweb="select"] svg {{
        fill: {subtext} !important;
    }}
    /* Dropdown menu (opened) */
    ul[data-testid="stSelectboxVirtualDropdown"],
    div[data-baseweb="popover"] > div,
    ul[role="listbox"] {{
        background-color: {card} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px !important;
    }}
    /* Dropdown options */
    li[role="option"] {{
        color: {text} !important;
        background-color: {card} !important;
    }}
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {{
        background-color: {hover_bg} !important;
        color: {text} !important;
    }}
    /* Highlighted option */
    li[data-highlighted="true"],
    li[aria-selected="true"] {{
        background-color: {accent} !important;
        color: #FFFFFF !important;
    }}

    /* ── [CHANGED] Text input – fixes invisible placeholder ── */
    input[type="text"],
    .stTextInput input,
    div[data-baseweb="input"] input {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border-color: {input_border} !important;
        border-radius: 8px !important;
        caret-color: {text} !important;
    }}
    div[data-baseweb="input"] {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
        border-radius: 8px !important;
    }}
    /* Placeholder text */
    input::placeholder {{
        color: {subtext} !important;
        opacity: 0.6 !important;
    }}

    /* ── [CHANGED] Number input ── */
    .stNumberInput input,
    input[type="number"] {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border-color: {input_border} !important;
        border-radius: 8px !important;
    }}
    .stNumberInput button {{
        background-color: {card} !important;
        color: {text} !important;
        border-color: {input_border} !important;
    }}
    .stNumberInput button:hover {{
        background-color: {hover_bg} !important;
    }}

    /* ── [CHANGED] Radio buttons ── */
    .stRadio label span,
    .stRadio div[role="radiogroup"] label {{
        color: {subtext} !important;
    }}
    .stRadio div[role="radiogroup"] label:hover {{
        color: {text} !important;
    }}

    /* ── [CHANGED] Checkbox ── */
    .stCheckbox label span {{
        color: {subtext} !important;
    }}

    /* ── [CHANGED] Slider ── */
    .stSlider label,
    .stSlider p {{
        color: {subtext} !important;
    }}
    .stSlider [data-testid="stThumbValue"] {{
        color: {text} !important;
    }}
    div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {accent} !important;
        border-color: {accent} !important;
    }}

    /* ── [CHANGED] Expander styling ── */
    details[data-testid="stExpander"] {{
        background-color: {card} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px !important;
    }}
    details[data-testid="stExpander"] summary {{
        color: {text} !important;
    }}
    details[data-testid="stExpander"] summary span {{
        color: {text} !important;
    }}
    details[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background-color: {card} !important;
    }}
    /* Expander arrow icon */
    details[data-testid="stExpander"] summary svg {{
        fill: {subtext} !important;
    }}

    /* ── [CHANGED] Tabs ── */
    button[data-baseweb="tab"] {{
        color: {subtext} !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        background-color: transparent !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: {text} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {tab_active} !important;
    }}

    /* ── [CHANGED] Buttons — text color is now theme-aware ── */
    .stButton > button {{
        background-color: {accent} !important;
        color: {btn_text} !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem 1rem !important;
    }}
    .stButton > button:hover {{
        background-color: {accent_hover} !important;
        color: {btn_text} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* ── [CHANGED] Download button ── */
    .stDownloadButton > button {{
        background-color: {card} !important;
        color: {text} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    .stDownloadButton > button:hover {{
        background-color: {hover_bg} !important;
        border-color: {accent} !important;
    }}

    /* ── [CHANGED] Metric cards (custom HTML) ── */
    .metric-card {{
        background: {card} !important;
        border: 1px solid {card_border} !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin: 10px 0 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .metric-card:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
    }}
    .metric-value {{
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: {cyan} !important;
        font-family: 'Inter', sans-serif !important;
    }}
    .metric-label {{
        color: {text} !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 4px !important;
    }}
    .metric-card p {{
        color: {subtext} !important;
        font-size: 0.85rem !important;
        margin: 4px 0 0 0 !important;
    }}

    /* ── [CHANGED] Streamlit native metric widget ── */
    div[data-testid="stMetric"] {{
        background-color: {card} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }}
    div[data-testid="stMetric"] label {{
        color: {subtext} !important;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {text} !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{
        font-weight: 600 !important;
    }}

    /* ── [CHANGED] Dataframe / table ── */
    .stDataFrame {{
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    /* ── [CHANGED] Info / Success / Warning / Error alert boxes ── */
    .stAlert {{
        border-radius: 8px !important;
    }}
    div[data-testid="stAlert"] p {{
        color: inherit !important;
    }}

    /* ── [CHANGED] Progress bar ── */
    .stProgress > div > div {{
        background-color: {accent} !important;
    }}

    /* ── [CHANGED] Horizontal rule / divider ── */
    hr {{
        border-color: {card_border} !important;
        opacity: 0.5 !important;
    }}

    /* ── [CHANGED] Markdown links ── */
    a {{
        color: {accent} !important;
    }}
    a:hover {{
        color: {accent_hover} !important;
    }}

    /* ── [CHANGED] Tooltip ── */
    div[data-baseweb="tooltip"] {{
        background-color: {card} !important;
        color: {text} !important;
        border: 1px solid {card_border} !important;
    }}

    /* ── [CHANGED] Popover / BaseWeb menu overrides ── */
    div[data-baseweb="popover"] {{
        background-color: {card} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="menu"] {{
        background-color: {card} !important;
    }}
    div[data-baseweb="menu"] li {{
        color: {text} !important;
    }}
    div[data-baseweb="menu"] li:hover {{
        background-color: {hover_bg} !important;
    }}

    /* ── [CHANGED] Scrollbar styling for dark mode ── */
    ::-webkit-scrollbar {{
        width: 8px !important;
        height: 8px !important;
    }}
    ::-webkit-scrollbar-track {{
        background: {bg} !important;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {card_border} !important;
        border-radius: 4px !important;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {subtext} !important;
    }}

    /* ── [CHANGED] Footer styling ── */
    .footer-container {{
        background: {card} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
        padding: 20px !important;
        text-align: center !important;
        margin-top: 20px !important;
    }}
    .footer-container p {{
        color: {subtext} !important;
    }}
    .footer-container strong {{
        color: {text} !important;
    }}

    /* ── [CHANGED] Spinner text ── */
    .stSpinner > div {{
        color: {subtext} !important;
    }}

    /* ── [CHANGED] Label helper text (form labels above inputs) ── */
    .stSelectbox label,
    .stTextInput label,
    .stNumberInput label,
    .stSlider label,
    .stRadio label,
    .stCheckbox label,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p {{
        color: {subtext} !important;
        font-weight: 500 !important;
    }}

    /* ── [CHANGED] Streamlit bottom toolbar / running indicator ── */
    footer {{
        visibility: hidden !important;
    }}

    </style>
    """, unsafe_allow_html=True)



# [CHANGED] Default theme set to 'Dark' to match config.toml dark theme
for key, default in [('theme', 'Dark'), ('trained_models', {}),
                     ('data_loaded', False), ('demo_mode', False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title(" IntelliTrade AI")
    st.markdown("---")

    theme = st.radio(" Theme", ["Dark", "Light"],
                     index=0 if st.session_state.theme == "Dark" else 1,
                     key="theme_toggle")
    st.session_state.theme = theme
    st.markdown("---")

    demo_mode = st.checkbox(" Demo Mode (sample data)",
                            value=st.session_state.demo_mode)

    st.session_state.demo_mode = demo_mode
    if demo_mode:
        st.info(" Using simulated data")
    st.markdown("---")

    st.subheader(" Stock Selection")
    sp500_list = get_sp500_list()
    selected_sp500 = st.selectbox(
        "Select S&P 500 Stock", options=[""] + sp500_list,
        format_func=lambda x: f"{x} - {get_company_name(x)}" if x else "Select a stock")
    custom_ticker = st.text_input("Or Enter Custom Ticker", placeholder="e.g., AAPL")
    ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_sp500
    st.markdown("---")

    st.subheader(" Data Settings")
    data_period = st.selectbox("Historical Period",
                               ["1y", "2y", "5y", "10y", "max"], index=2)
    st.markdown("---")

    st.subheader(" Model Settings")
    train_size = st.slider("Train/Test Split (%)", 60, 90, 80) / 100
    with st.expander("LSTM Settings"):
        lstm_epochs = st.slider("Epochs", 10, 100, 50)
        lstm_lookback = st.slider("Look Back Period", 30, 90, 60)
    with st.expander("ARIMA Settings"):
        arima_p = st.slider("p (AR order)", 1, 10, 5)
        arima_d = st.slider("d (Differencing)", 0, 2, 1)
        arima_q = st.slider("q (MA order)", 0, 5, 0)
    st.markdown("---")

    st.subheader(" Backtesting Settings")
    initial_capital = st.number_input("Initial Capital ($)",
                                      min_value=1000, value=10000, step=1000)
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
    st.markdown("---")

    load_button = st.button(" Load & Analyze", use_container_width=True)

# ── Apply CSS ────────────────────────────────────────────────────────
load_css(st.session_state.theme)

# ── Dynamic Chart Theme ─────────────────────────────────────────────
# [CHANGED] Expanded chart theme with legend font color, tick color,
# and annotation color so ALL chart text adapts to the active theme.
def get_chart_theme(theme):

    if theme == "Dark":
        return {
            "template": "plotly_dark",
            "bg": "#0E1117",
            "paper_bg": "#0E1117",
            "font": "#FFFFFF",
            "grid": "rgba(255,255,255,0.1)",
            "legend_font": "#FFFFFF",
            "tick_font": "#D1D5DB",
            "title_font": "#FFFFFF",
            "annotation_font": "#D1D5DB",

            "close": "#00E5FF",
            "sma20": "#FFD166",
            "sma50": "#06D6A0",
            "sma200": "#EF476F"
        }

    else:
        # [CHANGED] Light mode chart colors — all text is dark (#111827)
        # so legend entries, axis ticks, and subplot titles are readable
        return {
            "template": "plotly_white",
            "bg": "#F9FAFB",
            "paper_bg": "#F9FAFB",
            "font": "#111827",
            "grid": "rgba(0,0,0,0.08)",
            "legend_font": "#111827",
            "tick_font": "#374151",
            "title_font": "#111827",
            "annotation_font": "#374151",

            "close": "#1565C0",
            "sma20": "#E65100",
            "sma50": "#1B5E20",
            "sma200": "#B71C1C"
        }

chart_theme = get_chart_theme(st.session_state.theme)


# [CHANGED] Helper that applies full theme-aware styling to any Plotly
# figure — prevents the "white legend on white background" bug that
# occurred when charts had no explicit font/color overrides.
def apply_chart_theme(fig, chart_theme, **extra_layout):
    """Apply consistent theme styling to a Plotly figure."""
    fig.update_layout(
        template=chart_theme["template"],
        paper_bgcolor=chart_theme["paper_bg"],
        plot_bgcolor=chart_theme["bg"],
        font=dict(color=chart_theme["font"], family="Inter, sans-serif"),
        title_font=dict(color=chart_theme["title_font"], size=16),
        legend=dict(
            font=dict(color=chart_theme["legend_font"], size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        **extra_layout,
    )
    fig.update_xaxes(
        tickfont=dict(color=chart_theme["tick_font"]),
        title_font=dict(color=chart_theme["font"]),
        gridcolor=chart_theme["grid"],
    )
    fig.update_yaxes(
        tickfont=dict(color=chart_theme["tick_font"]),
        title_font=dict(color=chart_theme["font"]),
        gridcolor=chart_theme["grid"],
    )
    # Update subplot annotation titles (e.g. "Price & Moving Averages")
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=chart_theme["title_font"], size=14))
    return fig


# ── Helper: safe division ────────────────────────────────────────────
def safe_pct(a, b):
    return (a / b * 100) if b else 0.0

# ══════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════
if not ticker:
    st.title(" IntelliTrade AI")
    st.markdown("### Stock Price Prediction & Automated Trading Strategy System")
    st.info(" Select a stock from the sidebar to begin analysis")

    c1, c2, c3 = st.columns(3)
    for col, val, lbl, desc in [
        (c1, "3", "AI Models", "LSTM, ARIMA, Prophet"),
        (c2, "500+", "S&P 500 Stocks", "Full market coverage"),
        (c3, "∞", "Custom Tickers", "Any stock symbol"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div><p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("###  Features")
    st.markdown("""
-  **Multiple Prediction Models**: Compare LSTM, ARIMA, and Prophet
-  **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
-  **Backtesting Engine**: Simulate trading strategies with real metrics
-  **Risk Analysis**: Sharpe Ratio, Maximum Drawdown, Win/Loss Ratio
-  **Real-time & Historical Data**: Yahoo Finance integration
-  **Customizable Theme**: Dark / Light mode toggle
""")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════════
elif load_button or st.session_state.data_loaded:
    st.session_state.data_loaded = True

    # ── Load data ────────────────────────────────────────────────────
    with st.spinner(f"Loading data for {ticker}..."):
        period_days = {'1y': 252, '2y': 504, '5y': 1260, '10y': 2520, 'max': 3780}

        if st.session_state.demo_mode:
            df = generate_demo_stock_data(ticker, period_days.get(data_period, 1260))
            real_time = get_demo_real_time_data(ticker)
            st.success(f" Loaded demo data for {ticker}")
        else:
            loader = StockDataLoader(ticker)
            df = loader.get_historical_data(period=data_period)
            real_time = loader.get_real_time_data()
            if df is None or df.empty:
                st.warning(f" Live data unavailable for {ticker}. Using demo data.")
                df = generate_demo_stock_data(ticker, period_days.get(data_period, 1260))
                real_time = get_demo_real_time_data(ticker)

        if df is None or df.empty:
            st.error(f" No data for {ticker}.")
            st.session_state.data_loaded = False
            st.stop()

        # Normalise date column
        if 'date' not in df.columns:
            df['date'] = df.get('Date', df.index)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        # Fill NaN forward instead of dropping (preserves rows for SMA-200)
        required = ['open', 'high', 'low', 'close', 'volume']
        for c in required:
            if c in df.columns:
                df[c] = df[c].ffill().bfill()
        df = df.dropna(subset=required)
        df = df.reset_index(drop=True)

        # Add technical indicators
        df = TechnicalIndicators.add_all_indicators(df)
        df = TechnicalIndicators.get_trading_signals(df)

        # Split — use rows that have all indicators (after SMA-200 warmup)
        df_clean = df.dropna(subset=['sma_200']).reset_index(drop=True)
        split_idx = int(len(df_clean) * train_size)
        train_df = df_clean.iloc[:split_idx].copy()
        test_df  = df_clean.iloc[split_idx:].copy()

    # ── Header ───────────────────────────────────────────────────────
    st.title(f" {ticker} - {get_company_name(ticker)}")

    if real_time and real_time.get('previous_close'):
        chg = real_time['current_price'] - real_time['previous_close']
        chg_pct = safe_pct(chg, real_time['previous_close'])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Price", f"${real_time['current_price']:.2f}",
                  f"{chg:.2f} ({chg_pct:+.2f}%)")
        c2.metric("Day High", f"${real_time['day_high']:.2f}")
        c3.metric("Day Low",  f"${real_time['day_low']:.2f}")
        c4.metric("Volume",   f"{real_time['volume']:,.0f}")
        pe = real_time.get('pe_ratio')
        c5.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Price & Indicators", " Model Predictions",
        " Backtesting", " Model Comparison", " Data Table"])

    # ── TAB 1: Price & Indicators ────────────────────────────────────
    with tab1:
        st.subheader("Historical Price & Technical Indicators")

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Moving Averages', 'RSI', 'MACD',
                            'Bollinger Bands'),
            row_heights=[0.4, 0.2, 0.2, 0.2])

        fig.add_trace(go.Scatter(x=df['date'], y=df['close'],
            name='Close', line=dict(color=chart_theme["close"], width=2)), row=1, col=1)
            
        fig.add_trace(go.Scatter(x=df['date'], y=df['sma_20'],
            name='SMA 20', line=dict(color=chart_theme["sma20"], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['sma_50'],
            name='SMA 50', line=dict(color=chart_theme["sma50"], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['sma_200'],
            name='SMA 200', line=dict(color=chart_theme["sma200"], width=1)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'],
            name='RSI', line=dict(color='#FFD93D', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.add_trace(go.Scatter(x=df['date'], y=df['macd'],
            name='MACD', line=dict(color='#6BCF7F', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['macd_signal'],
            name='Signal', line=dict(color='#FF6B9D', width=2)), row=3, col=1)
        fig.add_trace(go.Bar(x=df['date'], y=df['macd_diff'],
            name='Histogram', marker_color='#C1A1D3'), row=3, col=1)

        fig.add_trace(go.Scatter(x=df['date'], y=df['bb_high'],
            name='BB High', line=dict(color='#FF8787', width=1)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['bb_mid'],
            name='BB Mid', line=dict(color='#4ECDC4', width=2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['bb_low'],
            name='BB Low', line=dict(color='#95E1D3', width=1)), row=4, col=1)

        # [CHANGED] Use apply_chart_theme() for full theme coverage
        apply_chart_theme(fig, chart_theme,
            height=1000, showlegend=True, hovermode='x unified')
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=4, col=1)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(" Recent Trading Signals")
        sig_df = df[df['signal'] != 0].tail(10)[
            ['date', 'close', 'signal', 'rsi', 'macd']].copy()
        sig_df['signal'] = sig_df['signal'].map({1: ' BUY', -1: ' SELL'})
        st.dataframe(sig_df, use_container_width=True)

    # ── TAB 2: Model Predictions ─────────────────────────────────────
    with tab2:
        st.subheader(" AI Model Predictions")
        prediction_days = st.slider("Prediction Horizon (days)", 7, 90, 30)

        if st.button("Train Models & Predict"):
            progress = st.progress(0, text="Initializing...")
            model_results = {}

            # ---- LSTM ---
            try:
                progress.progress(5, text="Training LSTM...")
                from models.lstm_model import LSTMModel
                lstm = LSTMModel(look_back=lstm_lookback)
                X_train, y_train = lstm.prepare_data(train_df, fit_scaler=True)
                X_test, y_test   = lstm.prepare_data(test_df, fit_scaler=False)

                if X_train.shape[0] > 0:
                    lstm.train(X_train, y_train, epochs=lstm_epochs, batch_size=32)
                    lstm_future = lstm.predict_future(df_clean, days=prediction_days)

                    if X_test.shape[0] > 0:
                        lstm_pred = lstm.predict(X_test).flatten()
                        test_actual = test_df['close'].values[lstm_lookback:]
                        min_len = min(len(test_actual), len(lstm_pred))
                        lstm_metrics = lstm.evaluate(
                            test_actual[:min_len], lstm_pred[:min_len])
                    else:
                        lstm_metrics = {}

                    model_results['lstm'] = {
                        'future': lstm_future, 'metrics': lstm_metrics}
                    progress.progress(35, text="LSTM done ")
                else:
                    st.warning("Not enough data for LSTM training")
            except Exception as e:
                st.warning(f"LSTM error: {e}")

            # ---- ARIMA ---
            try:
                progress.progress(40, text="Training ARIMA...")
                from models.arima_model import ARIMAModel
                arima = ARIMAModel(order=(arima_p, arima_d, arima_q))
                arima.train(train_df)
                arima_future = arima.predict(steps=prediction_days)

                # In-sample test evaluation
                arima_test_pred = arima.predict(steps=len(test_df))
                if arima_test_pred is not None and len(arima_test_pred) > 0:
                    min_len = min(len(test_df), len(arima_test_pred))
                    arima_metrics = arima.evaluate(
                        test_df['close'].values[:min_len],
                        arima_test_pred[:min_len])
                else:
                    arima_metrics = {}

                model_results['arima'] = {
                    'future': arima_future, 'metrics': arima_metrics}
                progress.progress(55, text="ARIMA done ")
            except Exception as e:
                st.warning(f"ARIMA error: {e}")

            # ---- Prophet ---
            try:
                progress.progress(60, text="Training Prophet...")

                from models.prophet_model import ProphetModel
                prophet = ProphetModel()
                prophet.train(train_df)
                prophet_forecast = prophet.predict(periods=prediction_days)
                prophet_future = prophet_forecast['yhat'].values[-prediction_days:]

                prophet_test_pred = prophet.predict_on_data(test_df)
                if prophet_test_pred is not None:
                    prophet_metrics = prophet.evaluate(
                        test_df['close'].values, prophet_test_pred)
                else:
                    prophet_metrics = {}

                model_results['prophet'] = {
                    'future': prophet_future, 'metrics': prophet_metrics}
                progress.progress(90, text="Prophet done ")
            except Exception as e:
                st.warning(f"Prophet error: {e}")

            progress.progress(100, text="All models trained!")
            st.session_state.trained_models = model_results
            st.success(" Models trained successfully!")

        # ---- Display predictions ---
        if st.session_state.trained_models:
            models = st.session_state.trained_models
            last_date = df['date'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=prediction_days)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['close'],
                name='Historical', line=dict(color='#4ECDC4', width=2)))

            colors = {'lstm': '#FF6B6B', 'arima': '#95E1D3', 'prophet': '#FFD93D'}
            labels = {'lstm': 'LSTM', 'arima': 'ARIMA', 'prophet': 'Prophet'}
            for name, color in colors.items():
                if name in models and models[name].get('future') is not None:
                    vals = models[name]['future']
                    fig.add_trace(go.Scatter(
                        x=future_dates[:len(vals)], y=vals,
                        name=f'{labels[name]} Prediction',
                        line=dict(color=color, width=2, dash='dash')))

            # [CHANGED] Apply full chart theme — was missing before,
            # causing white legend text on white background in light mode
            apply_chart_theme(fig, chart_theme,
                title="Multi-Model Price Predictions",
                xaxis_title="Date", yaxis_title="Price ($)",
                height=600, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            # Prediction table
            st.subheader(" Prediction Values")
            pred_data = {'Date': future_dates}
            for name in ['lstm', 'arima', 'prophet']:
                if name in models and models[name].get('future') is not None:
                    vals = models[name]['future']
                    pred_data[labels[name]] = list(vals) + [np.nan] * (prediction_days - len(vals))
            pred_df = pd.DataFrame(pred_data)
            num_cols = [c for c in pred_df.columns if c != 'Date']
            if num_cols:
                pred_df['Average'] = pred_df[num_cols].mean(axis=1)
            st.dataframe(pred_df.head(15), use_container_width=True)

    # ── TAB 3: Backtesting ───────────────────────────────────────────
    with tab3:
        st.subheader(" Trading Strategy Backtesting")

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                bt = Backtester(initial_capital=initial_capital,
                                transaction_cost=transaction_cost)
                metrics = bt.run_backtest(df_clean)

            if metrics:
                st.success(" Backtest completed!")

                c1, c2, c3, c4 = st.columns(4)
                for col, lbl, val in [
                    (c1, "Total Return", f"{metrics['total_return']:.2f}%"),
                    (c2, "Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
                    (c3, "Max Drawdown", f"{metrics['max_drawdown']:.2f}%"),
                    (c4, "Win Rate",     f"{metrics['win_rate']:.1f}%"),
                ]:
                    with col:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">{lbl}</div>
                            <div class="metric-value">{val}</div>
                        </div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Initial Capital", f"${metrics['initial_capital']:,.2f}")
                c2.metric("Final Capital",   f"${metrics['final_capital']:,.2f}")
                c3.metric("Total Trades",    metrics['total_trades'])

                # Equity curve
                st.subheader(" Equity Curve")
                pv = metrics['portfolio_values']
                dates_eq = df_clean['date'].values[:len(pv)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates_eq, y=pv,
                    name='Portfolio Value', fill='tozeroy',
                    line=dict(color='#4ECDC4', width=2)))
                fig.add_hline(y=initial_capital, line_dash="dash",
                              line_color="red", annotation_text="Initial")
                # [CHANGED] Apply full chart theme to equity curve
                apply_chart_theme(fig, chart_theme,
                    title="Portfolio Value Over Time",
                    xaxis_title="Date", yaxis_title="Value ($)", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Trade history
                st.subheader(" Trade History")
                trades_df = pd.DataFrame(metrics['trades'])
                if not trades_df.empty:
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades executed")
            else:
                st.warning("Backtest returned no results.")

    # ── TAB 4: Model Comparison ──────────────────────────────────────
    with tab4:
        st.subheader(" Model Performance Comparison")

        if st.session_state.trained_models:
            models = st.session_state.trained_models
            rows = []
            labels = {'lstm': 'LSTM', 'arima': 'ARIMA', 'prophet': 'Prophet'}
            for name in ['lstm', 'arima', 'prophet']:
                m = models.get(name, {}).get('metrics', {})
                if m:
                    rows.append({
                        'Model': labels[name],
                        'MAE': m.get('MAE', np.nan),
                        'RMSE': m.get('RMSE', np.nan),
                        'MAPE (%)': m.get('MAPE', np.nan),
                        'R² Score': m.get('R2', np.nan),
                    })
            if rows:
                comp_df = pd.DataFrame(rows)
                st.dataframe(comp_df, use_container_width=True)
                fig = go.Figure()
                for metric in ['MAE', 'RMSE', 'MAPE (%)']:
                    fig.add_trace(go.Bar(name=metric,
                        x=comp_df['Model'], y=comp_df[metric]))
                # [CHANGED] Apply full chart theme to model comparison
                apply_chart_theme(fig, chart_theme,
                    title="Model Metrics Comparison",
                    barmode='group', height=500)
                st.plotly_chart(fig, use_container_width=True)
                if len(rows) > 1:
                    best_mae = comp_df.loc[comp_df['MAE'].idxmin(), 'Model']
                    best_r2  = comp_df.loc[comp_df['R² Score'].idxmax(), 'Model']
                    st.success(f" Best by MAE: **{best_mae}**")
                    st.success(f" Best by R²: **{best_r2}**")
            else:
                st.info("Train models first to see comparison")
        else:
            st.info("Train models in the 'Model Predictions' tab first")

    # ── TAB 5: Data Table ────────────────────────────────────────────
    with tab5:
        st.subheader(" Historical Data with Indicators")
        c1, c2 = st.columns(2)
        show_rows = c1.slider("Rows to display", 10, 100, 50)
        sort_order = c2.radio("Sort order", ["Most Recent", "Oldest First"])
        display_df = (df.tail(show_rows) if sort_order == "Most Recent"
                      else df.head(show_rows))
        st.dataframe(display_df, use_container_width=True, height=600)
        csv = df.to_csv(index=False)
        st.download_button(" Download Full Dataset as CSV", data=csv,
                           file_name=f"{ticker}_data_{data_period}.csv",
                           mime="text/csv")

# ── Footer ───────────────────────────────────────────────────────────
# [CHANGED] Use footer-container class for proper dark mode styling
st.markdown("---")
st.markdown("""<div class='footer-container'>
<p> <strong>IntelliTrade AI</strong> — Stock Price Prediction &
Automated Trading Strategy System</p>
<p><small>Built with Streamlit | Data from Yahoo Finance |
Models: LSTM, ARIMA, Prophet</small></p>
</div>""", unsafe_allow_html=True)