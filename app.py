"""
========================================================
  app.py — Streamlit House Price Predictor
  Run: streamlit run app.py
  Requires: models/catboost_model.cbm
            data/houses_clean_new_one.csv
========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor, Pool

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Lanka House Price AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main { background: #f7f6f2; }
.block-container { padding: 2rem 2rem 2rem 2rem; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(229,160,60,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.6);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}
.hero-badge {
    display: inline-block;
    background: rgba(229,160,60,0.2);
    border: 1px solid rgba(229,160,60,0.4);
    color: #e5a03c;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.price-card {
    background: linear-gradient(135deg, #0f3460, #1a1a2e);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 20px 60px rgba(15,52,96,0.4);
    border: 1px solid rgba(229,160,60,0.3);
}
.price-label { font-size: 0.8rem; color: rgba(255,255,255,0.5); letter-spacing: 0.15em; text-transform: uppercase; }
.price-value { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #e5a03c; line-height: 1.1; margin: 0.5rem 0; }
.price-range { font-size: 0.85rem; color: rgba(255,255,255,0.45); }

.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e8e6e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stat-label { font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 0.1em; }
.stat-value { font-size: 1.4rem; font-weight: 600; color: #1a1a2e; margin-top: 0.2rem; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #1a1a2e;
    border-bottom: 2px solid #e5a03c;
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
    display: inline-block;
}

.insight-box {
    background: white;
    border-left: 4px solid #e5a03c;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

[data-testid="stSidebar"] {
    background: #1a1a2e;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] h2 {
    font-family: 'DM Serif Display', serif !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = CatBoostRegressor()
    m.load_model("models/catboost_model.cbm")
    return m

@st.cache_data
def load_data():
    return pd.read_csv("data/houses_clean_new_one.csv")

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

model    = load_model()
df       = load_data()
explainer = get_explainer(model)

FEATURES = ["bedrooms", "bathrooms", "land_perches", "floor_sqft", "district", "suburb"]
CAT_IDX  = [FEATURES.index(c) for c in ["district", "suburb"]]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Property Details")
    st.markdown("---")

    districts = sorted(df["district"].dropna().unique().tolist())
    district  = st.selectbox("📍 District", districts,
                              index=districts.index("Colombo") if "Colombo" in districts else 0)

    suburbs_in = sorted(df[df["district"] == district]["suburb"].dropna().unique().tolist())
    suburb     = st.selectbox("🏘 Suburb", suburbs_in if suburbs_in else ["Other"])

    st.markdown("---")
    bedrooms     = st.slider("🛏 Bedrooms",          1, 10, 4)
    bathrooms    = st.slider("🚿 Bathrooms",          1, 10, 2)
    land_perches = st.number_input("🌿 Land (Perches)", 1.0, 400.0, 12.0, step=0.5)
    floor_sqft   = st.number_input("📐 Floor Area (sqft)", 100, 15000, 2500, step=100)

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:rgba(255,255,255,0.4);'>CatBoost Regressor · SHAP Explainability<br>Data: lankapropertyweb.com</div>", unsafe_allow_html=True)


# ── Prediction ────────────────────────────────────────────────────────────────
input_df = pd.DataFrame([{
    "bedrooms"    : bedrooms,
    "bathrooms"   : bathrooms,
    "land_perches": land_perches,
    "floor_sqft"  : floor_sqft,
    "district"    : district,
    "suburb"      : suburb,
}])
pool      = Pool(input_df, cat_features=CAT_IDX)
pred_log  = model.predict(pool)[0]
pred_lkr  = np.expm1(pred_log)
pred_m    = pred_lkr / 1e6
pred_low  = pred_lkr * 0.80 / 1e6
pred_high = pred_lkr * 1.20 / 1e6

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Sri Lanka House<br>Price Predictor</div>
    <div class="hero-subtitle">Machine learning model trained on 3,806 real listings · lankapropertyweb.com</div>
</div>
""", unsafe_allow_html=True)

# ── Price + Stats row ─────────────────────────────────────────────────────────
col_price, col_stats = st.columns([1, 2])

with col_price:
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Price</div>
        <div class="price-value">LKR {pred_m:.2f}M</div>
        <div class="price-range">Range: {pred_low:.2f}M – {pred_high:.2f}M</div>
    </div>
    """, unsafe_allow_html=True)

with col_stats:
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Bedrooms</div><div class="stat-value">{bedrooms}</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Bathrooms</div><div class="stat-value">{bathrooms}</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Land</div><div class="stat-value">{land_perches}p</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Floor</div><div class="stat-value">{floor_sqft}sqft</div></div>', unsafe_allow_html=True)

    # Model accuracy info
    st.markdown("""
    <div style="margin-top:1rem; padding:1rem; background:white; border-radius:12px; border:1px solid #e8e6e0;">
        <div style="display:flex; justify-content:space-around; text-align:center;">
            <div><div style="font-size:0.7rem;color:#999;text-transform:uppercase;letter-spacing:0.1em;">R² Score</div><div style="font-size:1.3rem;font-weight:600;color:#0f3460;">0.75</div></div>
            <div><div style="font-size:0.7rem;color:#999;text-transform:uppercase;letter-spacing:0.1em;">Training Data</div><div style="font-size:1.3rem;font-weight:600;color:#0f3460;">3,806</div></div>
            <div><div style="font-size:0.7rem;color:#999;text-transform:uppercase;letter-spacing:0.1em;">Algorithm</div><div style="font-size:1.3rem;font-weight:600;color:#0f3460;">CatBoost</div></div>
            <div><div style="font-size:0.7rem;color:#999;text-transform:uppercase;letter-spacing:0.1em;">Districts</div><div style="font-size:1.3rem;font-weight:600;color:#0f3460;">20</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── SHAP Explanation ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title" style="color: #D4E3F3; font-size: 24px;">Why this price?</div>',
    unsafe_allow_html=True
)
st.caption("SHAP values show which features pushed the price **up** (red) or **down** (blue) from the average Sri Lanka house price.")

shap_vals = explainer.shap_values(pool)
explanation = shap.Explanation(
    values        = shap_vals[0],
    base_values   = explainer.expected_value,
    data          = input_df.iloc[0].values,
    feature_names = FEATURES,
)

col_wf, col_bar = st.columns([3, 2])

with col_wf:
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#ffffff')
    shap.waterfall_plot(explanation, show=False, max_display=6)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_bar:
    sv = pd.Series(shap_vals[0], index=FEATURES).sort_values()
    colors = ["#DC2626" if v > 0 else "#2563EB" for v in sv]
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor('#ffffff')
    sv.plot(kind="barh", ax=ax2, color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP Value", fontsize=10)
    ax2.set_title("Feature Contributions", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ── Key insights ──────────────────────────────────────────────────────────────
sv_series = pd.Series(shap_vals[0], index=FEATURES)
top_pos = sv_series[sv_series > 0].idxmax() if (sv_series > 0).any() else None
top_neg = sv_series[sv_series < 0].idxmin() if (sv_series < 0).any() else None

col_i1, col_i2 = st.columns(2)
if top_pos:
    with col_i1:
        st.markdown(f'<div class="insight-box">📈 <strong>{top_pos.replace("_"," ").title()}</strong> is pushing the price <strong>up</strong> for this property.</div>', unsafe_allow_html=True)
if top_neg:
    with col_i2:
        st.markdown(f'<div class="insight-box">📉 <strong>{top_neg.replace("_"," ").title()}</strong> is pulling the price <strong>down</strong> for this property.</div>', unsafe_allow_html=True)

# ── District comparison ─────────────────────────────────────────────────────── 
st.markdown('<div class="section-title"  style="color: #D4E3F3; font-size: 24px;">District Price Comparison</div>', unsafe_allow_html=True)

dist_med = df.groupby("district")["price_lkr"].median().sort_values(ascending=True) / 1e6
colors_bar = ["#e5a03c" if d == district else "#0f3460" for d in dist_med.index]

fig3, ax3 = plt.subplots(figsize=(12, 5))
fig3.patch.set_facecolor('#ffffff')
dist_med.plot(kind="barh", ax=ax3, color=colors_bar)
ax3.set_xlabel("Median House Price (LKR Millions)", fontsize=11)
ax3.set_title("Median House Price by District  (selected = gold)", fontsize=12, fontweight="bold")
gold_patch = mpatches.Patch(color="#e5a03c", label=f"Selected: {district}")
ax3.legend(handles=[gold_patch], fontsize=10)
ax3.tick_params(axis="both", labelsize=9)
plt.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close()

# ── Similar listings ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title"  style="color: #D4E3F3; font-size: 24px;">Similar Listings</div>', unsafe_allow_html=True)

mask = (
    (df["district"] == district) &
    (df["bedrooms"].between(bedrooms - 1, bedrooms + 1)) &
    (df["price_lkr"].between(pred_lkr * 0.5, pred_lkr * 1.5))
)
similar = df[mask].copy()
similar["Price (LKR M)"] = (similar["price_lkr"] / 1e6).round(2)

if len(similar) > 0:
    show_cols = [c for c in ["suburb", "bedrooms", "bathrooms", "land_perches", "floor_sqft", "Price (LKR M)"] if c in similar.columns]
    st.dataframe(similar[show_cols].head(8).reset_index(drop=True), use_container_width=True)
else:
    st.info("No similar listings found for this combination in the dataset.")