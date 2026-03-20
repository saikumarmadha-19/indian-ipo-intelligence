import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Indian IPO Intelligence Platform",
    page_icon  = "📈",
    layout     = "wide"
)

# ── Load models and data ─────────────────────────────────────
@st.cache_resource
def load_models():
    with open("src/xgb_regressor.pkl", "rb") as f:
        reg_model = pickle.load(f)
    with open("src/xgb_classifier.pkl", "rb") as f:
        cls_model = pickle.load(f)
    with open("src/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("src/feature_list.pkl", "rb") as f:
        features = pickle.load(f)
    return reg_model, cls_model, le, features

@st.cache_data
def load_data():
    df = pd.read_csv("data/validated/ipo_validated.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

reg_model, cls_model, le, FEATURES = load_models()
df = load_data()

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "IPO Predictor",
    "Market Analysis",
    "Top & Bottom IPOs"
])

# ════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "Home":
    st.title("📈 Indian IPO Intelligence Platform")
    st.markdown("### End-to-end ML platform for IPO listing gain prediction")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total IPOs Analyzed", len(df))
    with col2:
        st.metric("Avg Listing Gain",
                  f"{df['listing_gains_pct'].mean():.1f}%")
    with col3:
        st.metric("Best IPO Gain",
                  f"{df['listing_gains_pct'].max():.1f}%")
    with col4:
        st.metric("Worst IPO Gain",
                  f"{df['listing_gains_pct'].min():.1f}%")

    st.markdown("---")

    # Year-wise IPO count
    yearly = df.groupby("Year").size().reset_index(name="count")
    fig = px.bar(yearly, x="Year", y="count",
                 title="IPOs per Year (2010–2021)",
                 color="count",
                 color_continuous_scale="teal")
    st.plotly_chart(fig, use_container_width=True)

    # Performance breakdown
    col1, col2 = st.columns(2)
    with col1:
        cat = df["performance_category"].value_counts().reset_index()
        cat.columns = ["category", "count"]
        color_map = {
            "Blockbuster": "#1D9E75", "Strong": "#5DCAA5",
            "Moderate": "#FAC775",   "Weak": "#F0997B",
            "Loss": "#E24B4A"
        }
        fig2 = px.pie(cat, names="category", values="count",
                      title="Performance Category Breakdown",
                      color="category",
                      color_discrete_map=color_map,
                      hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.histogram(df, x="listing_gains_pct", nbins=30,
                            title="Listing Gains Distribution",
                            color_discrete_sequence=["#1D9E75"])
        fig3.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — IPO PREDICTOR
# ════════════════════════════════════════════════════════════
elif page == "IPO Predictor":
    st.title("🔮 IPO Listing Gain Predictor")
    st.markdown("Enter IPO details to predict listing gain and performance category.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        issue_size    = st.number_input("Issue Size (in Crores ₹)",
                                         min_value=1.0, value=500.0, step=10.0)
        issue_price   = st.number_input("Issue Price (₹)",
                                         min_value=1.0, value=100.0, step=1.0)
        total_sub     = st.slider("Total Subscription (x times)",
                                   0.0, 500.0, 50.0)
        qib_sub       = st.slider("QIB Subscription (x)",
                                   0.0, 300.0, 30.0)

    with col2:
        hni_sub       = st.slider("HNI Subscription (x)",
                                   0.0, 500.0, 40.0)
        rii_sub       = st.slider("RII Subscription (x)",
                                   0.0, 100.0, 20.0)
        nifty_return  = st.slider("Nifty 30-Day Return (%) at IPO date",
                                   -20.0, 30.0, 5.0)
        month         = st.selectbox("Listing Month",
                                      list(range(1, 13)),
                                      format_func=lambda x:
                                      ["Jan","Feb","Mar","Apr","May","Jun",
                                       "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

    quarter = (month - 1) // 3 + 1

    input_data = pd.DataFrame([{
        "issue_size_cr"     : issue_size,
        "qib_subscription"  : qib_sub,
        "hni_subscription"  : hni_sub,
        "rii_subscription"  : rii_sub,
        "total_subscription": total_sub,
        "issue_price"       : issue_price,
        "nifty_30d_return"  : nifty_return,
        "Month"             : month,
        "Quarter"           : quarter,
        "Year"              : 2024
    }])[FEATURES]

    if st.button("Predict Listing Gain", type="primary"):
        # Regression prediction
        pred_gain = reg_model.predict(input_data)[0]

        # Classification prediction
        pred_class_idx = cls_model.predict(input_data)[0]
        pred_class     = le.inverse_transform([pred_class_idx])[0]

        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            color = "normal" if pred_gain > 0 else "inverse"
            st.metric("Predicted Listing Gain",
                      f"{pred_gain:.1f}%", delta=f"{pred_gain:.1f}%")
        with col2:
            st.metric("Performance Category", pred_class)
        with col3:
            verdict = "APPLY" if pred_gain > 10 else \
                      "NEUTRAL" if pred_gain > 0 else "AVOID"
            st.metric("Verdict", verdict)

        # SHAP explanation
        st.markdown("---")
        st.subheader("Why did the model predict this? (SHAP Explanation)")

        explainer  = shap.Explainer(reg_model)
        shap_vals  = explainer(input_data)

        fig_shap, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_vals[0], show=False)
        st.pyplot(fig_shap)
        plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 3 — MARKET ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "Market Analysis":
    st.title("📊 Market Analysis")
    st.markdown("---")

    # Subscription vs gains
    fig1 = px.scatter(df, x="total_subscription",
                      y="listing_gains_pct",
                      color="performance_category",
                      hover_name="ipo_name",
                      title="Subscription vs Listing Gains",
                      size="issue_size_cr")
    st.plotly_chart(fig1, use_container_width=True)

    # Nifty vs IPO gains
    fig2 = px.scatter(
        df.dropna(subset=["nifty_30d_return"]),
        x="nifty_30d_return",
        y="listing_gains_pct",
        color="Year",
        hover_name="ipo_name",
        title="Market Mood vs IPO Listing Gains"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    numeric_cols = ["issue_size_cr", "qib_subscription",
                    "hni_subscription", "rii_subscription",
                    "total_subscription", "issue_price",
                    "listing_gains_pct", "nifty_30d_return"]
    corr = df[numeric_cols].corr()
    fig3 = px.imshow(corr, title="Feature Correlation Heatmap",
                     color_continuous_scale="RdBu",
                     zmin=-1, zmax=1, text_auto=".2f")
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — TOP & BOTTOM IPOs
# ════════════════════════════════════════════════════════════
elif page == "Top & Bottom IPOs":
    st.title("🏆 Top & Bottom IPOs")
    st.markdown("---")

    year_filter = st.selectbox("Filter by Year",
                                ["All"] + sorted(df["Year"].unique().tolist(),
                                                  reverse=True))

    filtered = df if year_filter == "All" else \
               df[df["Year"] == year_filter]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 IPOs")
        top10 = filtered.nlargest(10, "listing_gains_pct")[
            ["ipo_name", "listing_gains_pct", "Year"]
        ]
        fig_t = px.bar(top10, x="listing_gains_pct", y="ipo_name",
                       orientation="h",
                       color_discrete_sequence=["#1D9E75"],
                       title="Top 10 by Listing Gain")
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.subheader("Bottom 10 IPOs")
        bot10 = filtered.nsmallest(10, "listing_gains_pct")[
            ["ipo_name", "listing_gains_pct", "Year"]
        ]
        fig_b = px.bar(bot10, x="listing_gains_pct", y="ipo_name",
                       orientation="h",
                       color_discrete_sequence=["#E24B4A"],
                       title="Bottom 10 by Listing Gain")
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")
    st.subheader("Full IPO Table")
    st.dataframe(
        filtered[["ipo_name", "Date", "issue_price",
                  "listing_gains_pct", "total_subscription",
                  "performance_category"]].sort_values(
            "listing_gains_pct", ascending=False
        ),
        use_container_width=True
    )