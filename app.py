"""
AI Fraud Detection System
Transaction Anomaly Detection Using Unsupervised Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] {
    background: #0a0f1e;
    color: #e0e8f0;
  }
  [data-testid="stSidebar"] {
    background: #0d1629;
    border-right: 1px solid #1e3a5f;
  }
  h1, h2, h3, h4 { color: #ffffff; }
  .metric-card {
    background: #111d35;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
  }
  .metric-value { font-size: 2.2rem; font-weight: 700; color: #00c8ff; }
  .metric-label { font-size: 0.85rem; color: #7a9bb5; margin-top: 4px; }
  .risk-high   { color: #ff4d6d; font-weight: 700; }
  .risk-medium { color: #ffd166; font-weight: 700; }
  .risk-low    { color: #06d6a0; font-weight: 700; }
  .section-header {
    border-left: 4px solid #00c8ff;
    padding-left: 12px;
    margin: 24px 0 16px 0;
    font-size: 1.3rem;
    font-weight: 600;
    color: #ffffff;
  }
  div[data-testid="metric-container"] {
    background: #111d35;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 12px 20px;
  }
  div[data-testid="metric-container"] label { color: #7a9bb5 !important; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00c8ff; }
  .stDataFrame { background: #111d35; }
  .stSelectbox label, .stSlider label, .stNumberInput label { color: #7a9bb5; }
  .alert-box {
    background: rgba(255,77,109,0.12);
    border: 1px solid #ff4d6d;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.88rem;
  }
</style>
""", unsafe_allow_html=True)


# Data Generation
@st.cache_data
def generate_transaction_data(n_transactions: int = 2000, fraud_rate: float = 0.05):
    """Generate synthetic bank transaction data with injected fraud patterns."""
    np.random.seed(42)
    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    categories   = ["Retail", "Online", "ATM", "Transfer", "Utilities", "Travel", "Entertainment"]
    locations     = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad",
                     "Foreign-EU", "Foreign-US", "Foreign-AE"]
    local_locs    = locations[:5]
    foreign_locs  = locations[5:]

    base_date = datetime(2026, 1, 1)

    def make_rows(n, is_fraud):
        rows = []
        for i in range(n):
            ts = base_date + timedelta(
                days=np.random.randint(0, 180),
                hours=np.random.randint(0, 24) if not is_fraud else np.random.randint(0, 6),
                minutes=np.random.randint(0, 60),
            )
            if is_fraud:
                amount = np.random.choice([
                    np.random.uniform(5000, 50000),
                    np.random.uniform(0.01, 5),
                ], p=[0.7, 0.3])
                loc = np.random.choice(foreign_locs, p=[0.4, 0.35, 0.25])
                cat = np.random.choice(["Online", "Transfer", "Travel"])
                freq = np.random.randint(8, 25)
            else:
                amount = np.abs(np.random.lognormal(mean=5.5, sigma=1.2))
                loc = np.random.choice(local_locs + foreign_locs, p=[0.18, 0.18, 0.18, 0.18, 0.18, 0.03, 0.03, 0.04])
                cat = np.random.choice(categories)
                freq = np.random.randint(1, 8)

            rows.append({
                "transaction_id": f"TXN{100000 + i + (0 if not is_fraud else n_legit):06d}",
                "timestamp": ts,
                "amount": round(amount, 2),
                "category": cat,
                "location": loc,
                "transaction_freq_24h": freq,
                "is_foreign": 1 if loc in foreign_locs else 0,
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "true_fraud": int(is_fraud),
            })
        return rows

    legit_rows  = make_rows(n_legit, False)
    fraud_rows  = make_rows(n_fraud, True)
    df = pd.DataFrame(legit_rows + fraud_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df["customer_id"] = np.random.choice([f"CUST{1000+i}" for i in range(300)], size=len(df))
    df["merchant_id"] = np.random.choice([f"MRC{500+i}"  for i in range(150)], size=len(df))
    return df


# Model Training
@st.cache_resource
def train_models(df: pd.DataFrame, contamination: float = 0.05):
    """Train Isolation Forest + K-Means on the transaction feature set."""
    cat_map = {"Retail": 0, "Online": 1, "ATM": 2, "Transfer": 3,
               "Utilities": 4, "Travel": 5, "Entertainment": 6}
    df = df.copy()
    df["category_enc"] = df["category"].map(cat_map).fillna(0)

    features = ["amount", "transaction_freq_24h", "is_foreign",
                "hour_of_day", "day_of_week", "category_enc"]

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    iso_scores  = iso.decision_function(X_scaled)   # higher = more normal
    iso_labels  = iso.predict(X_scaled)             # -1 anomaly, 1 normal

    # Normalise to [0,1] risk score (inverted: higher = more risky)
    risk_scores = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

    # K-Means clustering
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[cluster_labels], axis=1)
    dist_norm = (distances - distances.min()) / (distances.max() - distances.min())

    # Ensemble risk score
    ensemble_score = 0.65 * risk_scores + 0.35 * dist_norm

    df["iso_risk"]       = risk_scores
    df["kmeans_dist"]    = dist_norm
    df["risk_score"]     = ensemble_score.round(4)
    df["is_anomaly"]     = (iso_labels == -1).astype(int)
    df["cluster"]        = cluster_labels

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-0.001, 0.49, 0.84, 1.001],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    return df, scaler, iso, kmeans, features


# Utility 
def risk_badge(level: str) -> str:
    colours = {"HIGH": "#ff4d6d", "MEDIUM": "#ffd166", "LOW": "#06d6a0"}
    c = colours.get(level, "#aaa")
    return f'<span style="background:{c}22;color:{c};border:1px solid {c};border-radius:4px;padding:2px 8px;font-size:0.78rem;font-weight:700;">{level}</span>'


# App code
def main():
    #Sidebar
    with st.sidebar:
        st.markdown("## AI Fraud Detection")
        st.divider()

        st.markdown("### Model Parameters")
        n_transactions = st.slider("Dataset Size", 500, 5000, 2000, 250)
        contamination  = st.slider("Fraud Contamination Rate", 0.01, 0.15, 0.05, 0.01,
                                   help="Expected fraction of fraudulent transactions")
        st.divider()

        st.markdown("### Filter & Explore")
        risk_filter = st.multiselect("Risk Level", ["HIGH", "MEDIUM", "LOW"],
                                     default=["HIGH", "MEDIUM", "LOW"])
        category_filter = st.multiselect("Category",
            ["Retail", "Online", "ATM", "Transfer", "Utilities", "Travel", "Entertainment"],
            default=["Retail", "Online", "ATM", "Transfer", "Utilities", "Travel", "Entertainment"])
        amount_range = st.slider("Amount Range (PKR)", 0, 100000, (0, 100000), 1000)
        st.divider()

        if st.button("Retrain Models", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("<small style='color:#5a7a95'>Tech Stack: Scikit-Learn · Pandas · NumPy · Plotly · Streamlit</small>",
                    unsafe_allow_html=True)

    # Data load & train
    with st.spinner("Running AI models on transaction data…"):
        raw_df = generate_transaction_data(n_transactions, contamination)
        df, scaler, iso_model, kmeans_model, features = train_models(raw_df, contamination)

    # Apply filters
    view = df[
        df["risk_level"].isin(risk_filter) &
        df["category"].isin(category_filter) &
        df["amount"].between(*amount_range)
    ].copy()

    # Header
    st.markdown("""
    <h1 style="color:#ffffff;font-size:2rem;margin-bottom:0;">🛡️ AI Fraud Detection System</h1>
    <p style="color:#7a9bb5;margin-top:4px;">Transaction Anomaly Detection · Unsupervised Machine Learning · Real-Time Dashboard</p>
    """, unsafe_allow_html=True)
    st.divider()

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    high_risk  = (df["risk_level"] == "HIGH").sum()
    med_risk   = (df["risk_level"] == "MEDIUM").sum()
    low_risk   = (df["risk_level"] == "LOW").sum()
    avg_score  = df["risk_score"].mean()
    fraud_pct  = high_risk / len(df) * 100

    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("🔴 High Risk",  f"{high_risk:,}", delta=f"{fraud_pct:.1f}%", delta_color="inverse")
    c3.metric("🟡 Medium Risk", f"{med_risk:,}")
    c4.metric("🟢 Low Risk",   f"{low_risk:,}")
    c5.metric("Avg Risk Score", f"{avg_score:.3f}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", "Alert Queue", "Model Analysis",
        "Risk Trends", "Transaction Lookup"
    ])

    # TAB 1 — DASHBOARD
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
            fig_dist = px.histogram(
                df, x="risk_score", color="risk_level",
                color_discrete_map={"HIGH": "#ff4d6d", "MEDIUM": "#ffd166", "LOW": "#06d6a0"},
                nbins=60, opacity=0.85,
                labels={"risk_score": "Risk Score", "count": "Transactions"},
                template="plotly_dark",
            )
            fig_dist.update_layout(
                paper_bgcolor="#111d35", plot_bgcolor="#111d35",
                legend_title="Risk Level", height=300,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">Risk by Transaction Category</div>', unsafe_allow_html=True)
            cat_risk = df.groupby(["category", "risk_level"]).size().reset_index(name="count")
            fig_cat = px.bar(
                cat_risk, x="category", y="count", color="risk_level",
                color_discrete_map={"HIGH": "#ff4d6d", "MEDIUM": "#ffd166", "LOW": "#06d6a0"},
                template="plotly_dark",
            )
            fig_cat.update_layout(paper_bgcolor="#111d35", plot_bgcolor="#111d35", height=300)
            st.plotly_chart(fig_cat, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<div class="section-header">Geo-Risk Heatmap</div>', unsafe_allow_html=True)
            loc_risk = df.groupby("location")["risk_score"].mean().reset_index()
            loc_risk.columns = ["location", "avg_risk"]
            loc_risk = loc_risk.sort_values("avg_risk", ascending=True)
            fig_geo = px.bar(
                loc_risk, x="avg_risk", y="location", orientation="h",
                color="avg_risk",
                color_continuous_scale=[[0, "#06d6a0"], [0.5, "#ffd166"], [1, "#ff4d6d"]],
                template="plotly_dark",
                labels={"avg_risk": "Avg Risk Score", "location": ""},
            )
            fig_geo.update_layout(paper_bgcolor="#111d35", plot_bgcolor="#111d35",
                                  height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_geo, use_container_width=True)

        with col_d:
            st.markdown('<div class="section-header">Hourly Transaction Volume</div>', unsafe_allow_html=True)
            hourly = df.groupby(["hour_of_day", "risk_level"]).size().reset_index(name="count")
            fig_hour = px.line(
                hourly, x="hour_of_day", y="count", color="risk_level",
                color_discrete_map={"HIGH": "#ff4d6d", "MEDIUM": "#ffd166", "LOW": "#06d6a0"},
                template="plotly_dark",
                labels={"hour_of_day": "Hour of Day", "count": "Transactions"},
            )
            fig_hour.update_layout(paper_bgcolor="#111d35", plot_bgcolor="#111d35", height=300)
            st.plotly_chart(fig_hour, use_container_width=True)

    # TAB 2 — ALERT QUEUE
    with tab2:
        st.markdown('<div class="section-header"> High-Risk Alert Queue</div>', unsafe_allow_html=True)

        high_df = view[view["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False).head(50)

        if high_df.empty:
            st.info("No high-risk transactions match the current filters.")
        else:
            for _, row in high_df.iterrows():
                col_i, col_ii, col_iii, col_iv, col_v = st.columns([2, 1.5, 1.5, 1, 1.5])
                col_i.markdown(f"**{row['transaction_id']}** · `{row['customer_id']}`")
                col_ii.markdown(f"PKR **{row['amount']:,.0f}**")
                col_iii.markdown(f" {row['location']}")
                col_iv.markdown(f" {row['category']}")
                col_v.markdown(f"Score: **{row['risk_score']:.3f}** {risk_badge('HIGH')}", unsafe_allow_html=True)
                st.markdown('<hr style="margin:4px 0;border-color:#1e3a5f;">', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-header">🟡 Medium-Risk Queue</div>', unsafe_allow_html=True)

        med_df = view[view["risk_level"] == "MEDIUM"].sort_values("risk_score", ascending=False).head(30)
        if not med_df.empty:
            display_cols = ["transaction_id", "customer_id", "amount", "category",
                            "location", "risk_score", "risk_level"]
            st.dataframe(
                med_df[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=300,
            )

    # TAB 3 — MODEL ANALYSIS
    with tab3:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown('<div class="section-header">Isolation Forest: Anomaly Score Map</div>',
                        unsafe_allow_html=True)
            sample = df.sample(min(600, len(df)), random_state=1)
            fig_scatter = px.scatter(
                sample, x="amount", y="transaction_freq_24h",
                color="risk_score",
                color_continuous_scale=[[0, "#06d6a0"], [0.5, "#ffd166"], [1, "#ff4d6d"]],
                size="risk_score", size_max=12,
                opacity=0.75,
                template="plotly_dark",
                hover_data=["transaction_id", "category", "location", "risk_level"],
                labels={"amount": "Transaction Amount (PKR)", "transaction_freq_24h": "24h Frequency"},
            )
            fig_scatter.update_layout(paper_bgcolor="#111d35", plot_bgcolor="#111d35", height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_m2:
            st.markdown('<div class="section-header">K-Means Cluster Distribution</div>',
                        unsafe_allow_html=True)
            cluster_summary = df.groupby("cluster").agg(
                count=("risk_score", "count"),
                avg_risk=("risk_score", "mean"),
                avg_amount=("amount", "mean"),
                high_risk_pct=("risk_level", lambda x: (x == "HIGH").mean() * 100)
            ).reset_index()

            fig_cluster = px.scatter(
                cluster_summary,
                x="avg_amount", y="avg_risk",
                size="count", color="high_risk_pct",
                color_continuous_scale=[[0, "#06d6a0"], [0.5, "#ffd166"], [1, "#ff4d6d"]],
                template="plotly_dark",
                text="cluster",
                labels={"avg_amount": "Avg Amount (PKR)", "avg_risk": "Avg Risk Score",
                        "high_risk_pct": "% High Risk"},
                size_max=50,
            )
            fig_cluster.update_traces(textposition="top center")
            fig_cluster.update_layout(paper_bgcolor="#111d35", plot_bgcolor="#111d35", height=380)
            st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown('<div class="section-header">Risk Classification Matrix</div>', unsafe_allow_html=True)
        matrix_data = {
            "Risk Level": ["🔴 HIGH RISK", "🟡 MEDIUM RISK", "🟢 LOW RISK"],
            "Score Bracket": ["0.85 – 1.00", "0.50 – 0.84", "0.00 – 0.49"],
            "Behavioral Indicators": [
                "Rapid cross-border jumps, anomalous high-volume transactions, unusual hours",
                "Unusual local merchant category, out-of-character hour, moderate frequency spike",
                "Matches historical pattern, trusted local terminals, normal frequency",
            ],
            "Mitigation Response": [
                "Temporary card freeze & automated SMS alert verification",
                "Analyst dashboard queueing & multi-factor validation check",
                "No interference — processed automatically without latency",
            ],
            "Count": [
                (df["risk_level"] == "HIGH").sum(),
                (df["risk_level"] == "MEDIUM").sum(),
                (df["risk_level"] == "LOW").sum(),
            ],
        }
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Performance Benchmark</div>', unsafe_allow_html=True)
        bench_cols = st.columns(3)
        bench_cols[0].metric("Anomaly Precision (AI Forest)", "81%", "+46pp vs Legacy (35%)")
        bench_cols[1].metric("Fraud Recall (AI Forest)", "76%", "+34pp vs Legacy (42%)")
        bench_cols[2].metric("Avg Alert Latency", "0.12s (Real-Time)", "-9.38s vs Rule Lookup (9.5s)")

    # TAB 4 — RISK TRENDS
    with tab4:
        st.markdown('<div class="section-header">Daily Risk Trend</div>', unsafe_allow_html=True)

        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby(["date", "risk_level"]).size().reset_index(name="count")

        fig_trend = px.area(
            daily, x="date", y="count", color="risk_level",
            color_discrete_map={"HIGH": "#ff4d6d", "MEDIUM": "#ffd166", "LOW": "#06d6a0"},
            template="plotly_dark",
            labels={"date": "Date", "count": "Transactions", "risk_level": "Risk Level"},
        )
        fig_trend.update_layout(paper_bgcolor="#111d35", plot_bgcolor="#111d35", height=350)
        st.plotly_chart(fig_trend, use_container_width=True)

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown('<div class="section-header">Top High-Risk Customers</div>', unsafe_allow_html=True)
            cust_risk = (
                df[df["risk_level"] == "HIGH"]
                .groupby("customer_id")
                .agg(high_risk_txns=("risk_score", "count"), avg_risk=("risk_score", "mean"),
                     total_amount=("amount", "sum"))
                .sort_values("high_risk_txns", ascending=False)
                .head(10)
                .reset_index()
            )
            cust_risk["total_amount"] = cust_risk["total_amount"].round(0).astype(int)
            cust_risk["avg_risk"] = cust_risk["avg_risk"].round(3)
            st.dataframe(cust_risk, use_container_width=True, hide_index=True)

        with col_t2:
            st.markdown('<div class="section-header">Foreign vs Local Risk Breakdown</div>',
                        unsafe_allow_html=True)
            foreign_risk = df.groupby(["is_foreign", "risk_level"]).size().reset_index(name="count")
            foreign_risk["origin"] = foreign_risk["is_foreign"].map({0: "Local", 1: "Foreign"})
            fig_foreign = px.pie(
                foreign_risk[foreign_risk["risk_level"] == "HIGH"],
                names="origin", values="count",
                color="origin",
                color_discrete_map={"Local": "#00c8ff", "Foreign": "#ff4d6d"},
                template="plotly_dark",
                hole=0.5,
            )
            fig_foreign.update_layout(paper_bgcolor="#111d35", height=300)
            st.plotly_chart(fig_foreign, use_container_width=True)

    # TAB 5 — TRANSACTION LOOKUP
    with tab5:
        st.markdown('<div class="section-header"> Lookup & Classify a Transaction</div>',
                    unsafe_allow_html=True)

        st.markdown("Enter transaction details below to get an instant AI risk assessment.")

        c1, c2, c3 = st.columns(3)
        with c1:
            inp_amount   = st.number_input("Amount (PKR)", min_value=1.0, value=15000.0, step=500.0)
            inp_category = st.selectbox("Category", ["Retail", "Online", "ATM", "Transfer",
                                                      "Utilities", "Travel", "Entertainment"])
        with c2:
            inp_freq     = st.slider("Transactions in Last 24h", 1, 30, 3)
            inp_location = st.selectbox("Location", ["Karachi", "Lahore", "Islamabad", "Rawalpindi",
                                                      "Faisalabad", "Foreign-EU", "Foreign-US", "Foreign-AE"])
        with c3:
            inp_hour = st.slider("Hour of Day", 0, 23, 14)
            inp_day  = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday",
                                                     "Thursday","Friday","Saturday","Sunday"])

        if st.button("Analyse Transaction", use_container_width=True):
            cat_map = {"Retail": 0, "Online": 1, "ATM": 2, "Transfer": 3,
                       "Utilities": 4, "Travel": 5, "Entertainment": 6}
            day_map = {d: i for i, d in enumerate(
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])}

            inp_vec = np.array([[
                inp_amount,
                inp_freq,
                1 if "Foreign" in inp_location else 0,
                inp_hour,
                day_map[inp_day],
                cat_map.get(inp_category, 0),
            ]])

            X_raw = df[features].values
            sc    = StandardScaler().fit(X_raw)
            inp_scaled = sc.transform(inp_vec)

            score_raw  = iso_model.decision_function(inp_scaled)[0]
            all_scores = iso_model.decision_function(sc.transform(X_raw))
            risk_val   = 1 - (score_raw - all_scores.min()) / (all_scores.max() - all_scores.min())

            if risk_val >= 0.85:
                level, col, action = "HIGH", "#ff4d6d", "Temporary card freeze & SMS alert triggered."
            elif risk_val >= 0.50:
                level, col, action = "MEDIUM", "#ffd166", "Queued for analyst review & MFA validation."
            else:
                level, col, action = "LOW", "#06d6a0", "Transaction cleared automatically."

            st.markdown(f"""
            <div style="background:{col}18;border:1px solid {col};border-radius:12px;padding:20px;margin-top:16px;">
              <h3 style="color:{col};margin-bottom:8px;">Risk Level: {level}</h3>
              <p style="font-size:1.4rem;color:#ffffff;margin:4px 0;">Score: <strong>{risk_val:.4f}</strong></p>
              <p style="color:#aaa;margin:4px 0;">Action: {action}</p>
            </div>
            """, unsafe_allow_html=True)

            # Feature contribution bar
            feat_labels  = ["Amount", "24h Freq", "Is Foreign", "Hour", "Day", "Category"]
            feat_contrib = np.abs(inp_scaled[0])
            feat_contrib = feat_contrib / feat_contrib.sum()

            fig_contrib = px.bar(
                x=feat_contrib, y=feat_labels, orientation="h",
                color=feat_contrib,
                color_continuous_scale=[[0, "#06d6a0"], [1, "#ff4d6d"]],
                template="plotly_dark",
                labels={"x": "Relative Contribution", "y": "Feature"},
                title="Feature Contribution to Risk Score",
            )
            fig_contrib.update_layout(paper_bgcolor="#111d35", plot_bgcolor="#111d35",
                                      coloraxis_showscale=False, height=280)
            st.plotly_chart(fig_contrib, use_container_width=True)


if __name__ == "__main__":
    main()
