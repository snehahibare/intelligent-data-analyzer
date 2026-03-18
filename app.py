import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Intelligent Data Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Intelligent Data Analyzer")
st.caption("Upload any CSV → instant AI-powered insights!")
st.divider()

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    target   = st.text_input("Target Column (optional)", value="")
    run_ml   = st.checkbox("Run AutoML", value=True)
    run_anom = st.checkbox("Run Anomaly Detection", value=True)

if not uploaded:
    st.info("👆 Upload a CSV file from the sidebar to get started!")
    st.stop()

df = pd.read_csv(uploaded)
target = target.strip() or None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows",    f"{df.shape[0]:,}")
col2.metric("Columns", df.shape[1])
col3.metric("Nulls",   f"{df.isnull().sum().sum():,}")
col4.metric("Dupes",   f"{df.duplicated().sum():,}")

@st.cache_data
def quick_clean(df, target):
    d = df.copy().drop_duplicates()
    for col in d.select_dtypes(include=np.number).columns:
        if col != target:
            d[col].fillna(d[col].median(), inplace=True)
    for col in d.select_dtypes(include="object").columns:
        if col != target:
            d[col].fillna(d[col].mode()[0], inplace=True)
            d[col] = LabelEncoder().fit_transform(d[col].astype(str))
    return d

df_c = quick_clean(df, target)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "🔥 Correlations", "🚨 Anomalies", "🤖 AutoML", "🧠 Features"
])

with tab1:
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    num_cols = df_c.select_dtypes(include=np.number).columns.tolist()
    feat = st.selectbox("Select feature to plot",
                        [c for c in num_cols if c != target])
    fig = px.histogram(df_c, x=feat, nbins=40, template="plotly_dark",
                       color_discrete_sequence=["#7c3aed"])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Correlation Heatmap")
    num = df_c.select_dtypes(include=np.number)
    corr = num.iloc[:, :15].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}"
    ))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if run_anom:
        st.subheader("Anomaly Detection")
        X_a = df_c.select_dtypes(include=np.number)
        if target and target in df_c.columns:
            X_a = X_a.drop(columns=[target])
        X_s = StandardScaler().fit_transform(X_a.fillna(0))
        iso = IsolationForest(contamination=0.05, random_state=42)
        labels = iso.fit_predict(X_s)
        df_c["anomaly"] = (labels == -1).astype(int)
        n_anom = df_c["anomaly"].sum()
        st.metric("Anomalies Found", f"{n_anom:,} ({n_anom/len(df_c)*100:.1f}%)")
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(X_s)
        plot_df = pd.DataFrame({
            "PC1": coords[:,0], "PC2": coords[:,1],
            "Status": df_c["anomaly"].map({0:"Normal", 1:"Anomaly"})
        })
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Status",
                         color_discrete_map={"Normal":"#2563eb","Anomaly":"#dc2626"},
                         template="plotly_dark", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    if run_ml and target and target in df_c.columns:
        st.subheader("AutoML Leaderboard")
        X = df_c.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)
        y = df_c[target]
        t = "classification" if y.nunique() <= 20 else "regression"
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Random Forest": RandomForestClassifier(100, random_state=42) if t=="classification" else RandomForestRegressor(100, random_state=42),
            "XGBoost": xgb.XGBClassifier(eval_metric="logloss", verbosity=0, random_state=42) if t=="classification" else xgb.XGBRegressor(verbosity=0, random_state=42),
            "LightGBM": lgb.LGBMClassifier(verbose=-1, random_state=42) if t=="classification" else lgb.LGBMRegressor(verbose=-1, random_state=42),
        }
        rows = []
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            p = m.predict(X_te)
            score = accuracy_score(y_te, p) if t=="classification" else r2_score(y_te, p)
            rows.append({"Model": name, "Score": round(score, 4),
                         "Metric": "Accuracy" if t=="classification" else "R²"})
        lb = pd.DataFrame(rows).sort_values("Score", ascending=False)
        st.dataframe(lb, use_container_width=True)
        fig = px.bar(lb, x="Model", y="Score", color="Score",
                     color_continuous_scale="Viridis", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Target column set karo sidebar mein!")

with tab5:
    if target and target in df_c.columns:
        st.subheader("AI Feature Importance")
        X = df_c.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)
        y = df_c[target]
        t = "classification" if y.nunique() <= 20 else "regression"
        m = RandomForestClassifier(50, random_state=42) if t=="classification" else RandomForestRegressor(50, random_state=42)
        m.fit(X, y)
        fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": m.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.markdown("### 🏆 Top 3 Features")
        for i, (_, row) in enumerate(fi.head(3).iterrows(), 1):
            st.success(f"#{i} `{row['Feature']}` — {row['Importance']*100:.2f}%")
        fig = px.bar(fi.head(15), x="Importance", y="Feature",
                     orientation="h", color="Importance",
                     color_continuous_scale="Viridis", template="plotly_dark")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Target column set karo sidebar mein!")
