import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Customer PCA | Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'DM Sans', sans-serif;}
.stApp {background: #f4f7fb;}
[data-testid="stHeader"] {background: rgba(244,247,251,0.9);}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
.hero {
  background: linear-gradient(115deg,#101d3a 0%,#173b68 65%,#24618b 100%);
  padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 22px;
}
.hero h1 {font-family:'Manrope',sans-serif; font-size:30px; font-weight:800; margin:0 0 8px 0; color:white;}
.hero p {color:#d9e8f7; margin:0; font-size:15px;}
.eyebrow {color:#8dc8f4; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:9px;}
.metric {
  background:white; border:1px solid #e5ebf3; border-radius:14px;
  padding:17px 19px; min-height:112px; box-shadow:0 3px 12px rgba(20,45,80,.04);
}
.metric-label {font-size:12px; color:#66758b; font-weight:600; margin-bottom:9px;}
.metric-value {font-family:'Manrope',sans-serif; font-size:27px; color:#142847; font-weight:800;}
.metric-note {font-size:11px; color:#8794a6; margin-top:4px;}
.section-title {font-family:'Manrope',sans-serif; font-size:21px; font-weight:800; color:#172b4d; margin:10px 0 4px;}
.section-sub {color:#748198; font-size:13px; margin-bottom:15px;}
div[data-testid="stSidebar"] {background:#101d36;}
div[data-testid="stSidebar"] * {color:#edf4ff;}
div[data-testid="stSidebar"] [data-testid="stRadio"] label {color:#edf4ff;}
div[data-testid="stDataFrame"] {border:1px solid #e5ebf3; border-radius:10px;}
div.stButton > button {border-radius:9px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    return pd.read_csv(url)

FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

try:
    df = load_data()
except Exception as exc:
    st.error("The UCI Wholesale Customers dataset could not be downloaded. Please check the app's internet connection.")
    st.exception(exc)
    st.stop()

X = df[FEATURES]

def run_pca(scale):
    values = StandardScaler().fit_transform(X) if scale else X.to_numpy()
    model = PCA()
    transformed = model.fit_transform(values)
    variance = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(FEATURES))],
        "Explained Variance (%)": model.explained_variance_ratio_ * 100,
        "Cumulative Variance (%)": np.cumsum(model.explained_variance_ratio_) * 100
    })
    return model, transformed, variance

raw_model, raw_transformed, raw_var = run_pca(False)
scaled_model, scaled_transformed, scaled_var = run_pca(True)

st.sidebar.markdown("## 📊 Customer PCA")
st.sidebar.caption("DIMENSIONALITY REDUCTION PROJECT")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "PCA Analysis", "Scaling Comparison", "Dataset"],
    label_visibility="visible"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Project workflow**")
st.sidebar.markdown("1. Load customer data\n2. Select six spending features\n3. Apply PCA\n4. Compare scaled vs. unscaled data")
st.sidebar.markdown("---")
st.sidebar.caption("Algorithm: Principal Component Analysis (PCA)")

st.markdown("""
<div class="hero">
  <div class="eyebrow">MACHINE LEARNING • UNSUPERVISED LEARNING</div>
  <h1>Customer Dataset Analytics</h1>
  <p>Dimensionality reduction using Principal Component Analysis (PCA)</p>
</div>
""", unsafe_allow_html=True)

def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

if page == "Overview":
    st.markdown('<div class="section-title">Project at a glance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A quick summary of the dataset and dimensionality-reduction results.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("CUSTOMER RECORDS", f"{df.shape[0]:,}", "Wholesale customer entries")
    with c2: metric_card("INPUT FEATURES", str(len(FEATURES)), "Annual spending categories")
    with c3: metric_card("90% VARIANCE • RAW", f"{int(np.argmax(raw_var['Cumulative Variance (%)'].values >= 90)+1)} PCs", "Components required")
    with c4: metric_card("90% VARIANCE • SCALED", f"{int(np.argmax(scaled_var['Cumulative Variance (%)'].values >= 90)+1)} PCs", "Components required")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section-title">What does this project do?</div>', unsafe_allow_html=True)
        st.write("This project uses PCA to reduce six customer spending variables into a smaller set of principal components while retaining as much information (variance) as possible.")
        st.markdown("**PCA is an unsupervised dimensionality-reduction technique.** It does not predict a target or assign customer labels.")
        st.markdown('<div class="section-title">Dataset features</div>', unsafe_allow_html=True)
        st.write(", ".join(FEATURES))
    with right:
        st.markdown('<div class="section-title">Key finding</div>', unsafe_allow_html=True)
        raw3 = raw_var["Cumulative Variance (%)"].iloc[2]
        scaled3 = scaled_var["Cumulative Variance (%)"].iloc[2]
        st.info(f"Without scaling, the first 3 components retain **{raw3:.2f}%** variance. With StandardScaler, the first 3 retain **{scaled3:.2f}%**. Scaling changes how feature magnitudes influence PCA.")
        st.markdown("**Use the sidebar** to explore the PCA results, compare both methods, and inspect the dataset.")

elif page == "PCA Analysis":
    st.markdown('<div class="section-title">PCA analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Choose whether to standardize the six spending features before applying PCA.</div>', unsafe_allow_html=True)
    choice = st.radio("PCA input", ["Without scaling", "With StandardScaler"], horizontal=True)
    model, transformed, variance = (raw_model, raw_transformed, raw_var) if choice == "Without scaling" else (scaled_model, scaled_transformed, scaled_var)
    k90 = int(np.argmax(variance["Cumulative Variance (%)"].values >= 90) + 1)
    a, b, c = st.columns(3)
    with a: metric_card("VARIANCE IN PC1", f"{variance['Explained Variance (%)'].iloc[0]:.2f}%", "Explained by first component")
    with b: metric_card("FIRST 2 COMPONENTS", f"{variance['Cumulative Variance (%)'].iloc[1]:.2f}%", "Cumulative variance retained")
    with c: metric_card("COMPONENTS FOR 90%", str(k90), "Minimum number of PCs")
    l, r = st.columns(2)
    with l:
        st.markdown("**Explained variance by component**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(variance["Principal Component"], variance["Explained Variance (%)"])
        ax.set_ylabel("Variance (%)")
        ax.set_xlabel("Principal component")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with r:
        st.markdown("**Cumulative explained variance**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(variance["Principal Component"], variance["Cumulative Variance (%)"], marker="o")
        ax.axhline(90, linestyle="--", linewidth=1, label="90% target")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Cumulative variance (%)")
        ax.set_xlabel("Principal component")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.markdown("**Customers projected onto PC1 and PC2**")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(transformed[:, 0], transformed[:, 1], alpha=.55, s=22)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2D projection of customer records")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("**Explained variance table**")
    st.dataframe(variance.round(2), hide_index=True, width="stretch")

elif page == "Scaling Comparison":
    st.markdown('<div class="section-title">Scaling comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Compare PCA results before and after StandardScaler using the same six features.</div>', unsafe_allow_html=True)
    raw90 = int(np.argmax(raw_var["Cumulative Variance (%)"].values >= 90) + 1)
    scaled90 = int(np.argmax(scaled_var["Cumulative Variance (%)"].values >= 90) + 1)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("RAW • FIRST 3 PCs", f"{raw_var['Cumulative Variance (%)'].iloc[2]:.2f}%", "Cumulative variance")
    with c2: metric_card("SCALED • FIRST 3 PCs", f"{scaled_var['Cumulative Variance (%)'].iloc[2]:.2f}%", "Cumulative variance")
    with c3: metric_card("PCs TO REACH 90%", f"{raw90} vs {scaled90}", "Without scaling vs. scaled")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(FEATURES))
    width = .36
    ax.bar(x - width/2, raw_var["Explained Variance (%)"], width, label="Without scaling")
    ax.bar(x + width/2, scaled_var["Explained Variance (%)"], width, label="With StandardScaler")
    ax.set_xticks(x)
    ax.set_xticklabels(raw_var["Principal Component"])
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Explained variance comparison")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    comparison = pd.DataFrame({
        "Principal Component": raw_var["Principal Component"],
        "Without scaling (%)": raw_var["Explained Variance (%)"],
        "With StandardScaler (%)": scaled_var["Explained Variance (%)"],
        "Raw cumulative (%)": raw_var["Cumulative Variance (%)"],
        "Scaled cumulative (%)": scaled_var["Cumulative Variance (%)"]
    })
    st.dataframe(comparison.round(2), hide_index=True, width="stretch")
    st.success("Interpretation: StandardScaler puts features on a comparable scale before PCA. This reduces the influence of variables simply because they have larger numerical variances.")

elif page == "Dataset":
    st.markdown('<div class="section-title">Dataset explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Wholesale Customers dataset from the UCI Machine Learning Repository.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("ROWS", f"{df.shape[0]:,}", "Customer records")
    with c2: metric_card("COLUMNS", str(df.shape[1]), "All dataset columns")
    with c3: metric_card("PCA FEATURES", str(len(FEATURES)), "Numerical spending variables")
    st.markdown("**Dataset preview**")
    st.dataframe(df.head(10), hide_index=True, width="stretch")
    st.markdown("**Summary statistics for PCA features**")
    st.dataframe(df[FEATURES].describe().round(2), width="stretch")
    st.caption("Channel and Region are retained in the original dataset for description, but are not used as PCA input features.")

st.markdown("---")
st.caption("Customer Dataset Dimensionality Reduction using PCA • Built with Python, Streamlit, Pandas and Scikit-learn")
