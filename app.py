import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer PCA", layout="wide")
st.title("Customer Dataset Dimensionality Reduction using PCA")
st.write("Explore how scaling changes PCA on customer spending data.")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    return pd.read_csv(url)

try:
    df = load_data()
except Exception as e:
    st.error("Could not download the UCI dataset. Check your internet connection.")
    st.exception(e)
    st.stop()

features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X = df[features]
mode = st.radio("Choose PCA method", ["Without scaling", "With StandardScaler"], horizontal=True)

if mode == "With StandardScaler":
    X_input = StandardScaler().fit_transform(X)
else:
    X_input = X

pca = PCA()
transformed = pca.fit_transform(X_input)
variance = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(features))],
    "Explained variance": pca.explained_variance_ratio_,
    "Cumulative variance": pca.explained_variance_ratio_.cumsum()
})
st.subheader("Dataset")
st.write(f"{df.shape[0]} customers and {len(features)} spending features")
st.dataframe(df.head())
st.subheader("Explained variance")
st.dataframe(variance, use_container_width=True)
fig, ax = plt.subplots()
ax.bar(variance["Component"], variance["Explained variance"])
ax.set_ylabel("Explained variance ratio")
ax.set_title(mode)
st.pyplot(fig)
fig2, ax2 = plt.subplots()
ax2.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6)
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_title("Customers in first two principal components")
st.pyplot(fig2)
st.info(f"First two components retain {pca.explained_variance_ratio_[:2].sum()*100:.2f}% of variance.")
