import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(layout="wide")
st.title("🧠 Interactive Clustering Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Raw Data Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col + "_enc"] = label_encoder.fit_transform(df[col])

    # Feature selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("🔧 Select features for clustering:", numeric_cols, default=numeric_cols[:2])

    if len(selected_features) >= 2:
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[selected_features])

        col1, col2 = st.columns(2)

        # --- KMeans ---
        with col1:
            st.subheader("📌 K-Means Clustering")
            k = st.slider("Number of Clusters (KMeans)", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df["KMeans_Cluster"] = kmeans.fit_predict(features_scaled)

            st.write("Cluster counts:")
            st.dataframe(df["KMeans_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

            fig1, ax1 = plt.subplots()
            sns.scatterplot(x=features_scaled[:, 0], y=features_scaled[:, 1], hue=df["KMeans_Cluster"], palette="viridis", ax=ax1)
            ax1.set_title("K-Means Clustering")
            ax1.set_xlabel(selected_features[0])
            ax1.set_ylabel(selected_features[1])
            st.pyplot(fig1)

        # --- DBSCAN ---
        with col2:
            st.subheader("🌌 DBSCAN Clustering")
            eps = st.slider("EPS (Neighborhood radius)", 0.1, 5.0, 1.0)
            min_samples = st.slider("Min Samples", 1, 10, 2)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            df["DBSCAN_Cluster"] = dbscan.fit_predict(features_scaled)

            st.write("Cluster counts (DBSCAN):")
            st.dataframe(df["DBSCAN_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=features_scaled[:, 0], y=features_scaled[:, 1], hue=df["DBSCAN_Cluster"], palette="plasma", ax=ax2)
            ax2.set_title("DBSCAN Clustering")
            ax2.set_xlabel(selected_features[0])
            ax2.set_ylabel(selected_features[1])
            st.pyplot(fig2)

        # --- Hierarchical ---
        st.subheader("🧬 Hierarchical Clustering (Dendrogram)")
        linkage_matrix = linkage(features_scaled, method="ward")

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        dendrogram(linkage_matrix, ax=ax3)
        ax3.set_title("Dendrogram (Ward linkage)")
        ax3.set_xlabel("Data Points")
        ax3.set_ylabel("Distance")
        st.pyplot(fig3)

        # Optional Agglomerative clustering labels
        n_hier_clusters = st.slider("Number of Clusters (Agglomerative)", 2, 10, 3)
        hier = AgglomerativeClustering(n_clusters=n_hier_clusters)
        df["Hierarchical_Cluster"] = hier.fit_predict(features_scaled)

        st.write("Cluster counts (Hierarchical):")
        st.dataframe(df["Hierarchical_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

        # Show labeled data
        st.subheader("📊 Clustered Data Preview")
        st.dataframe(df.head())

    else:
        st.warning("Please select at least two numeric features for clustering.")
else:
    st.info("📥 Upload a CSV file to get started.")
