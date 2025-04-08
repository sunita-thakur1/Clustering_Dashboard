import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Clustering Dashboard", layout="wide")

st.title("🔍 Interactive Clustering Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    # Optional label encoding
    st.sidebar.header("Preprocessing Options")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col + '_encoded'] = LabelEncoder().fit_transform(df[col])
    
    # Feature selection
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Select features for clustering", 
        options=all_numeric, 
        default=all_numeric[:2]
    )

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
    else:
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[selected_features])

        # Sidebar clustering settings
        st.sidebar.header("Clustering Parameters")
        k = st.sidebar.slider("Number of clusters (for KMeans & Hierarchical)", 2, 10, 3)
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 1.0)
        min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 10, 2)

        # KMeans
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        df["KMeans_Cluster"] = kmeans.fit_predict(features_scaled)

        # DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df["DBSCAN_Cluster"] = dbscan.fit_predict(features_scaled)

        # Agglomerative
        agglom = AgglomerativeClustering(n_clusters=k)
        df["Hierarchical_Cluster"] = agglom.fit_predict(features_scaled)

        # Tabs for visualization
        tab1, tab2, tab3 = st.tabs(["📊 KMeans & DBSCAN", "🌿 Hierarchical Dendrogram", "📈 Cluster Scatter"])

        with tab1:
            st.write("### KMeans Clustering")
            fig1, ax1 = plt.subplots()
            sns.scatterplot(
                x=df[selected_features[0]], 
                y=df[selected_features[1]], 
                hue=df["KMeans_Cluster"], 
                palette="viridis", ax=ax1
            )
            st.pyplot(fig1)

            st.write("### DBSCAN Clustering")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(
                x=df[selected_features[0]], 
                y=df[selected_features[1]], 
                hue=df["DBSCAN_Cluster"], 
                palette="plasma", ax=ax2
            )
            st.pyplot(fig2)

        with tab2:
            st.write("### Hierarchical Dendrogram")
            linkage_matrix = linkage(features_scaled, method='ward')
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            dendrogram(linkage_matrix, ax=ax3)
            st.pyplot(fig3)

        with tab3:
            st.write("### Combined Cluster Labels")
            st.dataframe(df[[*selected_features, "KMeans_Cluster", "DBSCAN_Cluster", "Hierarchical_Cluster"]])

else:
    st.info("👈 Upload a CSV file to get started.")

