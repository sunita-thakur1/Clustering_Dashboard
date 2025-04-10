import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(layout="wide")
st.title("🧠 Clustering Dashboard")
# App Description
st.markdown("""
Welcome to the **Clustering Dashboard**! This app allows you to upload your dataset, apply different clustering algorithms, and visualize the results interactively.

You can:
- Upload your own dataset (CSV format).
- Select numeric features for clustering.
- Apply **Standard Scaling** to normalize the features.
- Use **PCA** to reduce dimensionality for easier visualization.
- Run **K-Means**, **DBSCAN**, and **Hierarchical Clustering** algorithms.
- Visualize clusters with scatter plots, dendrograms, and feature importance from PCA.

### Key Features:
- **Scaling** of features using StandardScaler.
- **PCA** for dimensionality reduction (2D visualization).
- **Interactive Clustering** with K-Means, DBSCAN, and Hierarchical Clustering.
- **Feature Contributions** displayed in PCA components to interpret the results.

Let's get started by uploading your dataset!
""")
2. Add Descriptions for Specific Sections
After key sections such as PCA, Clustering, or Feature Selection, you can provide more details about each step.

For instance, after the PCA application, you might want to describe what PCA is and why it is useful for dimensionality reduction.

python
Copy
Edit
# After PCA application section
st.subheader("🧬 PCA - Principal Component Analysis")

st.markdown("""
**PCA** (Principal Component Analysis) is a statistical technique used to reduce the dimensionality of data while preserving as much variance as possible. By reducing the number of dimensions, PCA makes it easier to visualize and interpret high-dimensional data.

In this app, you can view how each feature contributes to the first two principal components (PC1 and PC2), helping you understand the importance of different features in your data.

We display:
- **PCA loadings table**: Shows the contribution of each feature to the first two principal components.
- **PCA feature importance bar chart**: Visualizes the weight of each feature in determining the principal components.
""")
3. Add Descriptions for Clustering Techniques
For each clustering algorithm, you can add a description explaining how it works and what the results mean.

python
Copy
Edit
# After KMeans clustering section
st.subheader("📌 K-Means Clustering")

st.markdown("""
**K-Means** is one of the most popular clustering algorithms. It works by partitioning the data into a predefined number of clusters (K) based on the feature similarities.

### How K-Means Works:
1. Choose a number of clusters, K.
2. Randomly assign initial centroids for the clusters.
3. Assign each data point to the nearest centroid.
4. Recalculate the centroids based on the assigned data points.
5. Repeat the process until the centroids stop changing.

In the app, you can select the number of clusters and visualize the clustering results in a scatter plot with different colors for each cluster.
""")
4. Adding Descriptions for Cluster Summary
After showing the clustering results, you can add a description about how to interpret the output.

python
Copy
Edit
# After displaying cluster counts and data preview
st.subheader("📊 Cluster Summary")

st.markdown("""
After performing clustering, you can review the number of data points in each cluster. The cluster counts table provides insight into the distribution of data points across clusters.

By inspecting the scatter plot or dendrogram, you can explore how well the data has been segmented. The clustering results can reveal patterns and groupings within the dataset.
""")
5. General Tips and Information
If you'd like to include a general info section, instructions, or other tips, you can add it at the end or at the beginning of the app.

python
Copy
Edit
# General Instructions/Info
st.sidebar.markdown("""
### How to Use:
1. **Upload your CSV file** with numeric data.
2. **Select features** you want to use for clustering.
3. Choose whether to apply **PCA** for dimensionality reduction or use the original data.
4. **Adjust clustering parameters** for K-Means, DBSCAN, or Hierarchical Clustering.
5. Explore the clustering results with scatter plots and dendrograms.

If you have any questions or feedback, feel free to reach out!
""")
Complete Example with Descriptions Added:
Here’s your updated code with descriptions added in various sections:

python
Copy
Edit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(layout="wide")
st.title("🧠 Clustering Dashboard")

# App Description
st.markdown("""
Welcome to the **Clustering Dashboard**! This app allows you to upload your dataset, apply different clustering algorithms, and visualize the results interactively.

You can:
- Upload your own dataset (CSV format).
- Select numeric features for clustering.
- Apply **Standard Scaling** to normalize the features.
- Use **PCA** to reduce dimensionality for easier visualization.
- Run **K-Means**, **DBSCAN**, and **Hierarchical Clustering** algorithms.
- Visualize clusters with scatter plots, dendrograms, and feature importance from PCA.

### Key Features:
- **Scaling** of features using StandardScaler.
- **PCA** for dimensionality reduction (2D visualization).
- **Interactive Clustering** with K-Means, DBSCAN, and Hierarchical Clustering.
- **Feature Contributions** displayed in PCA components to interpret the results.
""")

# Upload dataset
uploaded_file = st.file_uploader("📥 Upload your CSV file", type=["csv"])
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
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[selected_features])

        # Apply PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        df["PCA1"] = pca_data[:, 0]
        df["PCA2"] = pca_data[:, 1]

        # Show PCA feature contributions
        st.subheader("🧬 PCA - Principal Component Analysis")

        st.markdown("""
        **PCA** (Principal Component Analysis) is a statistical technique used to reduce the dimensionality of data while preserving as much variance as possible. By reducing the number of dimensions, PCA makes it easier to visualize and interpret high-dimensional data.
        
        In this app, you can view how each feature contributes to the first two principal components (PC1 and PC2), helping you understand the importance of different features in your data.
        
        We display:
        - **PCA loadings table**: Shows the contribution of each feature to the first two principal components.
        - **PCA feature importance bar chart**: Visualizes the weight of each feature in determining the principal components.
        """)

        pca_components_df = pd.DataFrame(
            pca.components_.T,
            columns=["PC1", "PC2"],
            index=selected_features
        )

        st.write("### 🔍 PCA Loadings Table")
        st.dataframe(pca_components_df.style.format("{:.2f}"))

        st.write("### 📊 PCA Feature Importance Bar Chart")
        fig_pca, ax_pca = plt.subplots(figsize=(10, 5))
        pca_components_df.plot(kind="bar", ax=ax_pca)
        ax_pca.set_title("Feature Contributions to Principal Components")
        ax_pca.set_ylabel("Loading Weight")
        ax_pca.set_xlabel("Features")
        ax_pca.axhline(0, color='gray', linewidth=0.8)
        plt.xticks(rotation=45)
        st.pyplot(fig_pca)

        # PCA toggle
        use_pca = st.checkbox("🧪 Use PCA-transformed data for clustering and visualization", value=True)
        data_for_clustering = pca_data if use_pca else scaled_data

        col1, col2 = st.columns(2)

        # --- KMeans ---
        with col1:
            st.subheader("📌 K-Means Clustering")

            st.markdown("""
            **K-Means** is one of the most popular clustering algorithms. It works by partitioning the data into a predefined number of clusters (K) based on the feature similarities.
            
            ### How K-Means Works:
            1. Choose a number of clusters, K.
            2. Randomly assign initial centroids for the clusters.
            3. Assign each data point to the nearest centroid.
            4. Recalculate the centroids based on the assigned data points.
            5. Repeat the process until the centroids stop changing.
            
            In the app, you can select the number of clusters and visualize the clustering results in a scatter plot with different colors for each cluster.
            """)

            k = st.slider("Number of Clusters (KMeans)", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df["KMeans_Cluster"] = kmeans.fit_predict(data_for_clustering)

            st.write("Cluster counts (KMeans):")
            st.dataframe(df["KMeans_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

            fig1, ax1 = plt.subplots()
            sns.scatterplot(
                x=df["PCA1"] if use_pca else df[selected_features[0]],
                y=df["PCA2"] if use_pca else df[selected_features[











# Upload dataset
uploaded_file = st.file_uploader("📥 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Raw Data Preview")
    st.dataframe(df.head())
    st.write("### Summary Statistics")
    st.write(df.describe())

    # Encode categorical columns
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col + "_enc"] = label_encoder.fit_transform(df[col])

    # Feature selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("🔧 Select features for clustering:", numeric_cols, default=numeric_cols[:2])

    if len(selected_features) >= 2:
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[selected_features])

        # Apply PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        df["PCA1"] = pca_data[:, 0]
        df["PCA2"] = pca_data[:, 1]

        # Show PCA feature contributions
        st.subheader("🧬 PCA Feature Contributions")

        pca_components_df = pd.DataFrame(
            pca.components_.T,
            columns=["PC1", "PC2"],
            index=selected_features
        )

        st.write("### 🔍 PCA Loadings Table")
        st.dataframe(pca_components_df.style.format("{:.2f}"))

        st.write("### 📊 PCA Feature Importance Bar Chart")
        fig_pca, ax_pca = plt.subplots(figsize=(10, 5))
        pca_components_df.plot(kind="bar", ax=ax_pca)
        ax_pca.set_title("Feature Contributions to Principal Components")
        ax_pca.set_ylabel("Loading Weight")
        ax_pca.set_xlabel("Features")
        ax_pca.axhline(0, color='gray', linewidth=0.8)
        plt.xticks(rotation=45)
        st.pyplot(fig_pca)

        # PCA toggle
        use_pca = st.checkbox("🧪 Use PCA-transformed data for clustering and visualization", value=True)
        data_for_clustering = pca_data if use_pca else scaled_data

        col1, col2 = st.columns(2)

        # --- KMeans ---
        with col1:
            st.subheader("📌 K-Means Clustering")
            k = st.slider("Number of Clusters (KMeans)", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df["KMeans_Cluster"] = kmeans.fit_predict(data_for_clustering)

            st.write("Cluster counts (KMeans):")
            st.dataframe(df["KMeans_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

            fig1, ax1 = plt.subplots()
            sns.scatterplot(
                x=df["PCA1"] if use_pca else df[selected_features[0]],
                y=df["PCA2"] if use_pca else df[selected_features[1]],
                hue=df["KMeans_Cluster"],
                palette="viridis",
                ax=ax1
            )
            ax1.set_title("K-Means Clustering")
            st.pyplot(fig1)

        # --- DBSCAN ---
        with col2:
            st.subheader("🌌 DBSCAN Clustering")
            eps = st.slider("EPS (Neighborhood radius)", 0.1, 100.0, 10.0)
            min_samples = st.slider("Min Samples", 1, 10, 2)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            df["DBSCAN_Cluster"] = dbscan.fit_predict(data_for_clustering)

            st.write("Cluster counts (DBSCAN):")
            st.dataframe(df["DBSCAN_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

            fig2, ax2 = plt.subplots()
            sns.scatterplot(
                x=df["PCA1"] if use_pca else df[selected_features[0]],
                y=df["PCA2"] if use_pca else df[selected_features[1]],
                hue=df["DBSCAN_Cluster"],
                palette="plasma",
                ax=ax2
            )
            ax2.set_title("DBSCAN Clustering")
            st.pyplot(fig2)

        # --- Hierarchical Dendrogram ---
        st.subheader("🧬 Hierarchical Clustering (Dendrogram)")
        linkage_matrix = linkage(data_for_clustering, method="ward")

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        dendrogram(linkage_matrix, ax=ax3)
        ax3.set_title("Dendrogram (Ward linkage)")
        ax3.set_xlabel("Data Points")
        ax3.set_ylabel("Distance")
        st.pyplot(fig3)

        # Agglomerative clustering labels
        n_hier_clusters = st.slider("Number of Clusters (Agglomerative)", 2, 10, 3)
        hier = AgglomerativeClustering(n_clusters=n_hier_clusters)
        df["Hierarchical_Cluster"] = hier.fit_predict(data_for_clustering)

        st.write("Cluster counts (Hierarchical):")
        st.dataframe(df["Hierarchical_Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

        # Show labeled data
        st.subheader("📊 Clustered Data Preview")
        st.dataframe(df.head())
        st.subheader("📊 Clustered Data")
        st.write(df)

    else:
        st.warning("Please select at least two numeric features for clustering.")
else:
    st.info("📂 Upload a CSV file to get started.")
