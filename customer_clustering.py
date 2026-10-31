from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd

# Load customer RFM data
rfm = pd.read_csv("outputs/customer_rfm.csv")

# Select numerical RFM columns
clustering_data = rfm[
    ["Recency", "Frequency", "Monetary"]
].copy()

# Display the first five rows
print("RFM DATA FOR CLUSTERING:")
print(clustering_data.head())

# Check dataset information
print("\nDATASET INFORMATION:")
print(clustering_data.info())

# Check missing values
print("\nMISSING VALUES:")
print(clustering_data.isnull().sum())

# Scale RFM features
scaler = StandardScaler()

scaled_data = scaler.fit_transform(clustering_data)

# Convert scaled data into a DataFrame
scaled_rfm = pd.DataFrame(
    scaled_data,
    columns=["Recency", "Frequency", "Monetary"]
)

print("\nSCALED RFM DATA:")
print(scaled_rfm.head())

print("\nSCALED DATA SUMMARY:")
print(scaled_rfm.describe().round(2))

# Elbow Method to find optimal number of clusters

inertia_values = []
silhouette_values = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(scaled_rfm)

    inertia_values.append(kmeans.inertia_)

    score = silhouette_score(scaled_rfm, cluster_labels)
    silhouette_values.append(score)


print("\nK-MEANS EVALUATION RESULTS:")

for k, inertia, silhouette in zip(
    range(2, 9),
    inertia_values,
    silhouette_values
):
    print(
        f"K={k}, "
        f"Inertia={inertia:.2f}, "
        f"Silhouette Score={silhouette:.4f}"
    )

# Train final K-means model with 4 clusters

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(scaled_rfm)

# PCA ANALYSIS

pca = PCA(n_components=2)

pca_result = pca.fit_transform(scaled_rfm)

print("\nPCA EXPLAINED VARIANCE RATIO:")
print(pca.explained_variance_ratio_)

print(
    "Total explained variance:",
    pca.explained_variance_ratio_.sum()
)

# Display cluster distribution
print("\nCLUSTER DISTRIBUTION:")
print(rfm["Cluster"].value_counts().sort_index())

# Display cluster averages
cluster_summary = rfm.groupby("Cluster").agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"),
    Customer_Count=("customer_id", "count")
).round(2)

print("\nCLUSTER SUMMARY:")
print(cluster_summary)


# Identify Premium cluster using highest average monetary value
premium_cluster = cluster_summary["Monetary"].idxmax()

# Identify other clusters
remaining_clusters = cluster_summary.drop(index=premium_cluster)

# Cluster with highest frequency among remaining clusters
frequent_cluster = remaining_clusters["Frequency"].idxmax()

remaining_clusters = remaining_clusters.drop(index=frequent_cluster)

# Cluster with highest recency among remaining clusters
inactive_cluster = remaining_clusters["Recency"].idxmax()

# Assign business names
segment_names = {
    premium_cluster: "Premium Customers",
    frequent_cluster: "Frequent & Recent Customers",
    inactive_cluster: "Inactive Customers"
}

# Assign a name to any remaining cluster
for cluster_id in cluster_summary.index:
    if cluster_id not in segment_names:
        segment_names[cluster_id] = "Occasional Customers"

# Add segment names and Premium flag
rfm["segment_name"] = rfm["Cluster"].map(segment_names)
rfm["is_premium"] = rfm["Cluster"] == premium_cluster

# Create the required customer segments file
customer_segments = rfm[
    ["customer_id", "Recency", "Frequency", "Monetary",
     "Cluster", "segment_name", "is_premium"]
].copy()

# Rename columns as required by the assignment
customer_segments.columns = [
    "customer_id", "recency", "frequency", "monetary",
    "segment_id", "segment_name", "is_premium"
]

# Save all output files
rfm.to_csv("outputs/customer_clusters.csv", index=False)
cluster_summary.to_csv("outputs/cluster_summary.csv")
customer_segments.to_csv("outputs/customer_segments.csv", index=False)

print("\nPREMIUM CLUSTER:", premium_cluster)

print("\nSEGMENT DISTRIBUTION:")
print(customer_segments["segment_name"].value_counts())

print("\nPREMIUM CUSTOMER COUNT:")
print(customer_segments["is_premium"].sum())

print("\nCustomer clustering results saved successfully!")
