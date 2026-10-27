#Customer segmentation: clustering
#Which retention strategies are best for which customers?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import seaborn as sns

#Access data from Kaggle

df = pd.read_csv("sales_and_customer_insights.csv")

#Define clustering features

clustering_features = [
    "Purchase_Frequency",
    "Average_Order_Value",
    "Time_Between_Purchases",
    "Lifetime_Value"
]

X = df[clustering_features].copy()

print("\nClustering features:")
print(X.head())

#Standardize variables

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

#Determine number of clusters

inertia = []
silhouette_scores = []

k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

plt.figure(figsize=(8, 5))

plt.plot(k_values, inertia, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

#plt.show()

plt.figure(figsize=(8, 5))

plt.plot(k_values, silhouette_scores, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores")

#plt.show()

#Create final clusters

best_k = 8 #highest silhouette score

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())

#Profile customer segments

cluster_profile = df.groupby("Cluster")[clustering_features].mean()

print("\nCluster Profiles:")
print(cluster_profile)

#Determine which retention strategies dominant within clusters

strategy_by_cluster = pd.crosstab(
    df["Cluster"],
    df["Retention_Strategy"],
    normalize="index"
) * 100

print("\nRetention Strategy Distribution by Cluster:")
print(strategy_by_cluster.round(2))

#Visualizations

#Scatterplot

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": df["Cluster"].astype(str)
})

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    alpha=0.6,
    s=35
)

explained = pca.explained_variance_ratio_.sum() * 100

plt.title(
    f"Customer Segments Using PCA (K=8)\n"
    f"Two components explain {explained:.1f}% of variance"
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

#Retention strategy distribution

ax = strategy_by_cluster.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    width=0.8
)

ax.set_title("Retention Strategy Distribution by Customer Cluster")
ax.set_xlabel("Customer Cluster")
ax.set_ylabel("Percentage of Customers")
ax.set_ylim(0, 100)
ax.legend(
    title="Retention Strategy",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()