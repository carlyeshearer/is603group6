#Customer segmentation: clustering
#Which retention strategies are best for which customers?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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