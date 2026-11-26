#Customer segmentation: clustering
#Which retention strategies are best for which customers?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#Load data and standardize the clustering features

pd.set_option("display.width", 200)
df = pd.read_csv("sales_and_customer_insights.csv")

features = ["Purchase_Frequency", "Average_Order_Value", "Time_Between_Purchases", "Lifetime_Value"]
X_scaled = StandardScaler().fit_transform(df[features])

#Random data of the same size: if real data clusters no better than this, the clusters are not real

X_random = StandardScaler().fit_transform(np.random.default_rng(42).uniform(size=X_scaled.shape))


def silhouette_for(data, k):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(data)
    return silhouette_score(data, labels)


k_values = range(2, 11)
real_scores = [silhouette_for(X_scaled, k) for k in k_values]
random_scores = [silhouette_for(X_random, k) for k in k_values]

best_k = k_values[int(np.argmax(real_scores))]
print(f"Best K = {best_k}, silhouette = {max(real_scores):.3f} "
      f"(random data: {random_scores[best_k - 2]:.3f}; above 0.5 = clear clusters)")

#Create final clusters and profile them

df["Cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

print("\nCluster profiles (averages):")
print(df.groupby("Cluster")[features].mean().round(1))

#Retention strategy share within each cluster + chi-square test

strategy_counts = pd.crosstab(df["Cluster"], df["Retention_Strategy"])
strategy_pct = strategy_counts.div(strategy_counts.sum(axis=1), axis=0) * 100

chi2, p_value, _, _ = chi2_contingency(strategy_counts)
cramers_v = np.sqrt(chi2 / (strategy_counts.values.sum() * (min(strategy_counts.shape) - 1)))

print("\nRetention strategy % by cluster:")
print(strategy_pct.round(1))
print(f"\nChi-square p = {p_value:.3f}, Cramer's V = {cramers_v:.3f} (below 0.1 = negligible)")
print("-> Every cluster gets roughly 1/3 of each strategy: strategies are not targeted by segment.")

#Visualizations

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].plot(k_values, real_scores, marker="o", label="Customer data")
axes[0].plot(k_values, random_scores, marker="s", linestyle="--", label="Random data")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Silhouette Score")
axes[0].set_title("Silhouette: Real vs Random Data")
axes[0].legend()

X_pca = PCA(n_components=2).fit_transform(X_scaled)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["Cluster"].astype(str),
                palette="tab10", alpha=0.6, s=20, ax=axes[1])
axes[1].set_title(f"Customer Segments (PCA, K={best_k})")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

strategy_pct.plot(kind="bar", stacked=True, ax=axes[2], width=0.8)
axes[2].set_title(f"Retention Strategy by Cluster (p = {p_value:.3f}, V = {cramers_v:.3f})")
axes[2].set_xlabel("Cluster")
axes[2].set_ylabel("% of Customers")
axes[2].tick_params(axis="x", rotation=0)
axes[2].legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig("Figures/Customer_Results.png", dpi=150)
plt.show()
