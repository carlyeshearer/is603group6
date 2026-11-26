#Product affinity: association rule mining
#What associations exist between seasons and most frequently purchased product categories?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("sales_and_customer_insights.csv")
data = df[["Season", "Most_Frequent_Category"]].dropna()

#Turn each row into a basket, e.g. ["Season=Winter", "Category=Electronics"]

transactions = [
    [f"Season={season}", f"Category={category}"]
    for season, category in zip(data["Season"], data["Most_Frequent_Category"])
]

encoder = TransactionEncoder()
basket_df = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)

#Find frequent item sets and Season -> Category rules (lift = 1 means no relationship)

frequent_itemsets = apriori(basket_df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

season_rules = rules[rules["antecedents"].apply(
    lambda items: any(item.startswith("Season=") for item in items)
)].sort_values("lift", ascending=False)

print("Top Season -> Category rules:")
print(season_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
      .head(10).to_string(index=False))

#Is the relationship real? Chi-square test + Cramer's V (effect size)

table = pd.crosstab(data["Season"], data["Most_Frequent_Category"])
chi2, p_value, _, _ = chi2_contingency(table)
cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))

print(f"\nChi-square p = {p_value:.3f}, Cramer's V = {cramers_v:.3f} (below 0.1 = negligible)")
print("-> Season and category are NOT related; lifts near 1.0 are random noise."
      if p_value >= 0.05 else "-> Season and category are related.")

#Visualization

category_pct = pd.crosstab(data["Season"], data["Most_Frequent_Category"], normalize="index") * 100

plt.figure(figsize=(10, 6))
sns.heatmap(category_pct, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title(f"Preferred Product Category by Season (%)\nChi-square p = {p_value:.2f}, Cramer's V = {cramers_v:.3f}")
plt.xlabel("Product Category")
plt.ylabel("Season")
plt.tight_layout()
plt.savefig("Figures/Product_Heatmap.png", dpi=150)
plt.show()
