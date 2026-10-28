#Product affinity: association rule mining
#What associations exist between seasons and most frequently purchased product categories?

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sales_and_customer_insights.csv")

#Selected attributes
columns = [
    "Season",
    "Most_Frequent_Category"
]

#Remove rows with missing values
analysis_df = df[columns].dropna().copy()

#Convert categorical values
transactions = []

for _, row in analysis_df.iterrows():
    basket = [
        f"Season={row['Season']}",
        f"Category={row['Most_Frequent_Category']}"
    ]
    transactions.append(basket)

encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)

basket_df = pd.DataFrame(
    encoded,
    columns=encoder.columns_
)

print("Number of records:", len(basket_df))
print("Number of distinct items:", len(basket_df.columns))
print(basket_df.head())


#Identify frequent item sets
frequent_itemsets = apriori(
    basket_df,
    min_support=0.05,
    use_colnames=True
)

print("Frequent item sets:")
print(frequent_itemsets.sort_values(
    "support", ascending=False
).head(10))

#Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.1
)

season_rules = rules[
    rules["antecedents"].apply(
        lambda items: any(
            item.startswith("Season=") for item in items
        )
    )
].copy()

#Sort by lift
season_rules = season_rules.sort_values(
    "lift", ascending=False
)

print("\nRules associated with seasons:")
print(
    season_rules[
        ["season", "category",
         "support", "confidence", "lift"]
    ].head(20).to_string(index=False)
)

#Visaulization

category_pct = pd.crosstab(
    analysis_df["Season"],
    analysis_df["Most_Frequent_Category"],
    normalize="index"
) * 100

plt.figure(figsize=(10, 6))
sns.heatmap(
    category_pct,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)
plt.title("Preferred Product Category by Season")
plt.xlabel("Product Category")
plt.ylabel("Season")
plt.tight_layout()
plt.show()