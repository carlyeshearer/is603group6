#Churn prediction: classification.
#Which customers will churn?

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#Load data and convert dates to year/month

df = pd.read_csv("sales_and_customer_insights.csv")

for col in ["Launch_Date", "Peak_Sales_Date"]:
    dates = pd.to_datetime(df[col], errors="coerce")
    df[col + "_Year"] = dates.dt.year
    df[col + "_Month"] = dates.dt.month
    df = df.drop(columns=col)

#1. Create churn label: infrequent buyers (bottom 25%) with long gaps (top 25%)

df["Churn"] = (
    (df["Purchase_Frequency"] <= 5) &
    (df["Time_Between_Purchases"] >= 68)
).astype(int)

print("Churn rate:", round(df["Churn"].mean(), 3))

#2. Validate churn definition: churners should have higher Churn_Probability

churners = df.loc[df["Churn"] == 1, "Churn_Probability"]
others = df.loc[df["Churn"] == 0, "Churn_Probability"]
p_value = mannwhitneyu(churners, others).pvalue

print("\n--- Churn definition check ---")
print(f"Churn_Probability: churners {churners.mean():.3f} vs others {others.mean():.3f} (p = {p_value:.3f})")
print("-> The label does NOT match Churn_Probability; treat it as a behavioural proxy.")

#3. Prevent leakage: remove columns used to build the label, IDs and outcome-like columns

leak_columns = ["Purchase_Frequency", "Time_Between_Purchases", "Churn_Probability", "Retention_Strategy"]
id_columns = ["Customer_ID", "Product_ID", "Transaction_ID"]

X = df.drop(columns=["Churn"] + leak_columns + id_columns)
y = df["Churn"]

numerical = X.select_dtypes(include="number").columns
categorical = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Leakage demo: a model that is allowed to see the label-defining columns

leaky_X = df[["Purchase_Frequency", "Time_Between_Purchases"]]
leaky_auc = cross_val_score(DecisionTreeClassifier(max_depth=3), leaky_X, y, cv=cv, scoring="roc_auc").mean()
print(f"\n--- Leakage check ---\nWith label columns included: ROC-AUC = {leaky_auc:.2f} (cheating, so they are removed)")

#4. Class imbalance: compare models x resampling techniques with 5-fold CV

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

samplers = {
    "None": None,
    "SMOTE": SMOTE(random_state=42),
    "Undersampling": RandomUnderSampler(random_state=42)
}


def make_pipeline(model, sampler):
    steps = [("prep", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)


results = []
for model_name, model in models.items():
    for sampler_name, sampler in samplers.items():
        auc = cross_val_score(make_pipeline(model, sampler), X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        results.append({"Model": model_name, "Imbalance": sampler_name, "ROC-AUC": auc})

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n--- Cross-validated ROC-AUC (0.5 = coin flip) ---")
print(results_df.round(3).to_string(index=False))

#5. Is the best model better than chance? (permutation test)

best = results_df.iloc[0]
best_pipeline = make_pipeline(models[best["Model"]], samplers[best["Imbalance"]])

score, _, perm_p = permutation_test_score(
    best_pipeline, X_train, y_train, cv=cv, scoring="roc_auc",
    n_permutations=50, random_state=42, n_jobs=-1
)
print(f"\n--- Permutation test: {best['Model']} + {best['Imbalance']} ---")
print(f"ROC-AUC = {score:.3f}, p = {perm_p:.2f}")
print("-> Better than chance" if perm_p < 0.05 else "-> NOT better than chance: predictors carry no churn signal")

#6. Final check on the hold-out test set

best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Hold-out test results ---")
print(classification_report(y_test, y_pred, zero_division=0))
print("Test ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

#Visualizations

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df["Churn"].value_counts().sort_index().plot(kind="bar", ax=axes[0])
axes[0].set_xticklabels(["Non-churn", "Churn"], rotation=0)
axes[0].set_title("Churn Distribution")

sns.barplot(data=results_df, x="Model", y="ROC-AUC", hue="Imbalance", ax=axes[1])
axes[1].axhline(0.5, color="black", linestyle="--")
axes[1].set_ylim(0, 1)
axes[1].set_title("Model Comparison (dashed line = coin flip)")
axes[1].tick_params(axis="x", rotation=15)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-churn", "Churn"], yticklabels=["Non-churn", "Churn"], ax=axes[2])
axes[2].set_title(f"Confusion Matrix: {best['Model']} + {best['Imbalance']}")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("Figures/Churn_Results.png", dpi=150)
plt.show()
