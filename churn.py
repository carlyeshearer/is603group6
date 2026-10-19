#Churn prediction: classification.
#Which customers will churn?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

#Access data from Kaggle

df = pd.read_csv("sales_and_customer_insights.csv")
#print("Dataset shape:", df.shape)
#print(df.head())
#print(df.info())

#print("Unique customers:",
      #df["Customer_ID"].nunique())

#print("Total rows:",
      #len(df))

#print(df["Purchase_Frequency"].describe())
#print(df["Time_Between_Purchases"].describe())

#Convert dates and drop original columns

date_columns = [
    "Launch_Date",
    "Peak_Sales_Date"
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["Launch_Year"] = df["Launch_Date"].dt.year
df["Launch_Month"] = df["Launch_Date"].dt.month

df["Peak_Sales_Year"] = df["Peak_Sales_Date"].dt.year
df["Peak_Sales_Month"] = df["Peak_Sales_Date"].dt.month

df = df.drop(columns=date_columns)

#Create churn label

df["Churn"] = (
    (df["Purchase_Frequency"] <= 5) & #less frequent purchases (25th percentile)
    (df["Time_Between_Purchases"] >= 68) #longer time between purchases (75th percentile)
).astype(int)

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print(df["Churn"].value_counts(normalize=True))

#Define predictors and target, excluding churn_probability

X = df.drop(columns=["Churn", "Churn_Probability"]) #predictors
y = df["Churn"] #target

#Remove IDs and predictor columns

X = df.drop(columns=[
    "Churn",
    "Churn_Probability",
    "Purchase_Frequency",
    "Time_Between_Purchases",
    "Customer_ID",
    "Product_ID",
    "Transaction_ID",
    "Retention_Strategy"
])

#Identify numerical/categorical variables

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

#Preprocessing

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

#Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

#Majority class baseline

majority_class = y_train.mode()[0]

baseline_predictions = np.full(
    len(y_test),
    majority_class
)

print("\n--- Majority Class Baseline ---")
print("Accuracy:", accuracy_score(y_test, baseline_predictions))
print("Precision:", precision_score(y_test, baseline_predictions, zero_division=0))
print("Recall:", recall_score(y_test, baseline_predictions, zero_division=0))
print("F1:", f1_score(y_test, baseline_predictions, zero_division=0))

#Define classification models

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

#Train and evaluate models 

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test, y_prob
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    print("\n-----------------------------")
    print(name)
    print("-----------------------------")

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    print("ROC-AUC:", roc_auc)

#Compare models

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.sort_values(
    by="F1",
    ascending=False
))