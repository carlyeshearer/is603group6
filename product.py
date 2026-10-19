#Product affinity: association rule mining
#What products do retained customers purchase together?

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

print(df["Transaction_ID"].nunique())
print(df["Product_ID"].nunique())

print(
    df.groupby("Transaction_ID")["Product_ID"]
      .nunique()
      .value_counts()
      .sort_index()
)