"""Data preprocessing demo: impute, encode, split, scale (no data leakage)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({
    "Country":   ["Nepal", "India", "Nepal", "China", "India", "China", "Nepal", "India", "China", "Nepal"],
    "Age":       [44, 27, 30, 38, 40, 35, np.nan, 48, 50, 37],
    "Salary":    [72000, 48000, 54000, 61000, np.nan, 58000, 52000, 79000, 83000, 67000],
    "Purchased": ["No", "Yes", "No", "No", "Yes", "Yes", "No", "Yes", "No", "Yes"],
})
print("Missing values per column:\n", df.isna().sum(), "\n")

X = df.drop(columns="Purchased")
y = (df["Purchased"] == "Yes").astype(int)          # label encoding

# Split FIRST, then fit the preprocessing on the training part only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

prep = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="mean")),
                      ("scale", StandardScaler())]), ["Age", "Salary"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Country"]),
])
X_train_ready = prep.fit_transform(X_train)          # learn mean/std from train
X_test_ready = prep.transform(X_test)                # reuse them on test

print("Train shape:", X_train_ready.shape, "| Test shape:", X_test_ready.shape)
print("First training row:", np.round(np.asarray(X_train_ready)[0], 2))
