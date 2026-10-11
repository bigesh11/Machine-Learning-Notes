"""Multiple linear regression with a categorical feature and adjusted R2."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

rng = np.random.default_rng(1)
n = 120
df = pd.DataFrame({
    "rnd": rng.uniform(10, 150, n),
    "admin": rng.uniform(50, 180, n),
    "marketing": rng.uniform(20, 400, n),
    "state": rng.choice(["Kathmandu", "Pokhara", "Lalitpur"], n),
})
df["profit"] = 20000 + 800 * df["rnd"] + 0.5 * df["marketing"] + rng.normal(0, 5000, n)  # admin has NO effect

X, y = df.drop(columns="profit"), df["profit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# drop="first" avoids the dummy variable trap
prep = ColumnTransformer([("state", OneHotEncoder(drop="first"), ["state"])], remainder="passthrough")
model = make_pipeline(prep, LinearRegression()).fit(X_train, y_train)

pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
p = 5                            # features after encoding: 3 numeric + 2 dummy columns
n_test = len(y_test)
adj = 1 - (1 - r2) * (n_test - 1) / (n_test - p - 1)
print(f"Test R2 = {r2:.3f}, adjusted R2 = {adj:.3f}")

names = model[0].get_feature_names_out()
for name, coef in zip(names, model[-1].coef_):
    print(f"  {name:28s} {coef:10.1f}")
print("Notice: the 'admin' coefficient is near 0, matching the data (it had no effect).")
