"""Polynomial regression: underfit vs good fit vs overfit."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(3)
x = rng.uniform(-3, 3, 40)
y = 0.5 * x**2 + x + 2 + rng.normal(0, 0.6, 40)        # true shape: a parabola
X = x.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("degree | train R2 | test R2")
for d in (1, 2, 10):
    model = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    tr = r2_score(y_train, model.predict(X_train))
    te = r2_score(y_test, model.predict(X_test))
    print(f"  {d:2d}   |  {tr:6.3f}  | {te:6.3f}")
print("Degree 1 underfits, degree 2 matches the true shape, degree 10 chases the noise.")
