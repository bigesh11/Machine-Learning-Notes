"""Simple linear regression: salary vs experience (synthetic data)."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
experience = rng.uniform(0, 10, 60)
salary = 30000 + 5000 * experience + rng.normal(0, 3000, 60)   # true line + noise

X = experience.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, salary, test_size=0.2, random_state=0)

model = LinearRegression().fit(X_train, y_train)
print(f"Intercept b0 = {model.intercept_:.0f}, slope b1 = {model.coef_[0]:.0f}  (true: 30000 and 5000)")

pred = model.predict(X_test)
print(f"Test R2  = {r2_score(y_test, pred):.3f}")
print(f"Test RMSE = {np.sqrt(mean_squared_error(y_test, pred)):.0f}")

# The same slope from the OLS formula, by hand
x, y = X_train.ravel(), y_train
b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b0 = y.mean() - b1 * x.mean()
print(f"By formula:  b0 = {b0:.0f}, b1 = {b1:.0f}  (matches scikit-learn)")
print("Predicted salary for 5 years experience:", round(model.predict([[5]])[0]))
