"""Support Vector Regression with scaled X and y."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

level = np.arange(1, 11).reshape(-1, 1)                         # position level 1..10
salary = np.array([45, 50, 60, 80, 110, 150, 200, 300, 500, 1000]) * 1000.0

# Scale X inside the pipeline and scale y with TransformedTargetRegressor
model = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1)),
    transformer=StandardScaler(),
)
model.fit(level, salary)

for lv in (3, 6.5, 9):
    print(f"Predicted salary at level {lv}: {model.predict([[lv]])[0]:,.0f}")
# TransformedTargetRegressor automatically converts predictions back to the original salary units.
