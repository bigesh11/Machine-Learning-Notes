"""Decision tree: classification on iris and a regression tree."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0, stratify=iris.target)

for depth in (1, 3, None):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    print(f"max_depth={str(depth):4s}  train acc={tree.score(X_train, y_train):.3f}  test acc={tree.score(X_test, y_test):.3f}")

small = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
print("\nThe rules of a depth-2 tree:")
print(export_text(small, feature_names=list(iris.feature_names)))

# Regression tree: predictions are step-shaped (mean of each leaf)
x = np.arange(1, 11).reshape(-1, 1)
y = np.array([45, 50, 60, 80, 110, 150, 200, 300, 500, 1000], dtype=float)
reg = DecisionTreeRegressor(max_depth=2, random_state=0).fit(x, y)
print("Regression tree predictions for levels 1..10:", reg.predict(x))
