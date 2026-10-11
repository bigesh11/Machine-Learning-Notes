"""Random forest: accuracy, OOB score and feature importance."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=0, stratify=data.target)

tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
forest = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0).fit(X_train, y_train)

print(f"Single tree  test accuracy: {tree.score(X_test, y_test):.3f}")
print(f"Random forest test accuracy: {forest.score(X_test, y_test):.3f}")
print(f"Forest out-of-bag score:     {forest.oob_score_:.3f}  (free validation estimate)")
print(f"Forest 5-fold CV accuracy:   {cross_val_score(forest, data.data, data.target, cv=5).mean():.3f}")

print("\nTop 5 most important features:")
order = np.argsort(forest.feature_importances_)[::-1][:5]
for i in order:
    print(f"  {data.feature_names[i]:25s} {forest.feature_importances_[i]:.3f}")
