"""KNN: effect of K, with feature scaling."""
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()
print("K  | 5-fold CV accuracy")
best_k, best_score = None, 0
for k in range(1, 16, 2):
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    score = cross_val_score(model, iris.data, iris.target, cv=5).mean()
    print(f"{k:2d} | {score:.3f}")
    if score > best_score:
        best_k, best_score = k, score
print(f"\nBest K = {best_k} (accuracy {best_score:.3f})")

# What happens without scaling? (iris features have similar ranges, so the effect is small here;
# with age vs salary data it would be huge)
raw = cross_val_score(KNeighborsClassifier(n_neighbors=best_k), iris.data, iris.target, cv=5).mean()
print(f"Without scaling: {raw:.3f}")
