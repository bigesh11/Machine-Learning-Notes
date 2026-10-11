"""SVM: linear vs kernel, and the effect of C and gamma."""
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=400, noise=0.25, random_state=0)   # two interleaved half-moons
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ("linear", "poly", "rbf"):
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel)).fit(X_train, y_train)
    print(f"{kernel:6s} kernel test accuracy: {model.score(X_test, y_test):.3f}")
print("The moons are not linearly separable, so the RBF/poly kernels win.\n")

grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]},
    cv=5,
).fit(X_train, y_train)
print("Best C and gamma:", grid.best_params_)
print(f"CV accuracy {grid.best_score_:.3f}, test accuracy {grid.score(X_test, y_test):.3f}")
n_sv = grid.best_estimator_[-1].n_support_.sum()
print("Number of support vectors:", n_sv, "of", len(X_train), "training points")
