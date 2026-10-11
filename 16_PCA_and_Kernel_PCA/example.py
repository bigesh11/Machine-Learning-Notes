"""PCA on the wine data, and Kernel PCA on non-linear data."""
import numpy as np
from sklearn.datasets import load_wine, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- PCA: 13 features -> a few components
wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
print("Explained variance per component:", pca.explained_variance_ratio_[:5].round(3), "...")
print("Components needed for 95% variance:", int(np.argmax(cum >= 0.95)) + 1, "of", X.shape[1])

X2 = PCA(n_components=2).fit_transform(X)
print("Reduced shape:", X2.shape, "(ready to plot in 2-D)")

# --- Kernel PCA: concentric circles (not linearly separable)
Xc, yc = make_circles(n_samples=400, factor=0.3, noise=0.08, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)

plain = LogisticRegression().fit(Xtr, ytr)
print(f"\nLogistic regression on raw circles:   {plain.score(Xte, yte):.3f}")

kpca = KernelPCA(n_components=2, kernel="rbf", gamma=10).fit(Xtr)
after = LogisticRegression().fit(kpca.transform(Xtr), ytr)
print(f"Logistic regression after Kernel PCA: {after.score(kpca.transform(Xte), yte):.3f}")
