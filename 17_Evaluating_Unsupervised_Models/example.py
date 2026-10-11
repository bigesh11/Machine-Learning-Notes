"""Unsupervised evaluation: silhouette, Davies-Bouldin, Calinski-Harabasz, ARI."""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

X, true = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=5)
X = StandardScaler().fit_transform(X)

print("K | silhouette (higher) | Davies-Bouldin (lower) | Calinski-Harabasz (higher)")
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print(f"{k} |       {silhouette_score(X, labels):.3f}         |"
          f"         {davies_bouldin_score(X, labels):.3f}          |"
          f"          {calinski_harabasz_score(X, labels):8.1f}")

labels4 = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
print(f"\nAdjusted Rand Index vs the true groups (K=4): {adjusted_rand_score(true, labels4):.3f}  (1 = perfect)")
