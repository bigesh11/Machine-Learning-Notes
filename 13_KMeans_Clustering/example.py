"""K-means: elbow method and silhouette score."""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=7)
X = StandardScaler().fit_transform(X)

print("K | WCSS (inertia) | silhouette")
scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(X)
    sil = silhouette_score(X, km.labels_)
    scores[k] = sil
    print(f"{k} | {km.inertia_:14.1f} | {sil:.3f}")
best = max(scores, key=scores.get)
print(f"\nBest K by silhouette: {best}  (the data was generated with 4 clusters)")

final = KMeans(n_clusters=best, n_init=10, random_state=0).fit(X)
print("Cluster sizes:", [int((final.labels_ == i).sum()) for i in range(best)])
print("Centroids:\n", final.cluster_centers_.round(2))
