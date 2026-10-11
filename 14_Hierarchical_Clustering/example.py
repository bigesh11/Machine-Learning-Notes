"""Hierarchical clustering: linkage, cutting the tree, and a dendrogram image."""
import os
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=40, centers=3, cluster_std=0.7, random_state=2)
X = StandardScaler().fit_transform(X)

Z = linkage(X, method="ward")              # the merge history (one row per merge)
labels_scipy = fcluster(Z, t=3, criterion="maxclust")       # cut the tree into 3 clusters
labels_sk = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
print("Cluster sizes (scipy):  ", np.bincount(labels_scipy)[1:])
print("Cluster sizes (sklearn):", np.bincount(labels_sk))
print("Last 4 merge distances:", Z[-4:, 2].round(2), "-> the biggest jump suggests 3 clusters")

plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("sample"); plt.ylabel("merge distance")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dendrogram.png")
plt.tight_layout(); plt.savefig(out, dpi=120)
print("Saved", out)
