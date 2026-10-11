# Evaluating Unsupervised Models

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## The challenge
There are **no labels**, so there is no single "accuracy". We judge how **well-separated and compact** the clusters are, or whether the result is useful.

## Clustering: internal measures (no labels needed)
- **WCSS / inertia** + **elbow method**: pick K where the drop levels off. WCSS always falls as K grows, so look for the bend.
- **Silhouette score**: s = (b - a) / max(a, b) for each point, where **a** = average distance to points in its own cluster and **b** = average distance to the nearest other cluster. Range **-1 to 1**: near 1 = well clustered, near 0 = on a border, negative = probably in the wrong cluster. Average over all points.
- **Davies-Bouldin index**: average similarity between each cluster and its closest one. **Lower is better** (0 is best).
- **Calinski-Harabasz index** (variance ratio): between-cluster vs within-cluster spread. **Higher is better**.
- **Dendrogram**: for hierarchical clustering, cut at the largest gap.

## Clustering: external measures (when true labels exist)
- **Adjusted Rand Index (ARI)**: agreement between clusters and true groups; 1 = perfect, about 0 = random.
- Also normalized mutual information, purity.

## Association rules (Apriori)
Support, confidence, lift (see the Apriori notes). Rules with high lift and enough support are the useful ones.

## Dimensionality reduction (PCA)
- **Explained variance ratio**: how much information the kept components retain (aim for roughly 95%).
- Check that a model trained on the reduced data still performs well.

## Remember
- Different metrics can disagree; combine them with common sense and **domain knowledge** ("do these customer segments make business sense?").
- Always scale features before distance-based clustering.
