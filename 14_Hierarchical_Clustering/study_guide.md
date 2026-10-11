# Hierarchical (Agglomerative) Clustering

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
Builds a **tree of clusters** (a hierarchy). You do **not** have to choose the number of clusters in advance.

## Agglomerative (bottom-up) algorithm
1. Start with every point as its own cluster.
2. Find the two **closest clusters** and merge them.
3. Repeat until only one cluster remains.
(*Divisive* is the top-down opposite and is rarely used.)

## Distance between clusters: linkage
- **Single**: closest pair of points (can create long "chains").
- **Complete**: farthest pair of points (compact clusters).
- **Average**: average distance between all pairs.
- **Ward**: merge the pair that increases the within-cluster variance the least. Gives round, even clusters and is the common default.

## Dendrogram
A tree diagram of the merges. The **height** of a join shows the distance at which two clusters merged.
**Choosing the number of clusters**: find the **longest vertical line that no horizontal line crosses** and cut the dendrogram with a horizontal line through it; the number of vertical lines it cuts = number of clusters.

## Compared with K-means
| | Hierarchical | K-means |
|---|---|---|
| Need K up front | No (read from dendrogram) | Yes |
| Speed | Slow, memory heavy (O(n^2)) | Fast |
| Cluster shape | depends on linkage | round |
| Result | deterministic | depends on initialization |

## Remember
Scale features first. Best for small/medium datasets. Merges cannot be undone.
