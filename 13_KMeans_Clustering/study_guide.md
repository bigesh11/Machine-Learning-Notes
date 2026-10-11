# K-Means Clustering

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
An **unsupervised** algorithm that groups data into **K clusters** of similar points. There are no labels.

## Algorithm
1. Choose **K**.
2. Place K **centroids** (starting points). **k-means++** picks them spread apart, which is better than purely random.
3. **Assign** every point to its nearest centroid.
4. **Update**: move each centroid to the mean of its points.
5. Repeat steps 3 and 4 until the assignments stop changing.

## What it optimizes
**WCSS** (within-cluster sum of squares, also called **inertia**): sum over clusters of sum of ||x - centroid||^2. Lower means tighter clusters.

## Choosing K
- **Elbow method**: plot WCSS against K; pick the "elbow" where the curve stops dropping sharply (adding more clusters gives little gain).
- **Silhouette score**: higher is better (see the evaluation notes).

## Things to remember
- **Scale the features** first (distance-based).
- Result depends on the starting centroids -> run several times (`n_init`) and keep the best.
- Assumes roughly **round, similar-size** clusters; poor for elongated shapes or very different densities.
- Sensitive to **outliers** (they drag centroids).
- **Random initialization trap**: bad starts give poor clusters; k-means++ solves most of this.

## Example
Group mall customers by annual income and spending score to find segments such as "high income, high spenders".
