# Principal Component Analysis (PCA) and Kernel PCA

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What PCA is
An **unsupervised dimensionality-reduction** method. It turns many correlated features into a few new uncorrelated features (**principal components**) that keep as much of the information (variance) as possible.

## Why use it
Fewer features -> faster models, less overfitting, easy 2-D/3-D visualization, removes redundancy (correlated features), can remove noise.

## How it works
1. **Standardize** the data (very important, otherwise large-scale features dominate).
2. Compute the **covariance matrix**.
3. Compute its **eigenvectors** (directions = the principal components) and **eigenvalues** (how much variance each direction holds).
4. Sort by eigenvalue, keep the top k components.
5. **Project** the data onto them.
PC1 holds the most variance, PC2 the next most (perpendicular to PC1), and so on.

## Choosing the number of components
Look at the **explained variance ratio**; keep enough components to reach about **95%** cumulative variance, or use a scree plot.

## Limits
- PCA is **linear**: it cannot unfold curved structure.
- The new components are combinations of the old features, so they are harder to interpret.
- It ignores the labels (it is not LDA).

## Kernel PCA
Applies the **kernel trick** (like kernel SVM) before PCA, so it can find **non-linear** structure. With the **RBF kernel** it can separate data like two concentric circles that ordinary PCA cannot. You must tune **gamma** and the kernel type.

## Remember
Fit PCA on the training set only, then transform the test set with the same fitted object.
