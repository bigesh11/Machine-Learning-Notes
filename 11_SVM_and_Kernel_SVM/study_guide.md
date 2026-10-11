# Support Vector Machine (SVM) and Kernel SVM

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
A classifier that finds the **best separating line (hyperplane)** between classes: the one with the **largest margin** (the widest empty street between the classes).

## Key terms
- **Hyperplane**: the decision boundary (w . x + b = 0).
- **Margin**: distance between the boundary and the closest points; its width is 2 / ||w||. SVM maximizes it.
- **Support vectors**: the closest points on each side. Only they define the boundary; the other points can be removed without changing it.

## Soft margin and C
Real data overlaps. **C** controls the trade-off:
- large C -> few errors allowed, narrow margin (risk of overfitting).
- small C -> wider margin, more mistakes tolerated (smoother, may underfit).

## Kernel trick (Kernel SVM)
If classes are not linearly separable, the kernel implicitly maps the data into a higher dimension where a flat boundary works, **without computing the new coordinates**: K(x, x') = phi(x) . phi(x').
- **Linear**: K = x . x'
- **Polynomial**: K = (gamma * x . x' + r)^d
- **RBF (Gaussian)**: K = exp(-gamma * ||x - x'||^2), the usual default. **gamma** large -> very local, wiggly boundary (overfit); small -> smooth.

## Practical rules
- **Always scale the features.**
- Tune **C** and **gamma** together with grid search and cross-validation.
- Works well on small/medium datasets and high dimensions; slow on very large datasets; gives no direct probabilities (needs `probability=True`).

## Remember
SVR is the same idea applied to regression.
