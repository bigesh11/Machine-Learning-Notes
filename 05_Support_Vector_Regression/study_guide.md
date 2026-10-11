# Support Vector Regression (SVR)

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
The regression version of the Support Vector Machine. Instead of minimizing squared error for every point, it fits a function with an **epsilon-insensitive tube**: errors smaller than epsilon are ignored.

## Key idea
- Draw a tube of half-width **epsilon (eps)** around the predicted line/curve.
- Points **inside** the tube cost nothing. Points **outside** are penalized by how far outside they are (slack).
- The points on or outside the tube boundary are the **support vectors**; only they determine the fit.

## Hyperparameters
- **epsilon**: tube width (larger = simpler, less accurate model).
- **C**: how strongly to penalize points outside the tube (large C = fit tightly, risk of overfitting).
- **kernel**: linear, polynomial or **RBF** (default; handles curves). **gamma** controls how local the RBF is.

## Practical rules
- **Feature scaling is essential** (and for SVR you usually scale **y** too), because the method is distance-based.
- Remember to **inverse-transform** predictions back to the original units of y.
- Works well on small and medium data; slow on very large datasets.

## Compared with other regressions
Robust to outliers inside the tube, can model non-linear shapes through kernels, but has more settings to tune than linear regression.

## Example
Predict salary from position level with an RBF kernel; the curve bends to follow the steep rise at senior levels.
