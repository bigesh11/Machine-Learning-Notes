# Random Forest

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
An **ensemble** of many decision trees whose predictions are combined: **majority vote** for classification, **average** for regression. "Many weak-ish models together beat one."

## How it is built
1. **Bagging (bootstrap aggregating)**: each tree trains on a random sample of the data **drawn with replacement** (a bootstrap sample).
2. **Random feature subsets**: at every split, a tree considers only a random subset of the features (about sqrt(n_features) for classification). This makes the trees different from each other.
3. Predict with all trees and combine.

## Why it works
A single deep tree has **high variance** (overfits). Averaging many de-correlated trees keeps low bias but **reduces variance**, so the forest generalizes much better.

## Useful extras
- **Out-of-bag (OOB) score**: each tree never saw about 37% of the samples, so those can be used as a free validation set.
- **Feature importance**: how much each feature reduced impurity across all trees.

## Hyperparameters
`n_estimators` (number of trees; more is better but slower), `max_depth`, `min_samples_leaf`, `max_features`.

## Pros and cons
- Pros: strong accuracy with little tuning, handles non-linearity and mixed data, no scaling needed, resistant to overfitting.
- Cons: less interpretable than one tree, bigger and slower than a single model.

## Remember
Random forest = bagging + random feature selection. Gradient boosting is a different ensemble idea (trees built one after another to fix earlier errors).
