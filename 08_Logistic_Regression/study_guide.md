# Logistic Regression

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
Despite the name, a **classification** algorithm. It predicts the **probability** that a sample belongs to a class (e.g. survived = 1).

## How it works
1. Compute a linear score: z = b0 + b1*x1 + ... + bn*xn.
2. Squash it into a probability with the **sigmoid** function: **p = 1 / (1 + e^(-z))**. The output is always between 0 and 1.
3. Apply a **threshold** (default 0.5): p >= 0.5 -> class 1, otherwise class 0.

## Odds and log-odds
- odds = p / (1 - p). The model is linear in the **log-odds**: ln(p/(1-p)) = b0 + b1*x1 + ...
- A coefficient bi means: one unit more of xi multiplies the odds by **e^(bi)**.

## Training: log loss
Coefficients are found by maximizing the likelihood, which means minimizing **log loss** (cross-entropy): -[y*ln(p) + (1-y)*ln(1-p)], averaged over samples. No closed form, so it uses an iterative optimizer.

## Decision boundary
It is a straight line (plane) in feature space, so it cannot separate classes that are not linearly separable (unless you add polynomial features).

## Multi-class
Use One-vs-Rest or **softmax** (multinomial) logistic regression.

## Hyperparameters
`C` = inverse of regularization strength (small C = stronger regularization). Scale features for best results.

## Evaluate with
Confusion matrix, precision, recall, F1, ROC-AUC (not R-squared). Moving the threshold trades precision against recall.
