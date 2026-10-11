# Polynomial Regression

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
Fits a **curve** instead of a straight line by adding powers of x as extra features.

## Equation
**y = b0 + b1*x + b2*x^2 + ... + bd*x^d**   (d = the degree)
It is still called a *linear* model because it is **linear in the coefficients** b; only the features were transformed (x -> x, x^2, x^3, ...).

## How to do it
1. Create polynomial features (`PolynomialFeatures(degree=d)`).
2. Fit an ordinary linear regression on them.

## Choosing the degree
- Degree too **low** -> **underfitting** (curve too stiff, misses the pattern).
- Degree too **high** -> **overfitting** (wiggly curve passes through the noise; great training score, poor test score).
- Pick the degree with the best **cross-validated / test** score, not the best training score.

## Pros and cons
- Pros: simple, captures curved relationships.
- Cons: sensitive to outliers; high degrees behave wildly outside the data range; many features for multi-input problems.

## Example
Salary versus job position level: salary grows slowly then shoots up, so a straight line fits badly but a degree-3 or 4 polynomial fits well.

## Remember
- Scale the features if using high degrees (x^10 becomes huge).
- Compare against a straight line to prove the curve is actually needed.
