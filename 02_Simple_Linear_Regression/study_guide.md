# Simple Linear Regression

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
Predicts a **number** from **one** input feature by fitting a straight line.

## Equation
**y = b0 + b1 * x**
- **b0** = intercept (value of y when x = 0). **b1** = slope (change in y for one unit of x).

## How it learns: Ordinary Least Squares (OLS)
It picks the line that minimizes the **sum of squared residuals**: SSR = sum of (y - y_hat)^2, where residual = actual - predicted.
- b1 = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
- b0 = y_mean - b1 * x_mean

## Evaluating it
- **MSE** = mean of squared errors, **RMSE** = sqrt(MSE) (same unit as y), **MAE** = mean absolute error.
- **R-squared** = 1 - SSR/SST: the fraction of the variation in y explained by x (1 = perfect, 0 = no better than predicting the mean).

## Assumptions (for reliable results)
1. **Linearity** between x and y. 2. **Independence** of observations. 3. **Homoscedasticity** (residuals have constant spread). 4. **Normally distributed residuals**.

## Example
Predicting salary from years of experience: each extra year adds about b1 to the salary.

## Remember
- Always plot the data first: if the cloud is curved, use polynomial regression instead.
- Training R-squared looks good by default; judge the model on the **test set**.
