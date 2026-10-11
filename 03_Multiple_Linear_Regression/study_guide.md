# Multiple Linear Regression

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
Linear regression with **several** input features.

## Equation
**y = b0 + b1*x1 + b2*x2 + ... + bn*xn**
Each coefficient bi is the effect of feature xi on y **while the other features are held fixed**.

## Categorical features
Convert them with **one-hot (dummy) encoding**. Always drop one dummy column, otherwise the columns add up to a constant and cause the **dummy variable trap** (perfect multicollinearity).

## Choosing which features to keep (feature selection)
- **Backward elimination**: start with all features, fit the model, remove the feature with the highest p-value above the significance level (usually **0.05**), refit, repeat.
- **Forward selection**: start with none, add the best feature one at a time.
- **Bidirectional elimination**: combine both. Alternatively score all subsets.
A low **p-value** means the feature's effect is statistically significant.

## Evaluating
- **R-squared** always increases when you add features, even useless ones. So also use **adjusted R-squared** = 1 - (1 - R^2)(n - 1)/(n - p - 1), which penalizes extra features.
- RMSE / MAE on the test set.

## Problems to know
- **Multicollinearity**: features strongly correlated with each other make coefficients unstable (check correlation or VIF).
- Outliers pull the fitted plane toward them.
- Scaling is not required for the plain model, but is needed if you compare coefficients or use regularization.

## Example
Predict a startup's profit from R&D spend, administration spend, marketing spend and state (a categorical feature).
