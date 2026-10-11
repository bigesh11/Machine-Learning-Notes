# Evaluating Supervised Models

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## Regression metrics
| Metric | Formula | Meaning |
|---|---|---|
| MAE | mean(abs(y - y_hat)) | average error, same unit as y |
| MSE | mean((y - y_hat)^2) | punishes large errors more |
| RMSE | sqrt(MSE) | error in y's unit |
| R-squared | 1 - SSR/SST | share of variance explained (1 = perfect) |
| Adjusted R-squared | 1 - (1-R^2)(n-1)/(n-p-1) | R-squared penalized for extra features |

## Classification metrics: the confusion matrix
|  | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP (true positive) | FN (false negative) |
| **Actually negative** | FP (false positive) | TN (true negative) |

- **Accuracy** = (TP + TN) / total. Misleading with imbalanced classes (99% "no fraud" gives 99% accuracy while catching nothing).
- **Precision** = TP / (TP + FP): of the predicted positives, how many are right.
- **Recall (sensitivity)** = TP / (TP + FN): of the real positives, how many we found.
- **F1** = 2 * P * R / (P + R): harmonic mean of precision and recall.
- **ROC curve**: true-positive rate vs false-positive rate at all thresholds. **AUC** = area under it (0.5 = guessing, 1 = perfect).
- Choose by cost: missing a disease (FN) is worse -> favour recall; flagging innocent email as spam (FP) is worse -> favour precision.

## Validation methods
- **Train/test split**: simple but the score depends on the lucky/unlucky split.
- **k-fold cross-validation**: split into k parts, train on k-1 and test on 1, repeat k times and average. More reliable.
- **Stratified** splitting keeps class proportions the same in each fold.

## Overfitting and underfitting
- Big gap between train and test score -> overfitting. Low scores on both -> underfitting.
- **Bias-variance trade-off**: simple models have high bias, complex ones high variance.

## Tuning
**Grid search** tries every combination of hyperparameters with cross-validation; **random search** samples combinations (faster).
