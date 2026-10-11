# Machine Learning Overview

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What is machine learning?
Machine learning (ML) lets a computer learn patterns from data instead of being given explicit rules. We give the algorithm examples, it builds a **model**, and the model makes predictions on new data.

## Types of learning
| Type | Data | Goal | Examples in this course |
|---|---|---|---|
| **Supervised** | inputs **X** with known answers **y** (labels) | learn the mapping X -> y | regression, KNN, SVM, trees, Naive Bayes |
| **Unsupervised** | inputs only, no labels | find hidden structure | K-means, hierarchical clustering, Apriori, PCA |

Supervised problems are of two kinds:
- **Regression**: predict a number (salary, price).
- **Classification**: predict a category (spam / not spam, survived / died).

## Standard workflow
1. Collect data -> 2. Clean and preprocess -> 3. Explore (EDA) -> 4. Choose features -> 5. Split into train and test sets -> 6. Train a model -> 7. Evaluate on the test set -> 8. Tune hyperparameters (cross-validation, grid search) -> 9. Deploy.

## Key vocabulary
- **Feature**: an input column. **Target / label**: the column we predict.
- **Parameter**: learned by the model (e.g. a regression coefficient). **Hyperparameter**: set by you before training (e.g. K in KNN, tree depth).
- **Training set** is used to learn; **test set** is used only to check performance on unseen data.

## Overfitting vs underfitting
- **Underfitting** (high bias): model too simple, bad on both train and test data.
- **Overfitting** (high variance): model memorizes the training data, great on train but bad on test.
- Goal: the sweet spot in between. Tools: more data, simpler model, regularization, cross-validation.

## Remember
- Always split the data **before** scaling or fitting anything, so the test set never leaks into training.
- Pick the metric that fits the problem (accuracy is misleading on imbalanced data).
