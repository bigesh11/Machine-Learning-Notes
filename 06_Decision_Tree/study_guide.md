# Decision Tree

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
A model that asks a series of yes/no questions about the features (like a flowchart) and ends in a **leaf** that gives the prediction. Works for **classification** and **regression**.

## Structure
- **Root node**: first question. **Internal nodes**: further questions. **Leaves**: final prediction.
- Each split is a threshold on one feature, e.g. "Age < 30?".

## How it chooses splits
It picks the split that makes the resulting groups as **pure** as possible.
- **Gini impurity** = 1 - sum(p_i^2), where p_i is the proportion of class i in the node (0 = pure).
- **Entropy** = - sum(p_i * log2(p_i)). **Information gain** = entropy before split - weighted entropy after.
- **Regression trees** choose splits that reduce the **variance / MSE**; a leaf predicts the **mean** of its samples (so predictions are step-shaped).

## Controlling overfitting
A fully grown tree memorizes the training data. Limit it with:
- `max_depth`, `min_samples_split`, `min_samples_leaf`, or **pruning**.

## Pros and cons
- Pros: easy to read and explain, no scaling needed, handles numeric and categorical data, captures non-linear patterns.
- Cons: **overfits easily**, unstable (a small data change can give a different tree), and each prediction is only a constant per region.

## Remember
- Many trees combined give a **Random Forest**, which fixes most of the weaknesses.
- Feature importance is available from the tree.
