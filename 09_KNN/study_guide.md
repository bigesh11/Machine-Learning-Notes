# K-Nearest Neighbors (KNN)

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
A simple classifier (also usable for regression) that predicts from the **K closest training samples**. It is a **lazy learner**: there is no training step, it just stores the data and does the work at prediction time.

## Algorithm
1. Choose **K**.
2. For a new point, compute its distance to every training point.
3. Take the K nearest.
4. **Classification**: majority vote of their labels. **Regression**: average of their values.

## Distance measures
- **Euclidean**: d = sqrt( sum (xi - yi)^2 ) (most common)
- **Manhattan**: d = sum |xi - yi|
- Minkowski generalizes both.

## Choosing K
- K = 1 -> follows noise (**overfits**, jagged boundary).
- Very large K -> too smooth (**underfits**).
- Use cross-validation to choose; use an **odd K** for binary problems to avoid ties.

## Must remember
- **Feature scaling is mandatory**, because distances are dominated by large-valued features.
- Struggles with many features (**curse of dimensionality**) and is slow on big data at prediction time.
- Sensitive to irrelevant features and class imbalance.

## Pros and cons
- Pros: very simple, no training time, adapts to any boundary shape, easy to explain.
- Cons: slow prediction, memory hungry, needs scaling, weak in high dimensions.

## Example
Classify a customer as likely to buy (yes/no) from age and salary by looking at the 5 most similar customers.
