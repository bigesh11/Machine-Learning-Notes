# Data Preprocessing

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
Raw data is messy. Preprocessing turns it into clean numeric input a model can use. Models are only as good as their data.

## Steps
1. **Import** the data (pandas `read_csv`) and separate **features X** from **target y**.
2. **Missing values**: either drop the rows/columns, or **impute** (fill in) with the mean, median (better with outliers) or most frequent value (for categories).
3. **Categorical data -> numbers**
   - **Label encoding**: Yes/No -> 1/0, or ordered categories (Low < Medium < High).
   - **One-hot encoding**: one 0/1 column per category (Nepal, India, China). Use for unordered categories so the model doesn't assume an order.
4. **Train/test split** (commonly 80/20). Do this **before** scaling.
5. **Feature scaling**
   - **Standardization**: z = (x - mean) / std. Result has mean 0, std 1.
   - **Normalization (min-max)**: x' = (x - min) / (max - min). Result in [0, 1].
   - **Fit the scaler on the training set only**, then use it to transform the test set (otherwise information leaks from the test set).
6. **Outliers**: detect with boxplots or the IQR rule (outside Q1 - 1.5*IQR to Q3 + 1.5*IQR); then remove, cap, or transform (e.g. log).

## Why scale?
Algorithms based on **distances or gradients** (KNN, SVM, K-means, PCA, neural nets) are dominated by large-valued features (salary in thousands vs age in tens). **Tree-based models** (decision tree, random forest) do not need scaling.

## Remember
- Dummy/one-hot: for regression drop one column to avoid the **dummy variable trap**.
- Use a `Pipeline` / `ColumnTransformer` in scikit-learn so every step is applied identically to train and test data.
