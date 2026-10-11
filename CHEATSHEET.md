# Machine Learning Cheat Sheet

## Which algorithm when?
| Problem | Try first | Notes |
|---|---|---|
| Predict a number, linear trend | Linear regression | one feature = simple, many = multiple |
| Predict a number, curved trend | Polynomial regression, SVR, Random forest | watch the degree (overfitting) |
| Predict a category | Logistic regression, KNN, Naive Bayes, SVM, Random forest | scale for KNN/SVM/logistic |
| Non-linear class boundary | Kernel SVM (RBF), Random forest | tune C and gamma |
| Group unlabeled data, K known | K-means | elbow + silhouette to choose K |
| Group unlabeled data, K unknown | Hierarchical clustering | read K from the dendrogram |
| Items bought together | Apriori | support, confidence, lift |
| Too many features | PCA (linear), Kernel PCA (non-linear) | keep about 95% variance |
| Find objects in images / video | YOLO | box + class + confidence |

## Needs feature scaling?
**Yes:** KNN, SVM/SVR, K-means, hierarchical, PCA, logistic regression (regularized), neural nets.
**No:** decision tree, random forest, Naive Bayes (mostly).

## Formulas
- Linear regression: y = b0 + b1x1 + ... ; OLS minimizes sum (y - y_hat)^2
- R-squared = 1 - SSR/SST ; Adjusted R-squared = 1 - (1-R^2)(n-1)/(n-p-1)
- Sigmoid: p = 1 / (1 + e^(-z))
- Euclidean distance: sqrt( sum (xi - yi)^2 )
- Gini = 1 - sum p_i^2 ; Entropy = - sum p_i log2 p_i
- Bayes: P(C|X) = P(X|C) P(C) / P(X)
- RBF kernel: exp(-gamma ||x - x'||^2)
- Standardization: z = (x - mean) / std ; Min-max: (x - min) / (max - min)

## Classification metrics
- Accuracy = (TP+TN)/all ; Precision = TP/(TP+FP) ; Recall = TP/(TP+FN) ; F1 = 2PR/(P+R)
- ROC-AUC: 0.5 = guessing, 1 = perfect

## Unsupervised metrics
- WCSS (elbow), Silhouette (-1 to 1, higher better), Davies-Bouldin (lower better), Calinski-Harabasz (higher better), ARI (needs labels)
- Apriori: support = count/N ; confidence = supp(A,B)/supp(A) ; lift = conf/supp(B) (> 1 is useful)

## YOLO
- IoU = overlap area / union area ; NMS removes duplicate boxes ; mAP50 = detection counted correct at IoU >= 0.5
- Label: class x_center y_center width height (normalized 0-1)

## Golden rules
1. Split into train/test **before** scaling or fitting anything (no data leakage).
2. Judge a model on **unseen** data; use cross-validation.
3. Big gap between train and test score = overfitting.
4. Accuracy is misleading on imbalanced classes; use precision/recall/F1.
