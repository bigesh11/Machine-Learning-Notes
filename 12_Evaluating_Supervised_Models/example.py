"""Evaluation metrics, checked against their formulas."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
pred = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Accuracy  by formula {(tp + tn) / (tp + tn + fp + fn):.3f}  | sklearn {accuracy_score(y_test, pred):.3f}")
print(f"Precision by formula {tp / (tp + fp):.3f}  | sklearn {precision_score(y_test, pred):.3f}")
print(f"Recall    by formula {tp / (tp + fn):.3f}  | sklearn {recall_score(y_test, pred):.3f}")
p, r = tp / (tp + fp), tp / (tp + fn)
print(f"F1        by formula {2 * p * r / (p + r):.3f}  | sklearn {f1_score(y_test, pred):.3f}")
print(f"ROC-AUC   {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, data.data, data.target, cv=cv)
print(f"\n5-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}  folds={scores.round(3)}")

# Regression metrics on a tiny example
y_true = np.array([3.0, 5.0, 7.5, 10.0])
y_hat = np.array([2.5, 5.5, 7.0, 11.0])
print(f"\nMAE={mean_absolute_error(y_true, y_hat):.3f}  MSE={mean_squared_error(y_true, y_hat):.3f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_true, y_hat)):.3f}  R2={r2_score(y_true, y_hat):.3f}")
