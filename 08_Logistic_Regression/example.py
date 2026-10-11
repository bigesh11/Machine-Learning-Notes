"""Logistic regression: probabilities, threshold and metrics."""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=0, stratify=data.target)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
pred = model.predict(X_test)

print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=data.target_names))

proba = model.predict_proba(X_test[:3])[:, 1]
print("Probability of class 'benign' for 3 test samples:", proba.round(3))

# Changing the threshold changes the precision/recall balance
p_all = model.predict_proba(X_test)[:, 1]
for t in (0.3, 0.5, 0.9):
    print(f"threshold {t}: predicted benign = {(p_all >= t).sum()} of {len(p_all)}")
