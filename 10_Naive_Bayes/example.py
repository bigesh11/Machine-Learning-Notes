"""Naive Bayes: Bayes by hand, then GaussianNB."""
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB

# --- Bayes' theorem by hand (the spam example from the notes)
p_spam, p_word_spam, p_word_ham = 0.4, 0.5, 0.1
posterior = (p_word_spam * p_spam) / (p_word_spam * p_spam + p_word_ham * (1 - p_spam))
print(f"P(spam | word) = {posterior:.3f}")

# --- Gaussian Naive Bayes on a real dataset
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=0, stratify=wine.target)
nb = GaussianNB().fit(X_train, y_train)
print(f"\nGaussianNB test accuracy: {nb.score(X_test, y_test):.3f}")
print(f"5-fold CV accuracy:       {cross_val_score(GaussianNB(), wine.data, wine.target, cv=5).mean():.3f}")
print("Class priors learned:", nb.class_prior_.round(3))
print("Posterior probabilities for the first test sample:", nb.predict_proba(X_test[:1]).round(3))
