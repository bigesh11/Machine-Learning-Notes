# Naive Bayes

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
A fast probabilistic **classifier** based on **Bayes' theorem**.

## Bayes' theorem
**P(C | X) = P(X | C) * P(C) / P(X)**
- P(C) = **prior**: how common the class is. P(X | C) = **likelihood**: how likely these features are within the class. P(C | X) = **posterior**: what we want.
- Predict the class with the **highest posterior**. P(X) is the same for every class, so it can be ignored when comparing.

## Why "naive"?
It assumes the features are **independent given the class**, so P(X | C) = P(x1|C) * P(x2|C) * ... * P(xn|C). This is rarely exactly true, yet it often works well anyway.

## Variants
- **Gaussian NB**: continuous features, each modelled with a normal distribution per class.
- **Multinomial NB**: word counts (text classification, spam filtering).
- **Bernoulli NB**: binary features (word present / absent).

## Laplace smoothing
If a feature value never appeared with a class, its probability would be 0 and wipe out the whole product. Add 1 to every count to avoid this.

## Worked example
P(spam) = 0.4, P(word | spam) = 0.5, P(word | ham) = 0.1.
P(spam | word) = (0.5 * 0.4) / (0.5 * 0.4 + 0.1 * 0.6) = 0.20 / 0.26 = **0.77**.

## Pros and cons
- Pros: very fast, works with small data and many features, good for text.
- Cons: independence assumption, poor probability estimates, struggles when features are strongly correlated.
