# Apriori and Association Rule Mining

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What it is
Finds items that are **frequently bought together** (market basket analysis), e.g. "customers who buy bread and butter also buy milk". It produces **if-then rules**: A -> B.

## Three measures (A, B = item sets, N = number of transactions)
- **Support(A)** = transactions containing A / N. How common the item set is.
- **Confidence(A -> B)** = support(A and B) / support(A). How often B appears when A is present.
- **Lift(A -> B)** = confidence(A -> B) / support(B).
  - lift > 1: A and B appear together **more than by chance** (useful rule).
  - lift = 1: independent. lift < 1: negatively associated.

## The Apriori principle
**Every subset of a frequent item set must also be frequent.** Equivalently, if an item set is infrequent, all its supersets are infrequent, so they can be skipped. This prunes the search massively.

## Algorithm
1. Choose a **minimum support** and **minimum confidence**.
2. Find all frequent 1-item sets (support >= minimum).
3. Join them to build 2-item candidates, count them, **prune** the infrequent ones.
4. Repeat for 3-item sets, 4-item sets ... until no new frequent sets.
5. From each frequent item set generate rules and keep those with confidence >= minimum.
6. Sort rules by **lift**.

## Pros and cons
- Pros: easy to understand, gives actionable rules (store layout, recommendations, bundles).
- Cons: many scans of the data, slow with many items or low support, produces many rules.

## Remember
It is **unsupervised** (no target). Correlation is not causation: a rule shows association only.
