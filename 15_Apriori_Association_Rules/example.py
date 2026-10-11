"""Apriori from scratch on a tiny basket dataset (pure Python)."""
from itertools import combinations

transactions = [
    {"bread", "milk"},
    {"bread", "butter", "milk"},
    {"bread", "butter"},
    {"milk", "eggs"},
    {"bread", "butter", "milk", "eggs"},
    {"bread", "milk", "eggs"},
    {"butter", "eggs"},
    {"bread", "butter", "milk"},
]
N = len(transactions)
MIN_SUPPORT, MIN_CONFIDENCE = 0.4, 0.7


def support(itemset):
    return sum(1 for t in transactions if itemset <= t) / N


# Level-wise search with Apriori pruning
items = sorted({i for t in transactions for i in t})
frequent = {}                                     # frozenset -> support
current = [frozenset([i]) for i in items if support({i}) >= MIN_SUPPORT]
k = 1
while current:
    for s in current:
        frequent[s] = support(s)
    k += 1
    # join step: combine frequent sets into candidates of size k
    candidates = {a | b for a in current for b in current if len(a | b) == k}
    # prune step: every (k-1)-subset must itself be frequent
    candidates = [c for c in candidates if all(frozenset(sub) in frequent for sub in combinations(c, k - 1))]
    current = [c for c in candidates if support(c) >= MIN_SUPPORT]

print(f"Frequent item sets (support >= {MIN_SUPPORT}):")
for s, sup in sorted(frequent.items(), key=lambda x: (-len(x[0]), -x[1])):
    print(f"  {sorted(s)}  support={sup:.2f}")

rules = []
for s in (s for s in frequent if len(s) > 1):
    for r in range(1, len(s)):
        for a in map(frozenset, combinations(s, r)):
            b = s - a
            conf = frequent[s] / frequent[a]
            lift = conf / frequent[b] if b in frequent else conf / support(b)
            if conf >= MIN_CONFIDENCE:
                rules.append((sorted(a), sorted(b), frequent[s], conf, lift))

print(f"\nRules (confidence >= {MIN_CONFIDENCE}), best lift first:")
for a, b, sup, conf, lift in sorted(rules, key=lambda x: -x[4]):
    print(f"  {a} -> {b}  support={sup:.2f} confidence={conf:.2f} lift={lift:.2f}")
