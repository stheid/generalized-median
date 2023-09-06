# Generalized Median for Ranking Correlations


Note: the current implementation assumes the rankings don't have ties
we also operate on rankings, not on orderings.

```python
from generalized_median import generalized_rank_median_spearman
from scipy.stats import rankdata

# this eliminates ties
rankings = rankdata([[2,2,1,4,5],
                     [1,3,3,5,1]], method="ordinal")
print(generalized_rank_median_spearman(rankings))
```