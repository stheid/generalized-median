from itertools import product

import numpy as np
from scipy.stats import rankdata

from generalized_median.util import to_ordering


def generalized_rank_median(data: np.array, function=None):
    """
    Calculates the generalized mean of a population of rankings.
    Generalized means, the median must not necessarily be part of the original data.

    :param data: Array-like list of full rankings to calculate the median of. Each row is supposed to be a ranking
    :param function: Function to calculate the median for
    :return: Generalized mean ranking
    """
    data = np.array(data)
    n, ranklength = data.shape

    return None


def generalized_rank_median_spearman(data: np.array):
    """
    Calculates the generalized mean of a population of rankings.
    Generalized means, the median must not necessarily be part of the original data.

    :param data: Array-like list of full rankings to calculate the median of. Each row is supposed to be a ranking
    :return: Generalized mean ranking
    """
    return rankdata(np.array(data).mean(axis=0), method="ordinal")


def kendal(ranking_pair):
    """
    operates on rankings
    :param r1:
    :param r2:
    :return:
    """
    data = np.array(ranking_pair)
    n, ranklength = data.shape
    #convert all rankings to orderings
    orderings = to_ordering(data)
    pairs = np.apply_along_axis(lambda a: np.sign(np.array([1, -1]) @ a[np.array(np.triu_indices(ranklength, 1))]), 1, orderings)
    kendal_correlation = (1 - np.abs(np.array([1, -1]) @ pairs)).mean()
    return kendal_correlation



if __name__ == '__main__':
    rankings = [[1,2,3], [3,2,1]]
    mean = kendal(rankings)
    print(mean)
