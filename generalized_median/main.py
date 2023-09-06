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


def generalized_rank_median_spearman(rankings: np.array):
    """
    Calculates the generalized mean of a population of rankings.
    Generalized means, the median must not necessarily be part of the original data.

    :param rankings: Array-like list of full rankings to calculate the median of. Each row is supposed to be a ranking
    :return: Generalized mean ranking
    """
    return rankdata(np.array(rankings).mean(axis=0), method="ordinal")


def generalized_rank_median_kendal(rankings):
    """
    Calculates the generalized mean of a population of rankings.
    Generalized means, the median must not necessarily be part of the original data.

    :param rankings: Array-like list of full rankings to calculate the median of. Each row is supposed to be a ranking
    :return: Generalized mean ranking
    """
    data = np.array(rankings)
    n, ranklength = data.shape
    # convert all rankings to orderings
    orderings = to_ordering(data)

    def calculate_pairwise_comparisons(ordering):
        # upper triangle matrix to calculate all pairwise comparisons
        pair_indices = np.array(np.triu_indices(ranklength, 1))
        return np.sign(np.array([-1, 1]) @ ordering[pair_indices])

    # calculate all pairwise comparisons for all orderings
    pairs = np.apply_along_axis(calculate_pairwise_comparisons, 1, orderings)
    average_agreement = pairs.mean(0)

    # reshape the pairs back into a matrix
    pairwise_ranks_as_matrix = np.zeros([ranklength, ranklength])
    pairwise_ranks_as_matrix[np.triu_indices(ranklength, 1)] = average_agreement
    pairwise_ranks_as_matrix[np.tril_indices(ranklength, -1)] = -average_agreement

    # collect average rank as the mean of pairwise comparisons
    median_ranking = rankdata(-pairwise_ranks_as_matrix.mean(1), method="ordinal")
    return median_ranking


if __name__ == '__main__':
    rankings = [[1, 2, 3, 4], [1, 2, 4, 3], [1, 4, 2, 3], [1, 3, 2, 4]]
    mean = generalized_rank_median_kendal(rankings)
    print(mean)
