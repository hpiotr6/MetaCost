from itertools import product
import numpy as np


def random_cost(y_train):
    l, counts = np.unique(y_train, return_counts=True)
    ps = counts / len(y_train)
    # cost = np.zeros((len(ps), len(ps)))
    cost = np.random.uniform(0, 2_000, (len(ps), len(ps)))
    for (i, p_i), (j, p_j) in product(enumerate(ps), enumerate(ps)):
        cost[i, j] *= p_i / p_j

    np.fill_diagonal(cost, np.random.uniform(0, 1000, len(ps)))
    return cost


def proportional_to_occurrence(y_train):
    l, counts = np.unique(y_train, return_counts=True)
    ps = counts / len(y_train)
    cost = np.ones((len(ps), len(ps)))
    for (i, p_i), (j, p_j) in product(enumerate(ps), enumerate(ps)):
        cost[i, j] *= 1000 * p_i / p_j

    np.fill_diagonal(cost, 0)
    return cost


def zeros_on_diag(y_train):
    l, counts = np.unique(y_train, return_counts=True)
    ps = counts / len(y_train)
    cost = np.ones((len(ps), len(ps)))

    np.fill_diagonal(cost, 0)
    return cost
