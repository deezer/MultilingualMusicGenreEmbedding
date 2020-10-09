import numpy as np
from copy import deepcopy


def retrofit_identity(X, edges, known, n_iter=100, alpha=None, beta=None, tol=1e-2, verbose=False):
    """ Implement the retrofitting method of Faruqui et al.
    :param X: distributional embeddings
    :param edges: edge dict; if multiple types of edges,
        this will be flattened.
    :param known: graph nodes which have initial embeddings (different than 0)
    :param n_iter: the maximum number of iterations to run
    :param alpha: func from `edges.keys()` to floats or None
    :param beta: func from `edges.keys()` to floats or None
    :param tol: If the average distance change between two rounds is at or
        below this value, we stop. Default to 10^-2
    :return: Y the retrofitted embeddings
    """
    def default_alpha(x):
        """func from `edges.keys()` to floats or None
        :param x: a node in the graph
        :return: 1 if node has initial distributional embedding, 0 otherwise
        """
        return 1 if x in known else 0

    def default_beta(x, y, etypes):
        """func from `edges.keys()` to floats or None
        :param x: a node in the graph
        :param y: a node in the graph
        :param etypes: edges grouped by their types
        :return: 1/degree of node x
        """
        return 1 / len(edges[x])

    etypes = deepcopy(edges)
    if isinstance(next(iter(edges.values())), dict):
        edges = flatten_edges(edges, len(X))

    if not alpha:
        alpha = default_alpha
    if not beta:
        beta = default_beta

    Y = X.copy()
    Y_prev = Y.copy()
    for iteration in range(1, n_iter + 1):
        if verbose:
            print("Iteration {} of {}".format(iteration, n_iter), end='\r')
        for i, vec in enumerate(X):
            neighbors = edges[i]
            n_neighbors = len(neighbors)
            if n_neighbors:
                a = alpha(i)
                retro = np.array([(beta(i, j, etypes) + beta(j, i, etypes)) * Y[j] for j in neighbors])
                retro = retro.sum(axis=0) + (a * X[i])
                norm = np.array([beta(i, j, etypes) + beta(j, i, etypes) for j in neighbors])
                norm = norm.sum(axis=0) + a
                Y[i] = retro / norm
        changes = np.abs(np.mean(np.linalg.norm(
            np.squeeze(Y_prev)[:1000] - np.squeeze(Y)[:1000], ord=2)))

        if changes <= tol:
            if verbose:
                print("Converged at iteration {}".format(iteration))
            return Y
        else:
            Y_prev = Y.copy()
    if verbose:
        print("Stopping at iteration {:d}; change was {:.4f}".format(iteration, changes))
    return Y


def flatten_edges(edges, n_nodes):
    """ Flatten a dict of dict of edges of different types.
    :param edges: maps edge type to dict that maps index to neighbors
    :param n_nodes: the number of nodes in the graph.
    :return edges: dict that maps index to all neighbors
    """
    edges_naive = {}
    for i in range(n_nodes):
        edges_naive[i] = []
        for rel_name in edges.keys():
            edges_r = edges[rel_name]
            try:
                my_edges = edges_r[i]
            except KeyError:
                continue
            edges_naive[i].extend(my_edges)
    return edges_naive
