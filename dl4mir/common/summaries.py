import numpy as np
from scipy.spatial.distance import cdist
import optimus.util as util
from sklearn.decomposition import PCA


def normalize(x):
    u = np.sqrt(np.power(x, 2.0).sum(axis=1)).reshape(-1, 1)
    u += 1.0 * (u == 0)
    return x / u


def compression(codebook, data, metric='euclidean'):
    return cdist(codebook, data, metric=metric).mean()


def diversity(codebook, metric='euclidean'):
    n_codes, n_dims = codebook.shape
    lower_indices = np.tril_indices(n_codes, -1)
    dist_mat = cdist(codebook, codebook, metric=metric)
    return np.exp(np.log(dist_mat[lower_indices]).mean())


def fitness_score(codebook, data, metric='euclidean'):
    """Lower is better, bounded on [0, 1]"""
    # print codebook.shape, data.shape
    c = np.sqrt(2) - compression(codebook, data, metric)
    d = diversity(codebook, metric) / np.sqrt(2)
    return 2 * c * d / (c + d)


def generate_population(pop_size, n_obs, k):
        return np.array([np.sort(np.random.permutation(n_obs)[:k])
                         for _ in range(pop_size)], dtype=int)


def gibbs(energy, beta):
    y = np.exp(-beta * energy)
    return y / y.sum()


def categorical_sample(pdf):
    pdf = pdf / pdf.sum()
    return int(np.random.multinomial(1, pdf).nonzero()[0])


def genetic_assignment_search(data, pop_size, k, fitness_func, max_iter=1000,
                              beta=4.0, mutation_rate=0.005, verbose=False):
    """
    """
    n_obs, n_dim = data.shape
    pop_set = generate_population(pop_size, n_obs, k)

    best_score = -np.inf
    best_indices = None
    count = 0
    DONE = False
    while not DONE:
        scores = np.array([fitness_func(data[idx], data) for idx in pop_set])

        if scores.max() > best_score:
            best_score = scores.max()
            best_indices = np.arange(n_obs)[pop_set[scores.argmax()]]
            print ">>> %6d\tNew best: %0.5f" % (count, best_score)
        else:
            print "%6d\t        : %0.5f" % (count, scores.max())

        pdf = gibbs(scores, beta)

        children = []
        for p in range(pop_size):
            idx1 = categorical_sample(pdf)
            idx2 = idx1
            while idx2 == idx1:
                idx2 = categorical_sample(pdf)

            genes = pop_set[idx1].tolist() + pop_set[idx2].tolist()
            if np.random.binomial(1, p=mutation_rate):
                genes += np.random.permutation(n_obs)[:k].tolist()
            child = np.unique(genes)
            np.random.shuffle(child)
            children += [child[:k]]

        count += 1
        pop_set = np.array(children)

        if count >= max_iter:
            DONE = True

    return best_indices, best_score


def random_assignment_search(data, k, pop_size, fitness_func, max_iter=1000,
                             beta=4.0, mutation_rate=0.005, verbose=False):

    n_obs, n_dim = data.shape
    pop_set = generate_population(pop_size, n_obs, k)

    best_score = -np.inf
    best_indices = None
    count = 0
    DONE = False
    # ...and run!
    while not DONE:
        # Evaluate Set
        scores = np.array([fitness_func(data[idx], data) for idx in pop_set])

        if scores.max() > best_score:
            best_score = scores.max()
            best_indices = pop_set[scores.argmax()]
            if verbose:
                print "%6d\tNew best: %0.5f" % (count, best_score)

        count += 1
        pop_set = generate_population(pop_size, n_obs, k)
        if count >= max_iter:
            DONE = True

    return best_indices, best_score
