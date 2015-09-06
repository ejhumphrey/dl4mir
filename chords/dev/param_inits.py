import numpy as np


def random_samples(obs, num_samples, shape):
    dims = obs.shape[1:]
    assert len(dims) == len(shape)
    samples = []
    for n in range(num_samples):
        slices = [np.random.randint(len(obs))]
        for idx, s in enumerate(shape):
            if dims[idx] == s:
                i0 = 0
            else:
                i0 = np.random.randint(0, dims[idx] - s)
            slices.append(slice(i0, i0 + s))
        w = obs[tuple(slices)]
        w -= w.mean()
        w /= np.abs(w).sum()
        samples.append(w)
    return np.array(samples)


def sample_init(input_data, nodes, graph):
    outputs = [input_data]
    for idx, n in enumerate(nodes):
        shape = n.weights.shape
        n.weights.value = random_samples(outputs[idx], shape[0], shape[1:])
        outputs = graph(input_data)
