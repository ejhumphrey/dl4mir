import optimus
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

N_CLASSES = 5


def build_model():
    x_in = optimus.Input(name="x", shape=(None, 2))
    class_idx = optimus.Input(name="y", shape=(None,), dtype='int32')
    learning_rate = optimus.Input(name='learning_rate', shape=None)

    layer0 = optimus.Affine(
        name='layer0', input_shape=x_in.shape,
        output_shape=(None, 100), act_type='relu')

    layer1 = optimus.Affine(
        name='layer1', input_shape=layer0.output.shape,
        output_shape=(None, 100), act_type='relu')

    classifier = optimus.Softmax(
        name='classifier', input_shape=layer1.output.shape,
        n_out=N_CLASSES, act_type='linear')

    nll = optimus.NegativeLogLikelihood(name="nll")
    posterior = optimus.Output(name='posterior')

    trainer_edges = optimus.ConnectionManager([
        (x_in, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, classifier.input),
        (classifier.output, nll.likelihood),
        (class_idx, nll.target_idx)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, layer0.weights),
        (learning_rate, layer0.bias),
        (learning_rate, layer1.weights),
        (learning_rate, layer1.bias),
        (learning_rate, classifier.weights),
        (learning_rate, classifier.bias)])

    trainer = optimus.Graph(
        name='trainer',
        inputs=[x_in, class_idx, learning_rate],
        nodes=[layer0, layer1, classifier],
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[nll],
        updates=update_manager.connections)

    optimus.random_init(layer0.weights)
    optimus.random_init(layer1.weights)
    optimus.random_init(classifier.weights)

    predictor_edges = optimus.ConnectionManager([
        (x_in, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, classifier.input),
        (classifier.output, posterior)])

    predictor = optimus.Graph(
        name='predictor',
        inputs=[x_in],
        nodes=[layer0, layer1, classifier],
        connections=predictor_edges.connections,
        outputs=[posterior])

    driver = optimus.Driver(graph=trainer, name='test')
    return driver, predictor


def gaussian2d(x_mean, y_mean, x_std, y_std):
    while True:
        x = np.random.normal(loc=x_mean, scale=x_std)
        y = np.random.normal(loc=y_mean, scale=y_std)
        yield np.array([x, y])


def multiplex_streams(streams, probs, batch_size=20):
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    cdf = np.cumsum(probs)
    while True:
        x, y = [], []
        while len(y) < batch_size:
            idx = (cdf < np.random.rand()).argmin()
            x.append(streams[idx].next())
            y.append(idx)
        yield dict(x=np.array(x), y=np.array(y))


def categorical_sample(weights):
    '''Sample from a categorical distribution.

    :parameters:
        - weights : np.array, shape=(n,)
          The distribution to sample from.
          Must be non-negative and sum to 1.0.

    :returns:
        - k : int in [0, n)
          The sample
    '''

    return np.flatnonzero(np.random.multinomial(1, weights))[0]


def train(source):
    driver, predictor = build_model()

    hyperparams = dict(learning_rate=0.01)
    driver.fit(source, hyperparams=hyperparams,
                print_freq=500, max_iter=10000)

    return predictor


def plot_model(source, predictor):
    colors = ['b', 'g', 'r', 'y', 'm']
    x = source.next()['x']
    y_est = predictor(x).values()[0].argmax(axis=1)

    fig = plt.figure()
    ax = fig.gca()
    for idx, c in enumerate(colors):
        ax.scatter(x[y_est == idx, 0], x[y_est == idx, 1], c=c)

    plt.show()


def eval_models(source1, source2, predictor1, predictor2):
    for source in source1, source2:
        data = source.next()
        y_est1 = predictor1(data['x']).values()[0].argmax(axis=1)
        y_est2 = predictor2(data['x']).values()[0].argmax(axis=1)
        # print np.array(
        #     metrics.precision_recall_fscore_support(data['y'], y_est1))
        # print np.array(
        #     metrics.precision_recall_fscore_support(data['y'], y_est2))
        print metrics.classification_report(data['y'], y_est1), '\n'
        print metrics.classification_report(data['y'], y_est2), '\n'


def generate_streams():
    streams = [gaussian2d(0, 0, 1.0, 1.0)]
    for x in -0.5, 0.5:
        for y in -0.5, 0.5:
            streams += [gaussian2d(x, y, 0.1, 0.1)]
    return streams


def main():
    streams = generate_streams()
    source1 = multiplex_streams(streams, np.ones(len(streams)), 200)
    source2 = multiplex_streams(streams, [6, 1, 1, 1, 1], 200)
    # Uniform
    predictor1 = train(source1)
    # Biased
    predictor2 = train(source2)
    return streams, predictor1, predictor2
