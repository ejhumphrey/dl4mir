import optimus
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def build_model():
    x_in = optimus.Input(name="x", shape=(None, 2))
    class_idx = optimus.Input(name="y", shape=(None,), dtype='int32')
    learning_rate = optimus.Input(name='learning_rate', shape=None)

    layer0 = optimus.Affine(
        name='layer0', input_shape=x_in.shape,
        output_shape=(None, 100), act_type='tanh')

    classifier = optimus.Softmax(
        name='classifier', input_shape=layer0.output.shape,
        n_out=2, act_type='linear')

    nll = optimus.NegativeLogLikelihood(name="nll")
    posterior = optimus.Output(name='posterior')

    trainer_edges = optimus.ConnectionManager([
        (x_in, layer0.input),
        (layer0.output, classifier.input),
        (classifier.output, nll.likelihood),
        (class_idx, nll.target_idx)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, layer0.weights),
        (learning_rate, layer0.bias),
        (learning_rate, classifier.weights),
        (learning_rate, classifier.bias)])

    trainer = optimus.Graph(
        name='trainer',
        inputs=[x_in, class_idx, learning_rate],
        nodes=[layer0, classifier],
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[nll],
        updates=update_manager.connections)

    optimus.random_init(classifier.weights)

    predictor_edges = optimus.ConnectionManager([
        (x_in, layer0.input),
        (layer0.output, classifier.input),
        (classifier.output, posterior)])

    predictor = optimus.Graph(
        name='predictor',
        inputs=[x_in],
        nodes=[layer0, classifier],
        connections=predictor_edges.connections,
        outputs=[posterior])

    driver = optimus.Driver(graph=trainer, name='test')
    return driver, predictor


def parabola(x_range=(-5, 5), scale=1, x_offset=0, y_offset=0):
    while True:
        x = np.random.rand()*np.abs(np.diff(x_range)) + x_range[0]
        y = scale * np.power(x - x_offset, 2.0) - y_offset
        yield np.array([x, y]).squeeze()


def gaussian2d(x_mean, y_mean, x_std, y_std):
    while True:
        x = np.random.normal(loc=x_mean, scale=x_std)
        y = np.random.normal(loc=y_mean, scale=y_std)
        yield np.array([x, y])


def multiplex_streams(streams, probs, batch_size=20):
    cdf = np.cumsum(probs)
    while True:
        x, y = [], []
        while len(y) < batch_size:
            idx = (cdf < np.random.rand()).argmin()
            x.append(streams[idx].next())
            y.append(idx)
        yield dict(x=np.array(x), y=np.array(y))


def train(stream1, stream2):
    driver1, predictor1 = build_model()
    source1 = multiplex_streams([stream1, stream2], [0.5, 0.5])

    hyperparams = dict(learning_rate=0.01)
    driver1.fit(source1, hyperparams=hyperparams,
                print_freq=500, max_iter=10000)

    driver2, predictor2 = build_model()
    source2 = multiplex_streams([stream1, stream2], [0.1, 0.9])

    driver2.fit(source2, hyperparams=hyperparams,
                print_freq=500, max_iter=10000)
    return predictor1, predictor2


def plot_models(stream1, stream2, predictor1, predictor2):
    source = multiplex_streams([stream1, stream2], [0.5, 0.5], 1000)
    x = source.next()['x']
    y_est1 = predictor1(x).values()[0].argmax(axis=1)
    y_est2 = predictor2(x).values()[0].argmax(axis=1)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(x[y_est1 == 0, 0], x[y_est1 == 0, 1], c='b')
    ax1.scatter(x[y_est1 == 1, 0], x[y_est1 == 1, 1], c='g')

    ax2 = fig.add_subplot(122)
    ax2.scatter(x[y_est2 == 0, 0], x[y_est2 == 0, 1], c='b')
    ax2.scatter(x[y_est2 == 1, 0], x[y_est2 == 1, 1], c='g')

    plt.show()


def eval_models(stream1, stream2, predictor1, predictor2):
    source1 = multiplex_streams([stream1, stream2], [0.5, 0.5], 1000)
    source2 = multiplex_streams([stream1, stream2], [0.1, 0.9], 1000)
    for source in source1, source2:
        data = source.next()
        y_est1 = predictor1(data['x']).values()[0].argmax(axis=1)
        y_est2 = predictor2(data['x']).values()[0].argmax(axis=1)
        print metrics.precision_recall_fscore_support(data['y'], y_est1)
        print metrics.precision_recall_fscore_support(data['y'], y_est2)


def main():
    stream1 = parabola((-2, 2), 2.5)
    stream2 = gaussian2d(0, 5, 0.25, 0.5)
    predictor1, predictor2 = train(stream1, stream2)
    plot_models(stream1, stream2, predictor1, predictor2)
    eval_models(stream1, stream2, predictor1, predictor2)
