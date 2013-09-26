
import numpy as np

from ..core.layers import Conv3DArgs
from ..core.layers import AffineArgs
from ..core.layers import SoftmaxArgs
from ..core.layers import Layer
from ..core.graphs import Network
from ..core.modules import Loss
from ..core.updates import SGD

from marl.hewey import file
from marl.hewey import sources

training_params = {'num_iter':10000,
                   'batch_size': 100}

base_hyperparams = {'affine2/bias/learning_rate': 0.0,
                    'affine2/dropout': 0.0,
                    'affine2/weights/learning_rate': 0.0,
                    'classifier/bias/learning_rate': 0.0,
                    'classifier/dropout': 0.0,
                    'classifier/weights/learning_rate': 0.0,
                    'convlayer0/bias/learning_rate': 0.0,
                    'convlayer0/dropout': 0.0,
                    'convlayer0/weights/l2_penalty': 0.0,
                    'convlayer0/weights/learning_rate': 0.0,
                    'convlayer1/bias/learning_rate': 0.0,
                    'convlayer1/dropout': 0.0,
                    'convlayer1/weights/learning_rate': 0.0 }

loss_pairs = [("z_output", 'nll'),
              ('convlayer0/weights', 'l2_penalty')]

base_train_params = {"max_iterations":1000,
                     "checkpoint_freq":25,
                     "batch_size":50}

def set_all_learning_rates(hyperparams, eta):
    new_hyperparams = hyperparams.copy()
    for k in new_hyperparams:
        if k.count("learning_rate"):
            new_hyperparams[k] = eta
    return new_hyperparams

def build_network():
    layer_0 = Layer(Conv3DArgs(name='convlayer0',
                           input_shape=(1, 28, 28),
                           weight_shape=(30, 5, 5),
                           pool_shape=(2, 2),
                           activation='tanh'))

    layer_1 = Layer(Conv3DArgs(name='convlayer1',
                               input_shape=layer_0.output_shape,
                               weight_shape=(50, 7, 7),
                               pool_shape=(2, 2),
                               activation='tanh'))

    layer_2 = Layer(AffineArgs(name='affine2',
                               input_shape=layer_1.output_shape,
                               output_shape=(128,),
                               activation='tanh'))

    classifier = Layer(SoftmaxArgs(name='classifier',
                                   input_dim=layer_2.output_shape[0],
                                   output_dim=10))

    dnn = Network([layer_0, layer_1, layer_2, classifier])
    dnn.compile()
    return dnn

def configure_losses(network):
    loss = Loss()
    [loss.register(network, iname, ltype) for iname, ltype in loss_pairs]
    loss.compile()
    return loss

def configure_updates(network, loss):
    sgd = SGD()
    sgd.compute_updates(loss, network.params)
    sgd.compile()
    return sgd

def training_source(filepath):
    file_handle = file.DataPointFile(filepath)
    return sources.RandomSource(file_handle, cache_size=5000)

def reshape_value(x):
    return x[np.newaxis, :, :]

def update_input_args_with_batch(input_args, batch, label_map):
    X = np.array([reshape_value(x.value()) for x in batch])
    Y = np.array([label_map.get(x.label()) for x in batch])
    input_args['x_input'] = X
    input_args['y_target'] = Y

def select_update(base, newdict):
    for k, v in newdict.iteritems():
        if k in base:
            base[k] = v

def run(network, loss, trainer, sources, train_params, hyperparams):
    Done = False
    train_inputs = trainer.empty_inputs()
    select_update(train_inputs, hyperparams)

    loss_inputs = loss.empty_inputs()
    select_update(loss_inputs, hyperparams)

    label_map = dict([("%d" % n, n) for n in range(10)])
    while not Done:
        try:
            batch = sources['train'].next_batch(train_params.get("batch_size"))
            update_input_args_with_batch(train_inputs, batch, label_map)
            train_loss = trainer(train_inputs)

            if (trainer.iteration % train_params["checkpoint_freq"]) == 0:
                print "%d: %0.4f" % (trainer.iteration, train_loss)

            if trainer.iteration >= train_params.get("max_iterations"):
                Done = True

        except KeyboardInterrupt:
            break
    print "\nFinished.\n"

def main():
    dnn = build_network()
    loss = configure_losses(dnn)
    trainer = configure_updates(dnn, loss)

    sources = {'train':training_source('/Volumes/speedy/mnist.dpf')}
    hyperparams = set_all_learning_rates(base_hyperparams, 0.1)
    train_params = base_train_params.copy()
    train_params.update(checkpoint_freq=5)

    run(dnn, loss, trainer, sources, train_params, hyperparams)


if __name__ == '__main__':
    main()
