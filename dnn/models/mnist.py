


from marl.hewey import file
from marl.hewey import sources
from ejhumphrey.dnn.core.framework import Trainer
from ejhumphrey.dnn.core.layers import Conv3DArgs
from ejhumphrey.dnn.core.layers import AffineArgs
from ejhumphrey.dnn.core.layers import SoftmaxArgs
from ejhumphrey.dnn.core.layers import Layer


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

loss_pairs = [("output", 'nll'),
              ('convlayer0/weights', 'l2_penalty')]

parameter_updates = ['affine2/bias',
                     'affine2/weights',
                     'classifier/bias',
                     'classifier/weights',
                     'convlayer0/bias',
                     'convlayer0/weights',
                     'convlayer1/bias',
                     'convlayer1/weights']

base_train_params = {"max_iterations":1000,
                     "checkpoint_freq":25,
                     "batch_size":50}

def build_layers():
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
    return [layer_0, layer_1, layer_2, classifier]


def set_all_learning_rates(hyperparams, eta):
    new_hyperparams = hyperparams.copy()
    for k in new_hyperparams:
        if k.count("learning_rate"):
            new_hyperparams[k] = eta
    return new_hyperparams


def training_source(filepath):
    file_handle = file.DataPointFile(filepath)
    return sources.RandomSource(file_handle, cache_size=5000)


def main():
    layers = build_layers()

    trainer = Trainer(name="mnist_classifier",
                      save_directory="/Users/ejhumphrey/Desktop/dnnmodels")
    trainer.build_network(layers)
    trainer.configure_losses(loss_pairs)
    trainer.configure_updates(parameter_updates)

    dset = training_source('/Volumes/speedy/mnist.dpf')
    dset.set_value_shape((1, 28, 28))
    dset.set_label_map(dict([("%d" % n, n) for n in range(10)]))

    sources = {'train':dset}
    hyperparams = set_all_learning_rates(base_hyperparams, 0.1)
    train_params = base_train_params.copy()
    train_params.update(checkpoint_freq=5)

    trainer.run(sources, train_params, hyperparams)


if __name__ == '__main__':
    main()
