"""
"""

import argparse
import cPickle

from matplotlib.pyplot import figure, show

from ejhumphrey.shufflr import core
from ejhumphrey.shufflr import keyutils
from ejhumphrey.shufflr import sources
from ejhumphrey.shufflr import selectors


def load_mnist(filename):
    """Load the MNIST dataset into memory.

    Parameters
    ----------
    filename : str
        Path to MNIST pickle file.

    Returns
    -------
    dset : dict of dicts
        MNIST data structure; top-level has three keys (train, valid, test),
        and each secondary dict has two (values, labels). The value array has
        been reshaped as (num_obs, dim0, dim1), where dim0 == dim1 == 28.
    """
    dataset = dict()
    for split, data in zip(['train', 'valid', 'test'],
                           cPickle.load(open(filename))):
        dataset[split] = dict(values=data[0].reshape(len(data[0]), 28, 28),
                              labels=data[1])

    return dataset


def populate_file(values, labels, sfile):
    """Add MNIST images to an instantiated SampleFile.

    Parameters
    ----------
    values : array_like
        MNIST image data; first dimension corresponds to unique observations.
    labels : array_like
        MNIST labels; first dimension corresponds to unique observations.
    sfile : instantiated SampleFile
        File object to add MNIST data.
    """
    # Depth is 2, as 256**2 > 50000 (the number of datapoints).
    key_gen = keyutils.uniform_keygen(2)
    for value, label, key in zip(values, labels, key_gen):
        sample = core.Sample(value=value, labels=[label], name=key)
        sfile.add(key=key, data=sample)

    sfile.create_tables()


def demo(sfile, n_iter, cache_size=1000, batch_size=10):

    # Create a permutation selector to build the cache.
    cache_selector = selectors.Permutation(sfile)
    cache = sources.Cache(cache_selector, cache_size, 0.05)

    batch_selector = selectors.Random(cache)
    batch = sources.Batch(batch_selector, batch_size)

    for n in range(n_iter):
        batch.next()
        draw_batch(batch)


def draw_batch(batch):
    fig = figure()
    num_imgs = min([10, len(batch)])
    for n in range(num_imgs):
        ax = fig.add_subplot(1, num_imgs, n + 1)
        ax.imshow(batch.values()[n])
        ax.set_title("%s" % batch.labels('0')[n])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

    show()


def main(args):
    """
    """
    dset = load_mnist(args.data_file)
    sfile = sources.SampleFile(args.output_file)
    populate_file(dset['train']['values'], dset['train']['labels'], sfile)

    demo(sfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Demonstration of loading MNIST data into the Shufflr "
                    "format, and making some sample queries.")

    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Path to pickled MNIST data.")

    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Shufflr file to create.")

    main(parser.parse_args())
