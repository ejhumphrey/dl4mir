"""Demonstration of the Shufflr library with the MNIST dataset."""

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


def populate_file(values, labels, fobj):
    """Add MNIST images to an instantiated File.

    Parameters
    ----------
    values : array_like
        MNIST image data; first dimension corresponds to unique observations.
    labels : array_like
        MNIST labels; first dimension corresponds to unique observations.
    fobj : Instantiated File
        File object to add MNIST data.
    """
    # Depth is 2, as 256**2 > 50000 (the number of datapoints).
    key_gen = keyutils.uniform_keygen(2)
    for value, label, key in zip(values, labels, key_gen):
        sample = core.Sample(value=value, labels=[label], name=key)
        fobj.add(key=key, data=sample)

    fobj.create_tables()


def demo(fobj, n_iter, cache_size=1000, batch_size=10):
    """Demonstrate how to establish the shufflr pipeline to yield data batches.

    Parameters
    ----------
    fobj : Instantied File
        Data source to poll.
    n_iter : int
        Number of batches to pull from the data source.
    cache_size : int
        Number of datapoints to keep in memory.
    batch_size : int
        Number of datapoints to draw at a time.
    """
    # Create a permutation selector to build the cache.
    file_selector = selectors.Permutation(fobj)
    cache = sources.Cache(file_selector, cache_size, 0.05)

    cache_selector = selectors.Random(cache)
    batch = sources.Batch(cache_selector, batch_size)

    for n in range(n_iter):
        batch.next()
        draw_batch(batch)


def draw_batch(batch):
    """Plot the first several datapoints of a batch.

    Parameters
    ----------
    batch : Batch instance
    """
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
    """Main example routine."""
    dset = load_mnist(args.data_file)
    fobj = sources.File(args.output_file)
    populate_file(dset['train']['values'], dset['train']['labels'], fobj)
    demo(fobj)


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
