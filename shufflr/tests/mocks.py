

import numpy as np

from .. import keyutils
from .. import utils
from .. import core


def create_sample_set(num_items, shape=(10, 10), label_max=10):
    depth = utils.max_depth(num_items)
    dataset = dict()
    for index in xrange(num_items):
        key = keyutils.index_to_key(index, depth)
        dataset[key] = core.Sample(name=key,
                                   value=np.zeros(shape) + index,
                                   labels=["%s" % (index % label_max)])
    return dataset


def create_sequence_set(num_items, shape=(10, 10), label_max=10):
    depth = utils.max_depth(num_items)
    dset = dict()
    for idx in xrange(num_items):
        key = keyutils.index_to_key(idx, depth)
        label_seq = ["%s" % (idx % label_max)] * shape[0]
        dset[key] = core.Sequence(name=key,
                                  value=np.zeros(shape) + idx,
                                  labels=[label_seq])
    return dset


def populate_datapointfile(dfile, num_items, shape=(10, 10), label_max=10):
    depth = utils.max_depth(num_items)
    for index in xrange(num_items):
        sample = core.Sample(
            value=np.zeros(shape) + index,
            label="%s" % (index % label_max))
        dfile.write(keyutils.index_to_key(index, depth), sample)

    dfile.create_tables(True)


def populate_datasequencefile(dsfile, num_items, shape=(10, 10), label_max=10):
    depth = utils.max_depth(num_items)
    for index in xrange(num_items):
        datapoint = core.DataSequence(
            name="",
            value=np.zeros(shape) + index,
            label=["%s" % (index % label_max)] * shape[0])
        dsfile.write(keyutils.index_to_key(index, depth), datapoint)

    dsfile.create_tables(True)
