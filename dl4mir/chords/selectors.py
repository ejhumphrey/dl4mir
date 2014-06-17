"""

dict(): keys -> entities

index:
    uid: (key, idx)

"""
import numpy as np
import mir_eval.chord as C
from . import parts_to_index
from random import choice
from ejhumphrey.dl4mir.chords.transformers import chord_sample


def entity_to_quality_index(vocabulary):
    def fx(entity):
        for root, semis in zip(*C.encode_many(entity.chord_labels.value)[:2]):
            chord_idx = parts_to_index(root, semis, vocabulary)
            if chord_idx is None:
                yield None
            else:
                yield chord_idx/12
    return fx


def index_chord_entities(obj, mapper):
    index = dict()
    for key in obj.keys():
        entity = obj.get(key)
        for idx, class_id in enumerate(mapper(entity)):
            if not class_id in index:
                index[class_id] = []
            index[class_id].append((key, idx))
    return index


class BaseSelector(object):
    def __init__(self, obj):
        self._obj = obj
        self._keys = self._obj.keys()
        self.num_items = len(self._keys)
        self.reset()

    def reset(self):
        self._idx = 0

    def next(self):
        key = self._keys[self._idx]
        self._idx += 1
        if self._idx > len(self._keys):
            self.reset()
        return key, self._obj.get(key)


class ChordSelector(BaseSelector):

    def __init__(self, obj, length, mapper=None, prior=None, index=None):
        """Create a uniform entity selector.

        Parameters
        ----------
        obj: dict-like object
            Must have a keys() method.
        length: Size of cqt sample to extract.
            Chord sampler must live inside the selector...
        mapper: func, or callable obj
            Consumes an entity, returns a hashable type (str, int)
        prior: array_like, shape=(num_classes,)
            Sampling prior for the selector. Must match the number of unique
            classes returned by 'mapper', or contained in 'index'. Defaults to
            uniform.
        index: dict
            Object contained the class-wise whereabouts of labeled data in the
            source object. Can be provided as an alternative to the mapper for
            efficiency concerns.
        """
        assert mapper or index, "Either mapper or index must be provided."
        # Maintain a handle just in case
        self._obj = obj
        self._keys = obj.keys()

        if index is None:
            index = index_chord_entities(obj, mapper)
        self.index = index
        if None in self.index:
            del self.index[None]
        self._class_ids = self.index.keys()
        num_classes = len(self._class_ids)
        if prior is None:
            prior = np.ones(num_classes) / num_classes

        assert len(prior) == num_classes
        prior /= np.sum(prior)
        self.weights = np.concatenate([np.zeros(1), np.cumsum(prior)])
        self.chord_sampler = chord_sample(length)

    def __call__(self, obj):
        def item_gen(obj):
            max_attempts = 20000
            while True:
                class_idx = (self.weights > np.random.rand()).argmax() - 1
                class_id = self._class_ids[class_idx]
                key, idx = choice(self.index[class_id])
                count = 0
                while not key in obj.keys():
                    key, idx = choice(self.index[class_id])
                    count += 1
                    if count > max_attempts:
                        break
                # print class_id, key, idx
                if key in obj.keys():
                    entity = self.chord_sampler(obj.get(key), idx)
                else:
                    entity = None
                yield key, entity

        return item_gen(obj)
