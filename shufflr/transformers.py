"""
"""


class Transformer(object):
    def __init__(self):
        pass

    def transform(self, x):
        raise NotImplementedError("Write me! Base class does nothing.")


class LabelMap(Transformer):

    def __init__(self, label_maps, default=-1):
        """
        """
        self.label_maps = label_maps.copy()
        self.default = default

    def transform(self, data):
        """write me"""
        new_labels = dict()
        for key in self.label_maps:
            new_labels[key] = [self.label_maps[key].get(l, self.default)
                               for l in data.labels[key]]
        return data.Type(name=data.name,
                         value=data.value,
                         labels=new_labels,
                         targets=data.targets,
                         metadata=data.metadata)
