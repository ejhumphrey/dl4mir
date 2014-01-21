
from .core.nodes import Affine
from .core.nodes import Conv3D
from .core.nodes import Softmax
from .core.graphs import Graph
from .core.losses import NegativeLogLikelihood
from .core.losses import L2Norm
from .core.losses import L1Norm
from .core.losses import Accumulator
from .core.updates import SGD

TYPE = 'type'


def Factory(args):
    """Node factory; uses 'type' in the node_args dictionary."""
    local_args = dict(args)
    obj_type = local_args.pop(TYPE)
    return eval("%s(**local_args)" % obj_type)
