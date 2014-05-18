
from .core.nodes import Affine
from .core.nodes import Conv3D
from .core.nodes import Softmax
from .core.nodes import LpDistance
from .core.graphs import Graph
from .core.losses import NegativeLogLikelihood
from .core.losses import ContrastiveDivergence
from .core.losses import L2Norm
from .core.losses import L1Norm
from .core.losses import Accumulator
from .core.updates import SGD
from .core.inputs import DataServer
from .core.inputs import Constant
from .core.inputs import Variable
from .driver import Driver


TYPE = 'type'


def Factory(args):
    """Node factory; uses 'type' in the node_args dictionary."""
    local_args = dict(args)
    obj_type = local_args.pop(TYPE)
    return eval("%s(**local_args)" % obj_type)
