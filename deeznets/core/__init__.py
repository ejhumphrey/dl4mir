import theano
import theano.tensor as T

FLOATX = theano.config.floatX

TENSOR_TYPES = {None: T.scalar,
                0: T.vector,
                1: T.matrix,
                2: T.tensor3,
                3: T.tensor4}
