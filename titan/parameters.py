"""
"""

import numpy as np
import theano

ALLOW_DOWNCAST = True

class Parameter(dict):
    """Wrapper around symbolic variables."""
    def __init__(self, name=None, value=None, shape=None):
        """
        Parameters
        ----------
        id: Name of this parameter.
        value: Value to give this parameter.
        shape: Tuple to allocate for this parameter.  
        """
        assert value is None or shape is None, \
            "Only one of value or shape may be specified."
        if value is None:
            value = np.zeros(shape)
        self._variable = theano.shared(value=value,
                                       name=name,
                                       allow_downcast=ALLOW_DOWNCAST)
        self.update({ "name":self.name, "shape":self.shape, })

    @property
    def name(self):
        return self.variable.name

    @property
    def shape(self):
        """Return a tuple of this parameter's anticipated shape."""
        return self.value.shape

    @property
    def value(self):
        """The numerical coefficients of this parameter."""
        return self.variable.get_value()

    @value.setter
    def value(self, val):
        assert val.shape == self.shape, \
            "New value must match current shape. Received: %s, Expected %s" % \
            (val.shape, self.shape)
        self._variable.set_value(val)

    @property
    def variable(self):
        """A symbolic variable of this parameter."""
        return self._variable
