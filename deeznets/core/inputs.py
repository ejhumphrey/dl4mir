"""
write me.
"""


class DataServer(object):
    """write me."""
    def __init__(self, sources):
        """Sources is a list of Input Objects"""
        self._sources = sources

    # Unnecessary?
    # def keys(self):
    #     """No?"""
    #     return [s.name for s in self._sources]

    def items(self):
        """No?"""
        return dict([(s.name, s.value()) for s in self._sources])

    def update(self):
        """write me."""
        for s in self._sources:
            s.update()

    def next(self):
        """write me."""
        self.update()
        return self.items()

    def __iter__(self):
        return self


class Input(object):
    """write me."""
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """write me."""
        return self._name

    def update(self):
        """write me."""
        pass

    def value(self):
        """write me."""
        raise NotImplementedError("Subclass me!!")


class Constant(Input):
    """write me."""
    def __init__(self, name, value):
        Input.__init__(self, name)
        self._value = value

    def value(self):
        """write me."""
        return self._value


# class ScalarFunction(object):
#     """write me."""
#     def __init__(self, name, fx, init_val=0, step=1):
#         self._fx = fx
#         self._x = init_val
#         self._step = step
#         self._name = name

#     @property
#     def name(self):
#         """write me."""
#         return self._name

#     def __call__(self):
#         return self._fx(self._x)

#     def update(self):
#         """write me."""
#         self._x += self._step


class Variable(Input):
    """write me."""
    def __init__(self, name, value_callback, update_callback=None):
        Input.__init__(self, name)
        self._value_callback = value_callback
        self._update_callback = update_callback

    def value(self):
        """write me."""
        return self._value_callback()

    def update(self):
        """write me."""
        if self._update_callback:
            self._update_callback()
