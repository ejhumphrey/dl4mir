'''
Created on Oct 7, 2013

@author: ejhumphrey
'''


class Transform(object):

    def __init__(self, config_file):
        """Consume a json filepath, and configure this object completely.
        """
        self._config_file = config_file

    def __call__(self, x):
        """Transform data.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        z : np.ndarray
            Output data.
        """
        return x

    def save(self, config_file):
        """Write configuration data to disk.
        """
        pass
