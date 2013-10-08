'''
Created on Oct 7, 2013

@author: ejhumphrey
'''
import json
from ejhumphrey.covers.processes import create_process


class Transform(object):

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def from_config(self, config_file):
        """Consume a json filepath, and configure this object completely.
        """
        pipeline = [create_process(a) for a in json.load(open(config_file))]
        return Transform(pipeline)

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
        for fx in self.pipeline:
            x = fx(x)
        return x



