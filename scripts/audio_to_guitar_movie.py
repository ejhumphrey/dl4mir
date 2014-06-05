"""
"""


import numpy as np

from ejhumphrey.dnn.core.graphs import Network
import tempfile

y0 = np.array([295, 258, 218, 180, 141, 104])
y1 = np.array([310.7, 266.7, 220, 177, 131.2, 86.8])
x = np.array([120, 267, 462, 650, 815, 983, 1130, 1275])

m = (y1 - y0) / (x[-1] - x[0])

y = np.array([mi * (x - x[0]) + yi for mi, yi in zip(m, y0)])
x = np.array([x for n in range(6)])

fret_png = "/Users/ejhumphrey/Dropbox/NYU/2013_03_Fall/chordrec/fender-7.png"


def generate_movie_from_fret_posterior(X, output_file):
    tmpdir = tempfile.gettempdir()

    "avconv -r 10 -i filename_%d.png -b:v 1000k %s" % (output_file)
