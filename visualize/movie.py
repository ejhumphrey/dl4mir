'''
Created on Oct 29, 2013

@author: ejhumphrey
'''


import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


def visualize_tensor(tensor, framerate, title, output_file):
    vid_title = "Santeria"

    output_file = "06-Santeria-frets.mov"

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=vid_title, artist='Matplotlib',
            comment='Fretboard Test!')
    writer = FFMpegWriter(fps=20, metadata=metadata)

    fig = plt.figure()
    ax = fig.gca()
    with writer.saving(fig, output_file, 360):
        for frame_idx, matrix in enumerate(tensor):
            ax.clear()
            ax.imshow(matrix, interpolation='nearest', aspect='auto')
            writer.grab_frame()
            if (frame_idx % 100) == 0:
                print "[%s] Finished %4d frames." % (time.asctime(), frame_idx)


