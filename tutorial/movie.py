'''
Created on Oct 29, 2013

@author: ejhumphrey
'''


import time
import numpy as np
import cPickle
import os
import matplotlib
import argparse
import glob
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


def visualize_weights(param_files, framerate, title, output_file):

    fbase = os.path.split(param_files[0])[-1]
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title, artist='Matplotlib',
            comment=fbase)

    writer = FFMpegWriter(fps=framerate, metadata=metadata)

    max_values = dict()
    num_layers = None
    for idx in [0, -1]:
        param_values = cPickle.load(open(param_files[idx]))
        if num_layers is None:
            num_layers = 0
            for k in param_values:
                if "/bias" in k:
                    num_layers += 1
        for layer_idx in range(num_layers):
            key = 'affine%d/weights' % layer_idx
            W = param_values[key]
            if not key in max_values:
                max_values[key] = -1
            max_val = np.abs(W).max()
            if max_val > max_values[key]:
                max_values[key] = max_val

    fig = plt.figure()

    fig.set_tight_layout(True)
    axes = [fig.add_subplot(1, num_layers, idx) for idx in range(1, num_layers + 1)]
    with writer.saving(fig, output_file, 360):
        for param_idx, param_file in enumerate(param_files):
            param_values = cPickle.load(open(param_file))
            for layer_idx in range(num_layers):
                key = 'affine%d/weights' % layer_idx
                W = param_values[key]
                axes[layer_idx].clear()
                axes[layer_idx].imshow(normalize(W),
                                       interpolation='nearest',
                                       aspect='equal',
                                       vmin= -1 * max_values[key],
                                       vmax=max_values[key])
                axes[layer_idx].set_title(key)
                axes[layer_idx].set_xticks([])
                axes[layer_idx].set_yticks([])
                axes[layer_idx].set_ylabel("n_in")
                axes[layer_idx].set_xlabel("n_out")
            writer.grab_frame()
            if (param_idx % framerate) == 0:
                print "[%s] Finished %4d frames." % (time.asctime(), param_idx)


def normalize(x):
    s = np.abs(x).max()
    if s == 0:
        s = 1.0
    return x / s


def draw_weights(param_file, framerate, title, output_file):

    fig = plt.figure()
    fig.set_tight_layout(True)
    axes = [fig.add_subplot(1, 3, idx) for idx in range(1, 4)]
    param_values = cPickle.load(open(param_file))
    for layer_idx in range(3):
        key = 'affine%d/weights' % layer_idx
        W = param_values[key]
        axes[layer_idx].clear()
        axes[layer_idx].imshow(normalize(W),
                               interpolation='nearest',
                               aspect='equal')
        axes[layer_idx].set_title(key)
    plt.show()



def main(args):
    param_files = glob.glob(os.path.join(args.param_dir, "*.params"))
    param_files.sort()
    print "Found %d files." % len(param_files)
    output_file = os.path.join(args.param_dir, "param_movie.mov")
    visualize_weights(param_files[::4],
                      framerate=10.0,
                      title=args.param_dir,
                      output_file=output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="")

    parser.add_argument("param_dir",
                        metavar="param_dir", type=str,
                        help="")


    main(parser.parse_args())
