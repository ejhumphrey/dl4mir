import argparse

import time
import numpy as np
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


FIG = plt.figure(figsize=(10, 4))
AX = FIG.gca()
AX.set_axis_off()


def render(data, fps, output_file, title='', dpi=300):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title,
                    artist='Matplotlib',
                    comment='Tensor movie')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Create string highlighters
    with writer.saving(FIG, output_file, dpi):
        im = AX.imshow(
            np.zeros_like(data[0]), vmin=data.min(), vmax=data.max(),
            interpolation='nearest', aspect='auto', origin='lower')
        for frame_num, frame in enumerate(data):
            im.set_array(frame)
            plt.draw()
            writer.grab_frame(pad_inches=0)
            if (frame_num % 100) == 0:
                print "[%s] Finished %4d frames." % (time.asctime(), frame_num)


def main(args):
    render(np.load(args.data_file), args.movie_fps, args.video_file)
    # cmd_fmt = "ffmpeg -i %s -i %s -c copy -map 0:0 -map 1:0 %s"
    # os.system(cmd_fmt % (args.video_file, args.audio_file, args.output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Filepath to a properly formatted npz archive. "
                             "or npy file.")
    parser.add_argument("audio_file",
                        metavar="audio_file", type=str,
                        help="")
    parser.add_argument("video_file",
                        metavar="video_file", type=str,
                        help="")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File path to save output movie.")
    parser.add_argument("movie_fps",
                        metavar="movie_fps", type=float,
                        help="Framerate for the movie.")
    main(parser.parse_args())
