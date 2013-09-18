"""
"""

import argparse
import os
import time

from multiprocessing import Pool

from marl.audio.sox import convert

NUM_CPUS = 1  # Use None for system max.


def parse_line(line):
    """Return the billboard_id and youtube_id from the line."""
    parts = line.strip("\n").split(" ")
    billboard_id, youtube_id = parts[0], None
    if len(parts) == 4:
        youtube_id = parts[-1]
    return billboard_id, youtube_id


def download_audio(input_data):
    """

    """
    youtube_id, billboard_id, output_dir = input_data
    output_template = "'%s'" % os.path.join(output_dir, "temp_%(id)s.%(ext)s")
    output_temp = os.path.join(output_dir, "temp_%s.wav" % youtube_id)
    output_file = os.path.join(output_dir, "%s.mp3" % billboard_id)
    if os.path.exists(output_file):
        print "[%s] Skipping: %s" % (time.asctime(), billboard_id)
        return
    base_cmd = "youtube-dl -x --audio-format 'wav' -o"
    command = base_cmd + " %s \"http://www.youtube.com/watch?v=%s\"" % (output_template, youtube_id)
    print command
    try:
        status = os.system(command)
    except:
        raise BaseException("Exited with status: %s" % status)

    convert(output_temp, output_file)
    os.remove(output_temp)
    print "[%s] Finished: %s" % (time.asctime(), billboard_id)


def iofiles(file_list, output_dir):
    """Generator for input/output file pairs.

    Parameters
    ----------
    file_list : str
        Path to a text file of space separated billboard_ids and youtube_ids.

    Yields
    ------
    file_pair : Pair of strings
        first=youtube_id, second=output_file
    """
    for line in open(file_list, "r"):
        billboard_id, youtube_id = parse_line(line)
        if youtube_id:
            yield (youtube_id, billboard_id, output_dir)


def main(args):
    """ . """
    pool = Pool(processes=NUM_CPUS)
    pool.map_async(func=download_audio,
                   iterable=iofiles(args.file_list, args.output_directory))
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a collection of YouTube IDs to file.")
    parser.add_argument("file_list",
                        metavar="file_list", type=str,
                        help="A text file produced from DPWE's yt-scrape.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Base path to save sound files.")
    main(parser.parse_args())
