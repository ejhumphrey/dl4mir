"""
"""

import argparse
import json
import marl.fileutils as futil
import glob
from pyechonest import song as S
from pyechonest import config as C
from os import path
import socket
import time

C.CODEGEN_BINARY_OVERRIDE = '/usr/local/bin/codegen.Darwin'


class Throttle(object):
    def __init__(self, delay=5.5):
        self.last_check = time.time()
        self.delay = delay

    def touch(self):
        self.last_check = time.time()

    def is_ready(self, delay=None):
        if delay is None:
            delay = self.delay
        return time.time() - delay > self.last_check

    def wait(self, delay=None):
        if delay:
            self.touch()
        while not self.is_ready(delay):
            pass
        self.touch()


def extract_info(song):
    data = dict()
    for k, v in song.__dict__.items():
        if k.startswith("_"):
            continue
        data[k] = v

    return data


def fetch_data(filepaths, result=None, overwrite=False, checkpoint_file=''):
    """
    Parameters
    ----------
    filepaths : list
        Collection of audio files on disk to query against the EchoNest API.
    result : dict, or None
        Dictionary to add info; will create if None.
    overwrite : bool, default=False
        If False, will skip any keys contained in `result`.
    checkpoint_file : str, or None
        Path to write results as they are accumulated; ignored if empty.

    Returns
    -------
    result : dict
        Map of filebases to metadata.
    """
    throttle = Throttle()
    throttle.touch()
    if result is None:
        result = dict()

    filepaths = set(filepaths)
    while filepaths:
        fpath = filepaths.pop()
        key = futil.filebase(fpath)
        # If we've already got data and we're not overwriting, move on.
        if key in result and not overwrite:
            print "[%s] %4d: '%s'" % (time.asctime(), len(filepaths), key)
            continue
        try:
            # Otherwise, let's make some requests.
            print "[%s] %4d: '%s'" % (time.asctime(), len(filepaths), key)
            song = S.identify(
                filename=fpath, codegen_start=0, codegen_duration=60)
            if song:
                result[key] = extract_info(song[0])
            if checkpoint_file:
                with open(checkpoint_file, 'w') as fp:
                    json.dump(result, fp, indent=2)
            throttle.wait()
        except S.util.EchoNestAPIError as err:
            if err.http_status == 429:
                print "You got rate limited braah ... hang on."
                throttle.wait(10)
                filepaths.add(fpath)
            elif err.http_status >= 500:
                print "Server error; moving on, dropping key: %s" % key
        except socket.error as err:
            print "Socket Error %s" % err
            filepaths.add(fpath)
            throttle.wait(10)
    return result


def main(args):
    filepaths = glob.glob(
        path.join(args.audio_directory, "*.%s" % args.ext.strip(".")))
    if path.exists(args.output_file):
        result = json.load(open(args.output_file))
        print "File exists: Found %d results" % len(result)
    else:
        futil.create_directory(path.split(args.output_file)[0])
        result = dict()

    result = fetch_data(
        filepaths, result=result, overwrite=False,
        checkpoint_file=args.output_file)

    with open(args.output_file, 'w') as fp:
        json.dump(result, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a directory of labeled interval files to JAMS.")
    parser.add_argument("audio_directory",
                        metavar="audio_directory", type=str,
                        help="Path to a directory of audio files.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the output metadata.")
    parser.add_argument("--ext",
                        metavar="--ext", type=str, default='mp3',
                        help="File extension to glob against.")
    parser.add_argument("--overwrite",
                        metavar="--overwrite", type=bool, default=False,
                        help="Overwrite any existing results.")
    main(parser.parse_args())
