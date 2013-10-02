"""Import a collection of CQT arrays into a Hewey DataSequenceFile.

Sample Call:
$ ipython ejhumphrey/scripts/LMD_OPs_to_dseqfile.py \
/Volumes/Audio/LMD/splits/train00_20131001.txt \
/Volumes/Audio/LMD/LMD_scalars240x10_mid.pkl \
/Volumes/speedy/LMD_train00_20131001.dsf
"""

import argparse
import cPickle

from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen

from ejhumphrey.datasets.lmd import matfile_to_datasequence


def create_datasequence_file(mat_files, filename, stdev_params):
    """
    Parameters
    ----------
    mat_files : list of strings
        Paths to matfiles.
    filename : string
        Output name for the DataSequenceFile
    stdev_params : np.ndarray
        The first dimension is mean, the second standard deviation.
    """
    file_handle = DataSequenceFile(filename)
    keygen = uniform_keygen(2)
    for i, mat_file in enumerate(mat_files):
        print "%03d: Importing %s" % (i, mat_file)
        dseq = matfile_to_datasequence(mat_file, stdev_params)
        key = keygen.next()
        file_handle.write(key, dseq)

    file_handle.create_tables()


def main(args):
    stdev_params = cPickle.load(open(args.stdev_file))
    mat_files = [l.strip('\n') for l in open(args.filelist)]
    create_datasequence_file(mat_files, args.output_file, stdev_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="WRITEME.")

    parser.add_argument("filelist",
                        metavar="filelist", type=str,
                        help="Text file list of matfiles to import.")

    parser.add_argument("stdev_file",
                        metavar="stdev_file", type=str,
                        help="WRITEME.")

    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filepath to write the output DataSequenceFile.")

    main(parser.parse_args())
