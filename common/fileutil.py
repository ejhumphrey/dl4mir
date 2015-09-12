"""Utilities for manipulating files and paths.
"""

import os
from collections import namedtuple
import tempfile as tmp

Pair = namedtuple('Pair', 'first second')


def is_empty(filepath):
    """Determine if file is empty. Return true if empty.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    status : bool
        True if the file exists, but is empty.
    """
    status = False
    if os.path.exists(filepath):
        status = os.stat(filepath)[6] == 0
    return status


def fileext(filepath):
    """Return the extension of a file path."""
    return os.path.splitext(filepath)[-1]


def filebase(filepath):
    """For a full filepath like '/path/to/some/file.xyz', return 'file'.

    Like `os.path.basename`, except it also strips the file extension.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    fbase : str
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def filedir(filepath):
    """For a full filepath like '/path/to/some/file.xyz', return
        /path/to/some/'.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    filedir : str
    """
    return os.path.dirname(filepath)


def expand_filebase(fbase, output_dir, ext):
    """For a filebase 'file', output directory '/path/to/some/',
        and extension 'xyz', return '/path/to/some/file.xyz'

    Parameters
    ----------
    fbase : str
    output_dir : str
    ext : str

    Returns
    -------
    filepath : str
    """
    ext = ext.strip(".")
    return os.path.join(output_dir, "%s.%s" % (fbase, ext))


def map_path_file_to_dir(path_file, output_dir, output_ext):
    """Generator for mapping a file of filepaths to similarly named files in a
    single output directory.

    Parameters
    ----------
    path_file : str
        Path to a human-readable text file of absolute file paths.
    output_dir : str
        Base directory to write outputs under the same file base.

    Yields
    ------
    file_pair : Pair of strings
        first=input_file, second=output_file
    """
    for line in open(path_file, "r"):
        input_file = line.strip("\n")
        output_file = expand_filebase(
            filebase(input_file), output_dir, output_ext)
        yield Pair(input_file, output_file)


def map_files_to_dir(files, output_dir, output_ext):
    """Generator for mapping a list of filepaths to similarly named files in a
    single output directory.

    Parameters
    ----------
    files : list
        List of absolute file paths.
    output_dir : str
        Base directory to write outputs under the same basename.

    Yields
    ------
    file_pair : Pair of strings
        first=input_file, second=output_file
    """
    for input_file in files:
        output_file = expand_filebase(
            filebase(input_file), output_dir, output_ext)
        yield Pair(input_file, output_file)


def create_directory(directory):
    """Create the output directory recursively if it doesn't already exist.

    Returns
    -------
    output_dir : str
        Expanded path, that now certainly exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def load_textlist(filepath):
    """Load a new-line separated list from a text-file.

    Parameters
    ----------
    filepath: str
        Path of the textfile list to load.

    Returns
    -------
    items: list
        List of strings (new-lines are removed).
    """
    return [line.strip("\n") for line in open(filepath)]


def dump_textlist(items, filepath):
    """Dump a list to a new-line separated textfile.

    Note that each item in the list will take its default string
    representation.

    Parameters
    ----------
    items: list
        Items to write to disk.

    filepath: str
        Filepath to write the textfile.
    """
    with open(filepath, 'w') as fh:
        fh.writelines(["%s\n" % item for item in items])


def temp_file(ext):
    """Generate a temporary file path with read/write permissions.

    Parameters
    ----------
    ext: str
        Extension for the temporary file.

    Returns
    -------
    tmpfile : string
        A writeable file path.
    """
    return tmp.mktemp(suffix=".%s" % ext.strip("."), dir=tmp.gettempdir())
