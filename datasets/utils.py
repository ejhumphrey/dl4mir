"""
"""

import os
import fnmatch

def collect_nested_files(base_dir, extensions):
    """Collect a list of filepaths matching a given extension recursively.

    Parameters
    ----------
    base_dir : str
        Top-level directory to walk.
    extensions : list of strings
        List of file extensions to match.

    Returns
    -------
    matched_files : list of strings
        All filepaths that match the set of extensions.
    """
    matched_files = []
    for root, _dirs, files in os.walk(base_dir):
        # Iterate over valid extensions.
        for ext in extensions:
            # Files that match the current extension.
            these_files = fnmatch.filter(files, "*%s" % ext)
            # Expand the full filepath of all matched files.
            matched_files += [os.path.join(root, f) for f in these_files]

    return matched_files


def join_on_filebase(filepaths):
    """Given a set of files, group into lists by filebase (minus the extension).

    Parameters
    ----------
    filepaths : list
        Collection of string filepaths.

    Returns
    -------
    path_collections : list of lists
    """
    results = dict()
    for filepath in filepaths:
        filebase = os.path.splitext(filepath)[0]
        if not filebase in results:
            results[filebase] = []
        results[filebase].append(filepath)

    return results.values()


