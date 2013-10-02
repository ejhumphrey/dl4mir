"""
"""

from sklearn import metrics

def print_classification_report(name, y_true, y_pred):
    hdr = "%s\n%s\n%s\n" % ("-" * len(name), name, "-" * len(name))
    return hdr + "%s\n" % metrics.classification_report(y_true, y_pred)

def print_confusion_matrix(matrix, top_k_confusions=5):
    """Print the top-k confusions per class.

    Parameters
    ----------
    matrix : np.ndarray
        Confusion matrix.
    top_k_confusions : int
        Number of confusions to print.

    Returns
    -------
    results : str
        Formatted string results.
    """
    header = "Confusions"
    msg = "\n%s\n%s\n%s\n" % ("-" * len(header),
                              header,
                              "-" * len(header))
    true_positives = 0
    matrix = matrix.astype(float).copy()
    total_count = matrix.sum()
    for i, row in enumerate(matrix):
        true_positives += row[i]
        total = max([row.sum(), 1])
        row *= 100.0 / float(total)
        msg += "%3d: %0.2f\t[" % (i, row[i])
        row[i] = -1
        idx = row.argsort()[::-1]
        msg += ", ".join(["%3d: %0.2f" % (j, row[j]) for j in idx[:top_k_confusions]])
        msg += "]\n"

    header = "Total Precision: %0.3f" % (100 * true_positives / total_count)
    msg += "\n%s\n%s\n%s\n" % ("-" * len(header),
                              header,
                              "-" * len(header))
    return msg
