"""
"""

from sklearn import metrics
import numpy as np

def print_classification_report(name, y_true, y_pred, labels=None):
    hdr = "%s\n%s\n%s\n" % ("-" * len(name), name, "-" * len(name))
    return hdr + "%s\n" % metrics.classification_report(y_true, y_pred, labels)

def print_confusion_matrix(matrix, top_k_confusions=5, labels=None):
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
    average = 0.0
    if labels is None:
        labels = np.arange(len(matrix))
    for i, row in enumerate(matrix):
        true_positives += row[i]
        total = max([row.sum(), 1])
        row *= 100.0 / float(total)
        msg += "%s: %0.2f\t[" % (labels[i], row[i])
        average += row[i]
        row[i] = -1
        idx = row.argsort()[::-1]
        msg += "\t".join(["%s: %0.2f" % (labels[j], row[j]) for j in idx[:top_k_confusions]])
        msg += "]\n"

    header = "Total Precision: %0.3f  |  Average Accuracy: %0.3f" % \
        ((100 * true_positives / total_count), (average / float(i + 1)))
    msg += "\n%s\n%s\n%s\n" % ("-" * len(header),
                               header,
                               "-" * len(header))
    return msg

def classification_report(y_true, y_pred, labels=None, target_names=None):
    """Build a text report showing the main classification metrics

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets

    y_pred : array, shape = [n_samples]
        Estimated targets

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report

    target_names : list of strings
        Optional display names matching the labels (same order)

    Returns
    -------
    report : string
        Text summary of the precision, recall, f1-score for each class

    """

    if labels is None:
        labels = metrics.metrics.unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    last_line_heading = 'avg / total'

    if target_names is None:
        width = len(last_line_heading)
        target_names = ['%d' % l for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = metrics.metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=labels)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["%0.2f" % float(v)]
        values += ["%d" % int(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["%0.4f" % float(v)]
    values += ['%d' % np.sum(s)]
    report += fmt % tuple(values)
    return report
