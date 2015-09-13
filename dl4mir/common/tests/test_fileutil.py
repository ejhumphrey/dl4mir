import os

import dl4mir.common.fileutil as F


def test_TempFile():
    tmp = F.TempFile("txt")
    expected = "Accomplishments are transient."
    with open(tmp.path, 'w') as fh:
        fh.write(expected)
    with open(tmp.path) as fh:
        actual = fh.read()
    assert actual == expected

    fpath = tmp.path
    assert os.path.exists(fpath)
    tmp.close()
    assert not os.path.exists(fpath)


def test_TempDir():
    tmp = F.TempDir()
    fpath = os.path.join(tmp.path, "my_file.txt")
    expected = "Accomplishments are transient."
    with open(fpath, 'w') as fh:
        fh.write(expected)
    dpath = tmp.path

    assert os.path.exists(fpath)
    assert os.path.exists(dpath)

    tmp.close()
    assert not os.path.exists(fpath)
    assert not os.path.exists(dpath)
