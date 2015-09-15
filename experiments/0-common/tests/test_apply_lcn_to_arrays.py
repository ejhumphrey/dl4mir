import numpy as np
from dl4mir.common import fileutil as F

from dl4mir.common import apply_lcn_to_arrays as LCN


def test_base_call():
    textlist = F.TempFile('.txt')
    params = F.TempFile('.json')
    npz = F.TempFile('.npz')
    cqt = np.random.uniform(size=(50, 252))
    np.savez(npz.path, cqt=cqt)
    F.dump_textlist([npz.path], textlist.path)

    output_dir = F.TempDir()
    assert LCN.main(textlist.path, 5, 5, output_dir.path, params.path)
