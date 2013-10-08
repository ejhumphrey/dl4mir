#!/usr/bin/env python
"""
"""


import os


def main():

    # name
    # def
    # config
    base_cmd = """ipython ejhumphrey/dnn/trainers/chordrec.py \
%s \
%s \
%s \
/media/attic/chords/models \
/home/ejhumphrey/chords/chordrec_lcn_train0_wBB_20131007.dsf \
/home/ejhumphrey/chords/MIREX09_chord_map_wBB.txt"""

    model_names = ["tanh_nodropout_lcn_wBB_00",
                   "relu_nodropout_lcn_wBB_00",
                   "tanh_dropout_lcn_wBB_00",
                   "relu_dropout_lcn_wBB_00", ]

    definition_files = ["/media/attic/chords/defs/base_model.definition",
                        "/media/attic/chords/defs/base_relu_model.definition",
                        "/media/attic/chords/defs/dropout_model.definition",
                        "/media/attic/chords/defs/dropout_relu_model.definition"]

    config_files = ["/media/attic/chords/default.config",
                    "/media/attic/chords/default.config",
                    "/media/attic/chords/dropout.config",
                    "/media/attic/chords/dropout.config"]
    for m, d, c in zip(model_names, definition_files, config_files):
        os.system(base_cmd % (m, d, c))

if __name__ == '__main__':
    main()
