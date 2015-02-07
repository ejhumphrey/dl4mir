#!/bin/bash
#
# Train a set of end-to-end classifiers and sweep over the checkpointed
#    parameters to identify the early stopping point.
#
# Requires the following:
#    - An environment variable `DL4MIR` has been set, pointing to the expected
#      directory structure of data.
#    - The script is called from the directory containing the top-level
#      `dl4mir` source code directory.

BASEDIR=${DL4MIR}/guitar
SRC=./dl4mir

# Directory of optimus data files, divided by index and split, like
#   ${BIGGIE}/${FOLD}/${SPLIT}.hdf5
ESTIMATIONS=${BASEDIR}/estimations
META=${BASEDIR}/metadata
MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs
REFERENCES=${DL4MIR}/chord_estimation/references
RESULTS=${BASEDIR}/results

TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"
VALIDATION_CONFIG="validation_config.json"
VALIDATION_PARAMS="validation_params.json"


ARCH_SIZE="$1"
DROPOUT="$2"

if [ -z "$3" ] || [ "$3" == "all" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$3
fi

CONFIG="${ARCH_SIZE}/${DROPOUT}"


for idx in ${FOLD_IDXS}
do
    for split in valid
    do
        for partition in all strict others
        do
            echo ${CONFIG}/${idx}/${split}/${partition}
            python ${SRC}/chords/score_jamsets.py \
${REFERENCES}_${partition}.jamset \
${ESTIMATIONS}/${CONFIG}/${idx}/${split}/chords.jamset \
${RESULTS}/${CONFIG}/${idx}/${split}/chords/${partition}.json
        done
    done
done

# for idx in ${FOLD_IDXS}
# do
#     for split in test valid train
#     do
#         for partition in all strict others
#         do
#             echo ${CONFIG}/${idx}/${split}/${partition}
#             python ${SRC}/chords/score_jamsets.py \
# ${REFERENCES}_${partition}.jamset \
# ${ESTIMATIONS}/${CONFIG}/${idx}/${split}/best.jamset \
# ${RESULTS}/${CONFIG}/${idx}/${split}/${partition}.json
#         done
#     done
# done

