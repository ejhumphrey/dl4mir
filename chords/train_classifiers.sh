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

# BASEDIR=/media/attic/dl4mir/chord_estimation
BASEDIR=${DL4MIR}/chord_estimation
SRC=./dl4mir

# Directory of optimus data files, divided by index and split, like
#   ${BIGGIE}/${FOLD}/${SPLIT}.hdf5
BIGGIE=${BASEDIR}/biggie/chords_l2n
INITS=${BASEDIR}/param_inits
MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs

TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"


if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage:"
    echo "train.sh {arch_size} {dropout} {[0-4]|*all} {fit|select|transform|*all} {}"
    echo $'\tarch_size - Architecture size, one of {L, XL, XXL}.'
    echo $'\tdropout - Dropout hyperparameter.'
    echo $'\tfold# - Number of the training fold, default=all.'
    echo $'\tphase - Name of training phase, default=all.'
    exit 0
fi

ARCH_SIZE="$1"
DROPOUT="$2"

if [ -z "$3" ] || [ "$3" == "all" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$3
fi

if [ -z "$4" ];
then
    PHASE="all"
else
    PHASE=$4
fi

CONFIG="${ARCH_SIZE}/${DROPOUT}"

# Fit networks
if [ $PHASE == "all" ] || [ $PHASE == "fit" ];
then
    for idx in ${FOLD_IDXS}
    do
        python ${SRC}/chords/driver.py \
${BIGGIE}/${idx}/train.hdf5 \
${ARCH_SIZE} \
${DROPOUT} \
${MODELS}/${CONFIG}/${idx} \
"ace_deepnet" \
${TRANSFORM_NAME}.json
    done
fi

# Model Selection
if [ $PHASE == "all" ] || [ $PHASE == "validate" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting parameters."
        python ${SRC}/common/collect_files.py \
${MODELS}/${CONFIG}/${idx} \
"*.npz" \
${MODELS}/${CONFIG}/${idx}/${PARAM_TEXTLIST}

        python ${SRC}/chords/validation_sweep.py \
${BIGGIE}/${idx}/valid.hdf5 \
${MODELS}/${CONFIG}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${CONFIG}/${idx}/${PARAM_TEXTLIST} \
${OUTPUTS}/${CONFIG}/${idx}/valid \
--start_index=9 \
--stride=10
    done
fi

# Transform data
if [ $PHASE == "all" ] || [ $PHASE == "transform" ];
then
    for idx in ${FOLD_IDXS}
    do
        for split in valid test train
        do
            echo "Transforming ${BIGGIE}/${idx}/${split}.hdf5"
            python ${SRC}/common/transform_stash.py \
${BIGGIE}/${idx}/${split}.hdf5 \
"cqt" \
${MODELS}/${CONFIG}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${CONFIG}/${idx}/${TRANSFORM_NAME}.npz \
${OUTPUTS}/${CONFIG}/${idx}/${split}.hdf5
        done
    done
fi
