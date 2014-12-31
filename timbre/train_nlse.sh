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

BASEDIR=${DL4MIR}/timbre_sim
SRC=./dl4mir

# Directory of optimus data files, divided by index and split, like
#   ${BIGGIE}/${FOLD}/${SPLIT}.hdf5
BIGGIE=${BASEDIR}/biggie
INITS=${BASEDIR}/param_inits
MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs

TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"


if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage:"
    echo "train.sh {config} {margin} {[0-4]|*all} {fit|select|transform|*all}"
    echo $'\tconfig - Instrument configuration, one of {c5, c8, c12, c24}.'
    echo $'\tmargin - Margin to use for training.'
    echo $'\tfold# - Number of the training fold, default=all.'
    echo $'\tphase - Name of training phase, default=all.'
    exit 0
fi

CONFIG="$1"
MARGIN="$2"

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

TRIAL="${CONFIG}/${MARGIN}"

# Fit networks
if [ $PHASE == "all" ] || [ $PHASE == "fit" ];
then
    for idx in ${FOLD_IDXS}
    do
        python ${SRC}/timbre/driver.py \
${BIGGIE}/${CONFIG}/${idx}/train.hdf5 \
${MARGIN} \
${MODELS}/${TRIAL}/${idx} \
"nlse" \
${TRANSFORM_NAME}.json
    done
fi

# Model Selection
if [ $PHASE == "all" ] || [ $PHASE == "select" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting parameters."
        python ${SRC}/common/collect_files.py \
${MODELS}/${TRIAL}/${idx} \
"nlse*.npz" \
${MODELS}/${TRIAL}/${idx}/${PARAM_TEXTLIST}

        python ${SRC}/timbre/final_params.py \
${MODELS}/${TRIAL}/${idx}/${PARAM_TEXTLIST} \
${MODELS}/${TRIAL}/${idx}/${TRANSFORM_NAME}.npz
    done
fi

# Transform data
if [ $PHASE == "all" ] || [ $PHASE == "transform" ];
then
    for idx in ${FOLD_IDXS}
    do
        for split in valid test train
        do
            echo "Transforming ${BIGGIE}/${CONFIG}/${idx}/${split}.hdf5"
            python ${SRC}/common/transform_stash.py \
${BIGGIE}/${CONFIG}/${idx}/${split}.hdf5 \
"cqt" \
${MODELS}/${TRIAL}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${TRIAL}/${idx}/${TRANSFORM_NAME}.npz \
${OUTPUTS}/${TRIAL}/${idx}/${split}.hdf5
        done
    done
fi
