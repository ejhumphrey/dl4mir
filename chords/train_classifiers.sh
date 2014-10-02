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
BIGGIE=${BASEDIR}/biggie
INITS=${BASEDIR}/param_inits
MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs

TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"


if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage:"
    echo "train.sh {driver} {model_name} {data} {[0-4]|all} {fit|select|transform|all} {}"
    echo $'\tdriver - Name of the training driver.'
    echo $'\tfold# - Number of the training fold, default=all.'
    echo $'\tphase - Name of training phase, default=all.'
    exit 0
fi

DRIVER="$1"
MODEL_NAME="$2"
DATA_SOURCE="$3"
TRIAL_NAME="${DRIVER}-${MODEL_NAME}-${DATA_SOURCE}"

if [ "$4" == "all" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$4
fi

PHASE="$5"

if [ -z "$6"];
then
    INIT_FILE=""
else
    INIT_FILE=$6
fi

# Fit networks
if [ $PHASE == "all" ] || [ $PHASE == "fit" ];
then
    for idx in ${FOLD_IDXS}
    do
        python ${SRC}/chords/drivers/${DRIVER}.py \
${BIGGIE}/${DATA_SOURCE}/${idx}/train.hdf5 \
${MODEL_NAME} \
${MODELS}/${TRIAL_NAME}/${idx}/ \
${TRIAL_NAME} \
${TRANSFORM_NAME}.json
# --init_param_file=${INITS}/$6.npz
    done
fi

# Model Selection
if [ $PHASE == "all" ] || [ $PHASE == "select" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting parameters."
        python ${SRC}/common/collect_files.py \
${MODELS}/${TRIAL_NAME}/${idx}/ \
"*.npz" \
${MODELS}/${TRIAL_NAME}/${idx}/${PARAM_TEXTLIST}

        python ${SRC}/chords/find_best_params.py \
${BIGGIE}/${DATA_SOURCE}/${idx}/valid.hdf5 \
${MODELS}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${TRIAL_NAME}/${idx}/${PARAM_TEXTLIST} \
${MODELS}/${TRIAL_NAME}/${idx}/validation_stats.json
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
${BIGGIE}/${DATA_SOURCE}/${idx}/${split}.hdf5 \
"cqt" \
${MODELS}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.npz \
${OUTPUTS}/${TRIAL_NAME}/${idx}/${split}.hdf5
        done
    done
fi
