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
#   ${OPTFILES}/${FOLD}/${SPLIT}.hdf5
# OPTFILES=${BASEDIR}/biggie/chords
# OPTFILES=${BASEDIR}/biggie/hpss
OPTFILES=${BASEDIR}/biggie/synth_wrap

MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs

VALIDATOR_NAME="validator"
TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"


if [ -z "$1" ]; then
    echo "Usage:"
    echo "train.sh {driver} {[0-4]|*all} {fit|select|transform|*all} {bs}"
    echo $'\tdriver - Name of the training driver.'
    echo $'\tfold# - Number of the training fold, default=all.'
    echo $'\tphase - Name of training phase, default=all.'
    exit 0
else
    DRIVER="$1"
fi

if [ "$2" == "all" ] || [ -z "$2" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$2
fi

if [ -z "$3" ]
then
    TRIAL_NAME="deleteme"
else
    TRIAL_NAME=$3
fi

if [ -z "$4" ]
then
    PHASE="all"
else
    PHASE="$4"
fi

if [ -z "$5" ]
then
    BS=""
else
    BS="_bs"
fi

TRIAL_NAME=${TRIAL_NAME}${BS}

# Fit networks
if [ $PHASE == "all" ] || [ $PHASE == "fit" ];
then
    for idx in ${FOLD_IDXS}
    do
            python ${SRC}/chords/drivers/${DRIVER}.py \
${OPTFILES}/${idx}/train${BS}.hdf5 \
${MODELS}/${DRIVER}/${TRIAL_NAME}/${idx} \
${TRIAL_NAME} \
${VALIDATOR_NAME}.json \
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
${MODELS}/${DRIVER}/${TRIAL_NAME}/${idx}/ \
"*.npz" \
${MODELS}/${DRIVER}/${TRIAL_NAME}/${idx}/${PARAM_TEXTLIST}

        python ${SRC}/chords/select_classification_params.py \
${OPTFILES}/${idx}/valid${BS}.hdf5 \
${MODELS}/${DRIVER}/${TRIAL_NAME}/${idx}/${VALIDATOR_NAME}.json \
${MODELS}/${DRIVER}/${TRIAL_NAME}/${idx}/${PARAM_TEXTLIST} \
${MODELS}/${DRIVER}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.npz
    done
fi

# Transform data
if [ $PHASE == "all" ] || [ $PHASE == "transform" ];
then
    for idx in ${FOLD_IDXS}
    do
        for split in valid test train
        do
            echo "Transforming ${OPTFILES}/${idx}/${split}${BS}.hdf5"
            python ${SRC}/common/convolve_graph_with_dset.py \
${OPTFILES}/${idx}/${split}${BS}.hdf5 \
${MODELS}/${DRIVER}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${DRIVER}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.npz \
${OUTPUTS}/${DRIVER}/${TRIAL_NAME}/${idx}/${split}${BS}.hdf5
        done
    done
fi
