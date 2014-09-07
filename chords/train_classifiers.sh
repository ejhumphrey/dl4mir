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

MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs

TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"


if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage:"
    echo "train.sh {model} {source} {[0-4]|*all} {trial} {fit|select|transform|*all}"
    echo $'\tdriver - Name of the training driver.'
    echo $'\tfold# - Number of the training fold, default=all.'
    echo $'\tphase - Name of training phase, default=all.'
    exit 0
else
    MODEL_NAME="$1"
    DATA_SOURCE="$2"
fi

if [ "$3" == "all" ] || [ -z "$3" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$3
fi

if [ -z "$4" ]
then
    TRIAL_NAME="deleteme"
else
    TRIAL_NAME=$4
fi

if [ -z "$5" ]
then
    PHASE="all"
else
    PHASE="$5"
fi


# Fit networks
if [ $PHASE == "all" ] || [ $PHASE == "fit" ];
then
    for idx in ${FOLD_IDXS}
    do
        python ${SRC}/chords/drivers/single_source.py \
${BIGGIE}/${DATA_SOURCE}/${idx}/train.hdf5 \
${MODEL_NAME} \
${MODELS}/${MODEL_NAME}/${DATA_SOURCE}/${idx}/${TRIAL_NAME} \
${TRIAL_NAME} \
${TRANSFORM_NAME}.json
# --init_param_file=${DL4MIR}/chord_estimation/models/cqt_nll_noreg_single/take_00/0/transform.npz
# --secondary_source=${BASEDIR}/biggie/synth/${idx}/train${BS}.hdf5
    done
fi

# Model Selection
if [ $PHASE == "all" ] || [ $PHASE == "select" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting parameters."
        python ${SRC}/common/collect_files.py \
${MODELS}/${MODEL_NAME}/${TRIAL_NAME}/${idx}/ \
"*.npz" \
${MODELS}/${MODEL_NAME}/${TRIAL_NAME}/${idx}/${PARAM_TEXTLIST}

        python ${SRC}/chords/select_classification_params.py \
${BIGGIE}/${DATA_SOURCE}/${idx}/valid${BS}.hdf5 \
${MODELS}/${MODEL_NAME}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${TRANSFORM_NAME}.json \
${MODELS}/${MODEL_NAME}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${PARAM_TEXTLIST} \
${MODELS}/${MODEL_NAME}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${TRANSFORM_NAME}.npz \
--num_obs=100 \
--start_idx=200
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
${BIGGIE}/${DATA_SOURCE}/${idx}/${split}${BS}.hdf5 \
${MODELS}/${MODEL_NAME}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${TRANSFORM_NAME}.json \
${MODELS}/${MODEL_NAME}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${TRANSFORM_NAME}.npz \
${OUTPUTS}/${MODEL_NAME}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${split}.hdf5
        done
    done
fi
