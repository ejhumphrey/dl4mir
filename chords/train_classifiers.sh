#!/bin/bash
#
# Train a set of end-to-end classifiers and sweep over the checkpointed
#    parameters to identify the early stopping point.

BASEDIR=/media/attic/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/dl4mir

# Directory of optimus data files, divided by index and split, like
#   ${OPTFILES}/${FOLD}/${SPLIT}.hdf5
OPTFILES=${BASEDIR}/biggie/chord_dsets

MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs

VALIDATOR_NAME="validator"
TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"


if [ -z "$1" ]; then
    echo "Usage:"
    echo "train.sh {driver|all} {[0-4]|*all}"
    echo $'\tdriver - Name of the training driver.'
    echo $'\tfold# - Number of the training fold, default=all.'
    exit 0
fi

if [ "$1" == "all" ]
then
    echo "Setting all known drivers..."
    DRIVER="chroma-L05 "\
"tonnetz-L05 "\
"classifier-L05-V157 "\
"classifier-L10-V157 "\
"classifier-L20-V157 "\
"classifier-L40-V157 "\
"classifier-L80-V157"
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

# Fit networks
if [ $PHASE == "all" ] || [ $PHASE == "fit" ];
then
    for drv in ${DRIVER}
    do
        for idx in ${FOLD_IDXS}
        do
            python ${SRC}/chords/drivers/${drv}.py \
${OPTFILES}/${idx}/train.hdf5 \
${MODELS}/${drv}/${TRIAL_NAME}/${idx} \
${TRIAL_NAME} \
${VALIDATOR_NAME}.json \
${TRANSFORM_NAME}.json
        done
    done
fi

# Model Selection
if [ $PHASE == "all" ] || [ $PHASE == "select" ];
then
    for drv in ${DRIVER}
    do
        for idx in ${FOLD_IDXS}
        do
            echo "Collecting parameters."
            python ${SRC}/common/collect_files.py \
${MODELS}/${drv}/${TRIAL_NAME}/${idx}/ \
"*.npz" \
${MODELS}/${drv}/${TRIAL_NAME}/${idx}/${PARAM_TEXTLIST}

            python ${SRC}/chords/select_params.py \
${OPTFILES}/${idx}/valid.hdf5 \
${MODELS}/${drv}/${TRIAL_NAME}/${idx}/${VALIDATOR_NAME}.json \
${MODELS}/${drv}/${TRIAL_NAME}/${idx}/${PARAM_TEXTLIST} \
${MODELS}/${drv}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.npz
        done
    done
fi

# Transform data
if [ $PHASE == "all" ] || [ $PHASE == "transform" ];
then
    for drv in ${DRIVER}
    do
        for idx in ${FOLD_IDXS}
        do
            for split in valid train test
            do
                echo "Transforming ${OPTFILES}/${idx}/${split}.hdf5"
                python ${SRC}/common/convolve_graph_with_dset.py \
${OPTFILES}/${idx}/${split}.hdf5 \
${MODELS}/${drv}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${drv}/${TRIAL_NAME}/${idx}/${TRANSFORM_NAME}.npz \
${OUTPUTS}/${drv}/${TRIAL_NAME}/${idx}/${split}.hdf5
            done
        done
    done
fi
