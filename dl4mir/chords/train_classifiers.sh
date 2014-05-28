#!/bin/bash
#
# Train a set of end-to-end classifiers and sweep over the checkpointed
#    parameters to identify the early stopping point.

BASEDIR=/media/attic/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/ejhumphrey/dl4mir

# Directory of optimus data files, divided by index and split, like
#   ${OPTFILES}/${FOLD}/${SPLIT}.hdf5
OPTFILES=${BASEDIR}/optfiles

CONFIGS=${BASEDIR}/configs
MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs

TRIAL_NAME="test123"
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
    DRIVER="classifier-L05-V157 "\
# "classifier-L10-V157 "\
# "classifier-L20-V157 "\
# "classifier-L40-V157 "\
"classifier-L80-V157"
    exit 0
else
    DRIVER="$1"
fi

if [ "$2" == "all"] || [ -z "$2" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$2
fi

# Train networks
for drv in ${DRIVER}
do
    for idx in ${FOLD_IDXS}
    do
        python ${SRC}/chords/drivers/${drv}.py \
${OPTFILES}/${idx}/train.hdf5 \
${MODELS}/${drv}/${idx} \
${TRIAL_NAME} \
${VALIDATOR_NAME}.json \
${TRANSFORM_NAME}.json
    done
done

# Model Selection
for drv in ${DRIVER}
do
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting parameters."
        python ${SRC}/common/collect_files.py \
${MODELS}/${drv}/${idx}/${TRIAL_NAME} \
"*.npz" \
${MODELS}/${drv}/${idx}/${TRIAL_NAME}/${PARAM_TEXTLIST}

        python ${SRC}/chords/select_params.py \
${OPTFILES}/${idx}/valid.hdf5 \
${MODELS}/${drv}/${idx}/${TRIAL_NAME}/${VALIDATOR_NAME}.json \
${MODELS}/${drv}/${idx}/${TRIAL_NAME}/${PARAM_TEXTLIST} \
${MODELS}/${drv}/${idx}/${TRIAL_NAME}/${TRANSFORM_NAME}.npz
    done
done

# Transform data
for drv in ${DRIVER}
do
    for idx in ${FOLD_IDXS}
    do
        for split in valid train test
        do
            echo "Transforming ${OPTFILES}/${idx}/${split}.hdf5"
            python ${SRC}/chords/transform_data.py \
${OPTFILES}/${idx}/${split}.hdf5 \
${MODELS}/${drv}/${idx}/${TRIAL_NAME}/${TRANSFORM_NAME}.json \
${MODELS}/${drv}/${idx}/${TRIAL_NAME}/${TRANSFORM_NAME}.npz \
${OUTPUTS}/${drv}/${idx}/${TRIAL_NAME}/${split}.hdf5
        done
    done
done
