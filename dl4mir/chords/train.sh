#!/bin/bash
BASEDIR=/media/attic/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/ejhumphrey/dl4mir

# Directory of optimus data files, divided by index and split, like
#   ${OPTFILES}/${FOLD}/${SPLIT}.hdf5
OPTFILES=${BASEDIR}/optfiles

CONFIGS=${BASEDIR}/configs
MODELS=${BASEDIR}/models

TRIAL_NAME="test123"
VALIDATOR_NAME="validator"
TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"


if [ -z "$1" ]; then
    echo "Usage:"
    echo "train_classifier.sh {L}"
    # echo $'\tV - Vocabulary'
    echo $'\tL - Length of the input window'
    exit 0
fi

if [ "$1" == "all" ]
then
    echo "Setting all drivers"
    DRIVER="classifier-L05-V157 "\
# "classifier-L10-V157 "\
# "classifier-L20-V157 "\
# "classifier-L40-V157 "\
"classifier-L80-V157"
    exit 0
else
    DRIVER="classifier-L$1-V157"
fi

for (( idx=0; idx< 1; idx++ ))
do
    for drv in DRIVER
    do
        python ${SRC}/chords/drivers/${DRIVER}.py \
${OPTFILES}/${idx}/train.hdf5 \
${MODELS}/${DRIVER}/${idx} \
${TRIAL_NAME} \
${VALIDATOR_NAME}.json \
${TRANSFORM_NAME}.json

        echo "Collecting parameters."
        python ${SRC}/common/collect_files.py \
${MODELS}/${DRIVER}/${idx}/${TRIAL_NAME} \
"*.npz" \
${MODELS}/${DRIVER}/${idx}/${TRIAL_NAME}/${PARAM_TEXTLIST}

        python ${SRC}/chords/select_params.py \
${OPTFILES}/${idx}/valid.hdf5 \
${MODELS}/${DRIVER}/${idx}/${TRIAL_NAME}/${VALIDATOR_NAME}.json \
${MODELS}/${DRIVER}/${idx}/${TRIAL_NAME}/${PARAM_TEXTLIST} \
${MODELS}/${DRIVER}/${idx}/${TRIAL_NAME}/${TRANSFORM_NAME}.npz
    done
done
