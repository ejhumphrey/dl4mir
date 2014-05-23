#!/bin/bash
# BASEDIR=/media/attic/dl4mir/chord_estimation
BASEDIR=~/dl4mir_test/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/ejhumphrey/dl4mir

# Flat directory of all audio

# Directory of optimus data files, divided by index and split, like
#   ${OPTFILES}/${FOLD}/${SPLIT}.hdf5
OPTFILES=${BASEDIR}/optfiles

CONFIGS=${BASEDIR}/configs
MODELS=${BASEDIR}/models
PREDICTIONS=${BASEDIR}/predictions
RESULTS=${BASEDIR}/results

TRIAL_NAME="testing"
TRANSFORM_NAME="transform.json"

if [ -z "$1" ]; then
    echo "Usage:"
    echo "train.sh {chroma|tonnetz|classifier|tabs|all}"
    echo $'\tchroma - Train a chroma model'
    echo $'\ttonnetz - Train a tonnetz model'
    echo $'\tclassifier - Train a classifier model'
    echo $'\tguitar - Train a guitar model'
    echo $'\tall - Do everything, in order'
    exit 0
fi

if [ "$1" == "all" ]
then
    echo "Setting all drivers"
    DRIVER="chroma tonnetz classifier-L05 classifier-L10 classifier-L20 "\
"classifier-L40 classifier-L80 guitar"
else
    DRIVER="$1"
fi

SPLIT=train
for (( idx=0; idx< 1; idx++ ))
do
    for drv in DRIVER
    do
        python ${SRC}/chords/drivers/${DRIVER}.py \
${OPTFILES}/${idx}/${SPLIT}.hdf5 \
${MODELS}/${DRIVER}/${idx} \
${TRIAL_NAME} \
${TRANSFORM_NAME}
    done
done
