#!/bin/bash
#
# Run the evaluation pipeline over a set of posteriors.

BASEDIR=${DL4MIR}/chord_estimation
SRC=dl4mir

# Flat directory of all audio
LABS=${BASEDIR}/labs
META=${BASEDIR}/metadata
# Directory of biggie datasets, divided by index and split, like
#   ${DSETS}/${FOLD_IDX}/${SPLIT_NAME}.hdf5
DSETS=${BASEDIR}/biggie/chord_dsets
OUTPUTS=${BASEDIR}/outputs
ESTIMATIONS=${BASEDIR}/estimations
RESULTS=${BASEDIR}/results
NUM_FOLDS=5

SPLIT_FILE=${META}/data_splits.json

if [ -z "$3" ]; then
    echo "Usage:"
    echo "evaluate.sh driver trial_name {fold|all} {aggregate|score|all}"
    echo $'\taggregate - Collects output classes over posteriors'
    echo $'\tscore - Scores flattened predictions'
    echo $'\tall - Do everything, in order'
    exit 0
fi

DRIVER=$1
DATA_SOURCE=$2
TRIAL_NAME=$3

if [ -z "$4" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$4
fi

if [ -z "$5" ]
then
    PREDICTION_FX=""
else
    PREDICTION_FX="$5"
    PRED_TAG=_"$5"
fi

if [ -z "$6" ]
then
    MODE="all"
else
    MODE="$6"
fi


# -- Aggregate transformed outputs --
if [ $MODE == "all" ] || [ $MODE == "aggregate" ];
then
    for idx in ${FOLD_IDXS}
    do
        for split in train valid test
        do
            echo $DRIVER
            echo $TRIAL_NAME
            python ${SRC}/chords/aggregate_likelihood_estimations.py \
${OUTPUTS}/${DRIVER}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${split}.hdf5 \
${ESTIMATIONS}/${DRIVER}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${split}${PRED_TAG}.json \
--prediction_fx=${PREDICTION_FX}
        done
    done
fi

# -- Score results --
if [ $MODE == "all" ] || [ $MODE == "score" ];
then
    for idx in ${FOLD_IDXS}
    do
        for split in train valid test
        do
            python ${SRC}/chords/score_estimations.py \
${ESTIMATIONS}/${DRIVER}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${split}${PRED_TAG}.json \
${RESULTS}/${DRIVER}/${DATA_SOURCE}/${idx}/${TRIAL_NAME}/${split}${PRED_TAG}.json
        done
    done
fi
