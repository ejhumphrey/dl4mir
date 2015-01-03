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
VALIDATION_PARAMS="validation_params.json"

PVALS="-10.0 -15.0 -20.0 -25.0 -30.0 -35.0"

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

# -- Model Selection --
# 1. Transform the validation stash with various parameter checkpoints.
if [ $PHASE == "all" ] || [ $PHASE == "validate" ] || [ $PHASE == "validate.transform" ];
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

# -- Model Selection --
# 2. Decode the resulting posteriors to JAMS estimations.
if [ $PHASE == "all" ] || [ $PHASE == "validate" ] || [ $PHASE == "validate.decode" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting parameters."
        python ${SRC}/common/collect_files.py \
${OUTPUTS}/${CONFIG}/${idx}/valid \
"*.hdf5" \
${OUTPUTS}/${CONFIG}/${idx}/valid/${PARAM_TEXTLIST}

        echo ${SRC}/chords/decode_posteriors_to_jams.py \
${OUTPUTS}/${CONFIG}/${idx}/valid/${PARAM_TEXTLIST} \
${ESTIMATIONS}/${CONFIG}/${idx}/valid/ \
--penalty_values=PVALS \
    done
fi

# -- Model Selection --
# 3. Compute cumulative scores over the collections
if [ $PHASE == "all" ] || [ $PHASE == "validate" ] || [ $PHASE == "validate.evaluate" ];
then
    for idx in ${FOLD_IDXS}
    do
        python ${SRC}/chords/validation_eval.py \
${ESTIMATIONS}/${CONFIG}/${idx}/valid \
${RESULTS}/${CONFIG}/${idx}/valid \
${MODELS}/${CONFIG}/${idx}/${TRANSFORM_NAME}.npz \
${MODELS}/${CONFIG}/${idx}/${VALIDATION_PARAMS}
    done
fi

# -- Final Predictions --
# 1. Transform data with the final parameters.
if [ $PHASE == "all" ] || [ $PHASE == "predict" ] || [ $PHASE == "predict.transform" ];
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

# Transform data
if [ $PHASE == "all" ] || [ $PHASE == "predict" ] || [ $PHASE == "predict.decode" ];
then
    for idx in ${FOLD_IDXS}
    do
        for split in valid test train
        do
            echo "Decoding ${BIGGIE}/${idx}/${split}.hdf5"
            echo ${OUTPUTS}/${CONFIG}/${idx}/${split}.hdf5 > ${PARAM_TEXTLIST}

            python ${SRC}/chords/decode_posteriors_to_jams.py \
${PARAM_TEXTLIST} \
${ESTIMATIONS}/${CONFIG}/${idx}/${split} \
--config=${MODELS}/${CONFIG}/${idx}/${VALIDATION_PARAMS}
            rm ${PARAM_TEXTLIST}
        done
    done
fi
