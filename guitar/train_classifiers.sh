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

BASEDIR=${DL4MIR}/guitar
SRC=./dl4mir

# Directory of optimus data files, divided by index and split, like
#   ${BIGGIE}/${FOLD}/${SPLIT}.hdf5
BIGGIE=${BASEDIR}/biggie/chords_l2n
ESTIMATIONS=${BASEDIR}/estimations
INITS=${BASEDIR}/param_inits
META=${BASEDIR}/metadata
MODELS=${BASEDIR}/models
OUTPUTS=${BASEDIR}/outputs
RESULTS=${BASEDIR}/results

TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"
VALIDATION_CONFIG="validation_config.json"
VALIDATION_PARAMS="validation_params.json"

REFERENCES=${DL4MIR}/chord_estimation/references.jamset


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
        python ${SRC}/guitar/driver.py \
${BIGGIE}/${idx}/train.hdf5 \
${ARCH_SIZE} \
${DROPOUT} \
${MODELS}/${CONFIG}/${idx} \
"guitarnet" \
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

        python ${SRC}/common/validation_sweep.py \
${BIGGIE}/${idx}/valid.hdf5 \
${MODELS}/${CONFIG}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${CONFIG}/${idx}/${PARAM_TEXTLIST} \
${OUTPUTS}/${CONFIG}/${idx}/valid \
--start_index=39 \
--stride=40
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

        python ${SRC}/guitar/decode_fretboards_to_jams.py \
${OUTPUTS}/${CONFIG}/${idx}/valid/${PARAM_TEXTLIST} \
${META}/${VALIDATION_CONFIG} \
"chords" \
${ESTIMATIONS}/${CONFIG}/${idx}/valid/
    done
fi

# -- Model Selection --
# 3. Compute cumulative scores over the collections
if [ $PHASE == "all" ] || [ $PHASE == "validate" ] || [ $PHASE == "validate.evaluate" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting estimations."
        python ${SRC}/common/collect_files.py \
${ESTIMATIONS}/${CONFIG}/${idx}/valid/ \
"*/*.jamset" \
${ESTIMATIONS}/${CONFIG}/${idx}/valid/${PARAM_TEXTLIST}

        python ${SRC}/chords/score_jamset_textlist.py \
${REFERENCES} \
${ESTIMATIONS}/${CONFIG}/${idx}/valid/${PARAM_TEXTLIST} \
${RESULTS}/${CONFIG}/${idx}/valid.json \
--min_support=60.0 \
--num_cpus=1
    done
fi

if [ $PHASE == "all" ] || [ $PHASE == "validate" ] || [ $PHASE == "validate.select" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting estimations."
        python ${SRC}/guitar/select_best.py \
${RESULTS}/${CONFIG}/${idx}/valid.json \
${MODELS}/${CONFIG}/${idx}/${TRANSFORM_NAME}.npz \
${MODELS}/${CONFIG}/${idx}/viterbi_params.json
    done
fi

# Transform data
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
            echo ${OUTPUTS}/${CONFIG}/${idx}/${split}.hdf5 > ${OUTPUTS}/${CONFIG}/${idx}/stash_list.txt

            python ${SRC}/guitar/decode_fretboards_to_jams.py \
${OUTPUTS}/${CONFIG}/${idx}/stash_list.txt \
config=${MODELS}/${CONFIG}/${idx}/viterbi_params.json \
"chords"
${ESTIMATIONS}/${CONFIG}/${idx}/${split}/chords

            python ${SRC}/guitar/decode_fretboards_to_jams.py \
${OUTPUTS}/${CONFIG}/${idx}/stash_list.txt \
config=${MODELS}/${CONFIG}/${idx}/viterbi_params.json \
"tabs"
${ESTIMATIONS}/${CONFIG}/${idx}/${split}/tabs
            rm ${OUTPUTS}/${CONFIG}/${idx}/stash_list.txt
        done
    done
fi

if [ $PHASE == "all" ] || [ $PHASE == "predict" ] || [ $PHASE == "predict.evaluate" ];
then
    for idx in ${FOLD_IDXS}
    do
        for split in valid test train
        do
            echo "Collecting estimations."
            python ${SRC}/common/collect_files.py \
${ESTIMATIONS}/${CONFIG}/${idx}/${split}/ \
"best/*.jamset" \
${ESTIMATIONS}/${CONFIG}/${idx}/${split}/${PARAM_TEXTLIST}

            python ${SRC}/chords/score_jamset_textlist.py \
${REFERENCES} \
${ESTIMATIONS}/${CONFIG}/${idx}/${split}/${PARAM_TEXTLIST} \
${RESULTS}/${CONFIG}/${idx}/final/${split}.json \
--min_support=60.0 \
--num_cpus=1
        done
    done
fi

