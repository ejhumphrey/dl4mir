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
BASEDIR=${DL4MIR}/rock_corpus
SRC=./dl4mir

# Directory of optimus data files, divided by index and split, like
#   ${BIGGIE}/${FOLD}/${SPLIT}.hdf5
BIGGIE=${BASEDIR}/biggie
ESTIMATIONS=${BASEDIR}/estimations
META=${BASEDIR}/metadata
OUTPUTS=${BASEDIR}/outputs
REFERENCES=${BASEDIR}/references.jamset
RESULTS=${BASEDIR}/results

MODELS=${DL4MIR}/chord_estimation/models

TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"
VALIDATION_CONFIG="validation_config.json"
VALIDATION_PARAMS="validation_params.json"

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
# -- Final Predictions --
# 1. Transform data with the final parameters.
if [ $PHASE == "all" ] || [ $PHASE == "predict" ] || [ $PHASE == "predict.transform" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Transforming ${BIGGIE}/${idx}/${split}.hdf5"
        python ${SRC}/common/transform_stash.py \
${BIGGIE}/stash.hdf5 \
"cqt" \
${MODELS}/${CONFIG}/${idx}/${TRANSFORM_NAME}.json \
${MODELS}/${CONFIG}/${idx}/${TRANSFORM_NAME}.npz \
${OUTPUTS}/${CONFIG}/${idx}/stash.hdf5
        done
    done
fi

# Transform data
if [ $PHASE == "all" ] || [ $PHASE == "predict" ] || [ $PHASE == "predict.decode" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Decoding ${BIGGIE}/${idx}/stash.hdf5"
        echo ${OUTPUTS}/${CONFIG}/${idx}/stash.hdf5 > ${OUTPUTS}/${CONFIG}/${idx}/stash_list.txt

        python ${SRC}/chords/decode_posteriors_to_jams.py \
${OUTPUTS}/${CONFIG}/${idx}/stash_list.txt \
${ESTIMATIONS}/${CONFIG}/${idx}/stash \
--config=${MODELS}/${CONFIG}/${idx}/viterbi_params.json
        rm ${OUTPUTS}/${CONFIG}/${idx}/stash_list.txt
        done
    done
fi

if [ $PHASE == "all" ] || [ $PHASE == "predict" ] || [ $PHASE == "predict.evaluate" ];
then
    for idx in ${FOLD_IDXS}
    do
        echo "Collecting estimations."
        python ${SRC}/common/collect_files.py \
${ESTIMATIONS}/${CONFIG}/${idx}/stash/ \
"best/*.jamset" \
${ESTIMATIONS}/${CONFIG}/${idx}/stash/${PARAM_TEXTLIST}

            python ${SRC}/chords/score_jamset_textlist.py \
${REFERENCES} \
${ESTIMATIONS}/${CONFIG}/${idx}/stash/${PARAM_TEXTLIST} \
${RESULTS}/${CONFIG}/${idx}/final/stash.json \
--min_support=60.0 \
--num_cpus=1
    done
fi


if [ $PHASE == "all" ] || [ $PHASE == "predict" ] || [ $PHASE == "predict.score" ];
then
    echo "Collecting estimations."
    python ${SRC}/common/collect_files.py \
${RESULTS}/${CONFIG}/ \
"*/final/stash.json" \
${RESULTS}/${CONFIG}/stash_${PARAM_TEXTLIST}

    python ${SRC}/chords/average_results.py \
${RESULTS}/${CONFIG}/stash_${PARAM_TEXTLIST} \
${RESULTS}/${CONFIG}/stash.json
fi
