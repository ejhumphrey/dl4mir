#!/bin/bash
# BASEDIR=/Volumes/megatron/dl4mir/chord_estimation
BASEDIR=/Volumes/megatron/dl4mir/rock_corpus
SRC=~/Dropbox/NYU/marldev/src/dl4mir

# Flat directory of all audio
AUDIO=${BASEDIR}/audio
CQTS=${BASEDIR}/features/cqts
LCNCQTS=${BASEDIR}/features/l2n_cqts
REFS=${BASEDIR}/references
META=${BASEDIR}/metadata
# Directory of biggie Stashes, divided by index and split, like
#   ${DATA}/${FOLD_IDX}/${SPLIT_NAME}.hdf5
BIGGIE=${BASEDIR}/biggie/chords_l2n

AUDIO_EXT="wav"
AUDIO_FILES=${AUDIO}/filelist.txt
CQT_FILES=${CQTS}/filelist.txt
CQT_PARAMS=${META}/cqt_params.json

# LCN params
LCNCQT_FILES=${LCNCQTS}/filelist.txt
LCN_DIM0=21
LCN_DIM1=11

# Split params
NUM_FOLDS=5
VALID_RATIO=0.15
SPLIT_FILE=${META}/data_splits.json


if [ -z "$1" ]; then
    echo "Usage:"
    echo "build.sh {clean|cqt|lcn|labs|splits|biggie|all}"
    echo $'\tclean - Cleans the directory structure'
    echo $'\tcqt - Builds the CQTs'
    echo $'\tlcn - Applies LCN to the CQTs (assumes the exist)'
    echo $'\tsplits - Builds the json metadata files'
    echo $'\tbiggie - Builds biggie dataset files'
    echo $'\tall - Do everything, in order'
    exit 0
fi

# -- CQT --
if [ "$1" == "cqt" ] || [ "$1" == "all" ]; then
    echo "Updating audio file list."
    python ${SRC}/common/collect_files.py \
${AUDIO} \
"*.${AUDIO_EXT}" \
${AUDIO_FILES}

    echo "Computing CQTs..."
    python ${SRC}/common/audio_files_to_cqt_arrays.py \
${AUDIO_FILES} \
${CQTS} \
--cqt_params=${CQT_PARAMS}

fi


# -- LCN --
if [ "$1" == "lcn" ] || [ "$1" == "all" ]; then
    echo "Updating CQT file list."
    python ${SRC}/common/collect_files.py \
${CQTS} \
"*.npz" \
${CQT_FILES}

    echo "Computing LCN over CQTs..."
    python ${SRC}/common/apply_lcn_to_arrays.py \
${CQT_FILES} \
${LCN_DIM0} \
${LCN_DIM1} \
${LCNCQTS}

fi


# -- Stratification --
if [ "$1" == "splits" ] || [ "$1" == "all" ]; then
    echo "Stratifying data..."
    python ${SRC}/common/stratify_data.py \
${AUDIO_FILES} \
${NUM_FOLDS} \
${VALID_RATIO} \
${SPLIT_FILE}
fi


# -- Biggie Files --
if [ "$1" == "biggie" ] || [ "$1" == "all" ]; then
    if [ -d ${BIGGIE} ]; then
        rm -r ${BIGGIE}
    fi
    echo "Updating LCN-CQT numpy file list."
    python ${SRC}/common/collect_files.py \
${LCNCQTS} \
"*.npz" \
${LCNCQT_FILES}

    echo "Building the Biggie files"
    python ${SRC}/chords/file_importer.py \
${SPLIT_FILE} \
${LCNCQTS} \
${REFS} \
${BIGGIE}
fi


if [ "$1" == "stats" ] || [ "$1" == "all" ]; then
    echo "Computing dataset statistics..."
    for ((idx=0; idx<NUM_FOLDS; idx++))
    do
        for split in valid test train
        do
            python ${SRC}/chords/compute_dataset_stats.py \
${BIGGIE}/${idx}/${split}.hdf5 \
${BIGGIE}/${idx}/${split}.json
        done
    done
fi
