#!/bin/bash
BASEDIR=/Volumes/megatron/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/dl4mir

# Flat directory of all audio
AUDIO=${BASEDIR}/audio
CQTS=${BASEDIR}/features/cqts
LCNCQTS=${BASEDIR}/features/lcn_cqts
LABS=${BASEDIR}/labs
META=${BASEDIR}/metadata
# Directory of biggie Stashes, divided by index and split, like
#   ${DATA}/${FOLD_IDX}/${SPLIT_NAME}.hdf5
DSETS=${BASEDIR}/biggie/chords

AUDIO_FILES=${AUDIO}/filelist.txt
CQT_FILES=${CQTS}/filelist.txt
CQT_PARAMS=${META}/cqt_params.json

LCNCQT_FILES=${LCNCQTS}/filelist.txt
LCN_DIM0=21
LCN_DIM1=11

NUM_FOLDS=5
VALID_RATIO=0.15
SPLIT_FILE=${META}/data_splits.json
REFERENCE_FILE=${META}/reference_chords.json

if [ -z "$1" ]; then
    echo "Usage:"
    echo "build.sh {clean|cqt|lcn|labs|splits|biggie|all}"
    echo $'\tclean - Cleans the directory structure'
    echo $'\tcqt - Builds the CQTs'
    echo $'\tlcn - Applies LCN to the CQTs (assumes the exist)'
    echo $'\tlabs - Collects labfiles as a single JSON object'
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
"*.mp3" \
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

# -- Labs --
if [ "$1" == "labs" ] || [ "$1" == "all" ]; then
    echo "Collecting lab files..."
    python ${SRC}/chords/collect_labfiles.py \
${LABS} \
${REFERENCE_FILE}
fi

# -- Biggie Files --
if [ "$1" == "biggie" ] || [ "$1" == "all" ]; then
    if [ -d ${DSETS} ]; then
        rm -r ${DSETS}
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
${LABS} \
${DSETS}
fi
