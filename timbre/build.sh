#!/bin/bash
BASEDIR=/Volumes/megatron/dl4mir/timbre_sim
SRC=~/Dropbox/NYU/marldev/src/dl4mir

# Flat directory of all audio
AUDIO=${BASEDIR}/audio
CQTS=${BASEDIR}/features/cqts
META=${BASEDIR}/metadata
# Directory of biggie Stashes, divided by index and split, like
#   ${DATA}/${FOLD_IDX}/${SPLIT_NAME}.hdf5
BIGGIE=${BASEDIR}/biggie/timbre

AUDIO_EXT="wav"
AUDIO_FILES=${AUDIO}/filelist.txt
CQT_FILES=${CQTS}/filelist.txt
CQT_PARAMS=${META}/cqt_params.json

# Stratification params
NUM_FOLDS=5
VALID_RATIO=0.25
SUBSETS=${META}/set_configs.json
SPLIT_FILE=${META}/data_splits.json


if [ -z "$1" ]; then
    echo "Usage:"
    echo "build.sh {clean|cqt|lcn|labs|splits|biggie|all}"
    echo $'\tclean - Cleans the directory structure'
    echo $'\tcqt - Builds the CQTs'
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


# -- Stratification --
if [ "$1" == "splits" ] || [ "$1" == "all" ]; then
    echo "Stratifying data..."
    python ${SRC}/timbre/stratify_data.py \
${AUDIO_FILES} \
${SUBSETS} \
${NUM_FOLDS} \
${VALID_RATIO} \
${SPLIT_FILE}
fi


# -- Biggie Files --
if [ "$1" == "biggie" ] || [ "$1" == "all" ]; then
    if [ -d ${BIGGIE} ]; then
        rm -r ${BIGGIE}
    fi
    echo "Building the Biggie files"
    python ${SRC}/timbre/file_importer.py \
${SPLIT_FILE} \
${CQTS} \
${BIGGIE}
fi
