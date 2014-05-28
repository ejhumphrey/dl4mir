#!/bin/bash
BASEDIR=/Volumes/megatron/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/ejhumphrey/dl4mir

# Flat directory of all audio
AUDIO=${BASEDIR}/audio
CQTS=${BASEDIR}/cqts
LCNCQTS=${BASEDIR}/lcn_cqts
LABS=${BASEDIR}/labs
META=${BASEDIR}/metadata
# Directory of optimus data files, divided by index and split, like
#   ${DATA}/${FOLD_IDX}/${SPLIT_NAME}.hdf5
OPTFILES=${BASEDIR}/optfiles

AUDIO_FILES=${AUDIO}/filelist.txt
CQT_FILES=${CQTS}/filelist.txt
CQT_PARAMS=${META}/cqt_params.json

LCNCQT_FILES=${LCNCQTS}/filelist.txt
LCN_DIM0=11
LCN_DIM1=45

NUM_FOLDS=5
VALID_RATIO=0.15
SPLIT_FILE=${META}/data_splits.json
REFERENCE_FILE=${META}/reference_chords.json

if [ -z "$1" ]; then
    echo "Usage:"
    echo "build.sh {clean|cqt|lcn|labs|splits|optimus|all}"
    echo $'\tclean - Cleans the directory structure'
    echo $'\tcqt - Builds the CQTs'
    echo $'\tlcn - Applies LCN to the CQTs (assumes the exist)'
    echo $'\tlabs - Collects labfiles as a single JSON object'
    echo $'\tsplits - Builds the json metadata files'
    echo $'\toptimus - Builds optimus files from the input data'
    echo $'\tall - Do everything, in order'
    exit 0
fi

echo "Updating audio file list."
python ${SRC}/common/collect_files.py \
${AUDIO} \
"*.mp3" \
${AUDIO_FILES}

# -- CQT --
if [ "$1" == "cqt" ] || [ "$1" == "all" ]; then
    echo "Computing CQTs..."
    python ${SRC}/common/audio_files_to_cqt_arrays.py \
${AUDIO_FILES} \
${CQTS} \
--cqt_params=${CQT_PARAMS}
exit 0
fi

echo "Updating CQT file list."
python ${SRC}/common/collect_files.py \
${CQTS} \
"*.npy" \
${CQT_FILES}


# -- LCN --
if [ "$1" == "lcn" ] || [ "$1" == "all" ]; then
    echo "Computing LCN over CQTs..."
    python ${SRC}/common/apply_lcn_to_arrays.py \
${CQT_FILES} \
${LCN_DIM0} \
${LCN_DIM1} \
${LCNCQTS}
exit 0
fi

echo "Updating LCN-CQT numpy file list."
python ${SRC}/common/collect_files.py \
${LCNCQTS} \
"*.npy" \
${LCNCQT_FILES}


# -- Stratification --
if [ "$1" == "splits" ] || [ "$1" == "all" ]; then
    echo "Stratifying data..."
    python ${SRC}/common/stratify_data.py \
${AUDIO_FILES} \
${NUM_FOLDS} \
${VALID_RATIO} \
${SPLIT_FILE}
exit 0
fi

# -- Labs --
if [ "$1" == "labs" ] || [ "$1" == "all" ]; then
    echo "Collecting lab files..."
    python ${SRC}/chords/collect_labfiles.py \
${LABS} \
${REFERENCE_FILE}
exit 0
fi

# -- Optimus --
if [ "$1" == "optimus" ] || [ "$1" == "all" ]; then
    if [ -d ${OPTFILES} ]; then
        rm -r ${OPTFILES}
    fi
    echo "Building the Optimus files"
    python ${SRC}/chords/file_importer.py \
${SPLIT_FILE} \
${LCNCQTS} \
${CQT_PARAMS} \
${LABS} \
${OPTFILES}
exit 0
fi
