#!/bin/bash
BASEDIR=/Volumes/megatron/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/ejhumphrey/dl4mir

# Flat directory of all audio
AUDIO=${BASEDIR}/audio
# Directory of optimus data files, divided by index and split, like
#   ${DATA}/fold_${IDX}/fold_${IDX}-${SPLIT}.hdf5
DATA=${BASEDIR}/data
# Flat directory of time-frequency representations
CQTS=${BASEDIR}/cqts
LCNCQTS=${BASEDIR}/lcn_cqts
LABS=${BASEDIR}/labs
META=${BASEDIR}/metadata
MODELS=${BASEDIR}/models
PREDICTIONS=${BASEDIR}/predictions
RESULTS=${BASEDIR}/results

# Manifests and sidechain files
CQT_PARAMS=${META}/cqt_params.json
AUDIO_FILES=${AUDIO}/filelist.txt
CQT_FILES=${CQTS}/filelist.txt
LCN_DIM0=11
LCN_DIM1=45

if [ -z "$1" ]; then
    echo "Usage:"
    echo "build.sh {clean|cqt|lcn|json|optimus|all}"
    echo $'\tclean - Cleans the directory structure'
    echo $'\tcqt - Builds the CQTs'
    echo $'\tcqt - Applies LCN to the CQTs (assumes the exist)'
    echo $'\tjson - Builds the json metadata files'
    echo $'\toptimus - Builds optimus files from the input data'
    echo $'\tall - Do everything, in order'
fi

if [ "$1" == "cqt" ] || [ "$1" == "all" ]; then
    echo "Collecting audio files"
    python ${SRC}/scripts/common/collect_files.py \
${AUDIO} \
"*.mp3" \
${AUDIO_FILES}

    echo "Building the CQTs"
    python ${SRC}/scripts/common/audio_files_to_cqt_arrays.py \
${AUDIO_FILES} \
${CQTS} \
--cqt_params=${CQT_PARAMS}
fi

if [ "$1" == "lcn" ]; then
    echo "Collecting CQT numpy files"
    python ${SRC}/scripts/common/collect_files.py \
${CQTS} \
"*.npy" \
${CQT_FILES}

    echo "Computing LCN over CQTs"
    python ${SRC}/scripts/common/apply_lcn_to_arrays.py \
${CQT_FILES} \
${LCN_DIM0} \
${LCN_DIM1} \
${LCNCQTS}
fi








# FILE_LIST=cal10k_filelist.txt
# INPUT_DIR=/Datasets/Cal10k/mp3
# CQT_DIR=/Datasets/Cal10k/cqts
# TRAIN_META_FILE=/Datasets/Cal10k/meta/tracktags_train.json
# TEST_META_FILE=/Datasets/Cal10k/meta/tracktags_test.json
# TRAIN_JSON_FILE=/Datasets/Cal10k/meta/optimus_import_train.json
# TEST_JSON_FILE=/Datasets/Cal10k/meta/optimus_import_test.json
# TRAIN_OPTIMUS_FP=/Datasets/Cal10k/cal10k_train.hdf5
# TEST_OPTIMUS_FP=/Datasets/Cal10k/cal10k_test.hdf5
# IMPORTER_CONFIG=import_config.json

# Builds the optimus file, given the input raw data.
# if [ -z "$1" ]; then
#     echo "Usage:"
#     echo "build.sh {clean|cqt|json|optimus}"
#     echo $'\tclean - cleans the cqt dir & optimus file'
#     echo $'\tcqt - builds the CQTs'
#     echo $'\tjson - builds the json meta file'
#     echo $'\toptimus - builds the optimus file from the input data'
# fi

# if [ "$1" == "clean" ]; then
#     echo "Cleaning cqt dir & optimus file."
#     rm ${CQT_DIR}/*
#     rm ${OPTIMUS_FP}/*
# fi

# # Step 1: convert the input MP3's to cqt/npy
# if [ "$1" == "cqt" ]; then
#     echo "Building the CQT"
#     python ${MARL}/scripts/audio_files_to_cqt_arrays.py ${FILE_LIST} ${CQT_DIR}
# fi

# # Step 2: Create the .json metafile containing the labels for each file
# if [ "$1" == "json" ]; then
#     echo "Building the json meta-file"
# # First, get the idxmap
#     python ${MARL}/dataset/cal10k.py idxmap
#     # then, generate the annotation tags
#     python ${MARL}/dataset/cal10k.py track_tags train -o ${TRAIN_META_FILE} -f 1
#     python ${MARL}/dataset/cal10k.py track_tags test -o ${TEST_META_FILE} -f 1
#     # Generate the optimus inpmort files
#     python ${MARL}/dataset/cal10k.py optimus ${CQT_DIR} ${TRAIN_META_FILE} ${TRAIN_JSON_FILE}
#     python ${MARL}/dataset/cal10k.py optimus ${CQT_DIR} ${TEST_META_FILE} ${TEST_JSON_FILE}
# fi

# # Step 3: import into optimus
# if [ "$1" == "optimus" ]; then
#     echo "Building the Optimus file"
#     python file_importer.py ${TRAIN_JSON_FILE} ${TRAIN_OPTIMUS_FP} ${IMPORTER_CONFIG}
#     python file_importer.py ${TEST_JSON_FILE} ${TEST_OPTIMUS_FP} ${IMPORTER_CONFIG}
# fi
