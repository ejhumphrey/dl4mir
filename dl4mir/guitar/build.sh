#!/bin/bash
BASEDIR=/Volumes/megatron/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/ejhumphrey/dl4mir

# Flat directory of all audio
LCNCQTS=${BASEDIR}/lcn_cqts
TABS=${BASEDIR}/tabs
META=${BASEDIR}/metadata
# Directory of optimus data files, divided by index and split, like
#   ${DATA}/${FOLD_IDX}/${SPLIT_NAME}.hdf5
GUITAR_DSETS=${BASEDIR}/guitar_dsets
CQT_PARAMS=${META}/cqt_params.json
LCNCQT_FILES=${LCNCQTS}/filelist.txt
SPLIT_FILE=${META}/data_splits.json

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


# -- Optimus --
if [ "$1" == "optimus" ] || [ "$1" == "all" ]; then
    if [ -d ${GUITAR_DSETS} ]; then
        rm -r ${GUITAR_DSETS}
    fi
    echo "Building the Optimus files"
    python ${SRC}/guitar/file_importer.py \
${SPLIT_FILE} \
${LCNCQTS} \
${CQT_PARAMS} \
${TABS} \
${GUITAR_DSETS}
fi
