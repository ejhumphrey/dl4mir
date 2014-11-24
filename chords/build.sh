#!/bin/bash
BASEDIR=/Volumes/megatron/dl4mir/chord_estimation/rock_corpus
SRC=~/Dropbox/NYU/marldev/src/dl4mir

# Flat directory of all audio
AUDIO=${BASEDIR}/audio
CQTS=${BASEDIR}/features/cqts
LCNCQTS=${BASEDIR}/features/l2n_cqts
LABS=${BASEDIR}/labs
META=${BASEDIR}/metadata
# Directory of biggie Stashes, divided by index and split, like
#   ${DATA}/${FOLD_IDX}/${SPLIT_NAME}.hdf5
STASH=${BASEDIR}/biggie/chords_l2n
WSTASH=${STASH}_wrap

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
REFERENCE_FILE=${META}/reference_chords.json

# Wrapping params
LENGTH=40
STRIDE=36

# Beat-sync params
BEAT_TIMES=${META}/beat_times.json
SUBDIVIDE=2
BS_STASH=${STASH}_bs${SUBDIVIDE}_med

if [ -z "$1" ]; then
    echo "Usage:"
    echo "build.sh {clean|cqt|lcn|labs|splits|biggie|all}"
    echo $'\tclean - Cleans the directory structure'
    echo $'\tcqt - Builds the CQTs'
    echo $'\tlcn - Applies LCN to the CQTs (assumes the exist)'
    echo $'\tlabs - Collects labfiles as a single JSON object'
    echo $'\tsplits - Builds the json metadata files'
    echo $'\tbiggie - Builds biggie dataset files'
    echo $'\twrap - Wraps cqts down to 3D tensors'
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

# -- Labs --
# Unnecessary??
if [ "$1" == "labs" ] || [ "$1" == "all" ]; then
    echo "Collecting lab files..."
    python ${SRC}/chords/collect_labfiles.py \
${LABS} \
${REFERENCE_FILE}
fi

# -- Biggie Files --
if [ "$1" == "biggie" ] || [ "$1" == "all" ]; then
    if [ -d ${STASH} ]; then
        rm -r ${STASH}
    fi
    echo "Updating LCN-CQT numpy file list."
    python ${SRC}/common/collect_files.py \
${LCNCQTS} \
"*.npz" \
${LCNCQT_FILES}

    echo "Building the Biggie files"
    python ${SRC}/chords/pitch_file_importer.py \
${SPLIT_FILE} \
${LCNCQTS} \
${LABS} \
${STASH}
fi


if [ "$1" == "stats" ] || [ "$1" == "all" ]; then
    echo "Computing dataset statistics..."
    for ((idx=0; idx<NUM_FOLDS; idx++))
    do
        for split in valid test train
        do
            python ${SRC}/chords/compute_dataset_stats.py \
${STASH}/${idx}/${split}.hdf5 \
${STASH}/${idx}/${split}.json
        done
    done
fi


# -- Wrap CQT octaves Files --
if [ "$1" == "wrap" ] || [ "$1" == "all" ]; then
    if [ -d ${WSTASH} ]; then
        rm -r ${WSTASH}
    fi
    echo "Wrapping the CQTs..."
    for ((idx=0; idx<NUM_FOLDS; idx++))
    do
        for split in valid test train
        do
            python ${SRC}/common/wrap_cqts.py \
${STASH}/${idx}/${split}.hdf5 \
${LENGTH} \
${STRIDE} \
${WSTASH}/${idx}/${split}.hdf5
        done
    done
fi

# -- Beat-sync Biggie files --
if [ "$1" == "beatsync" ] || [ "$1" == "all" ]; then
    if [ -d ${BS_STASH} ]; then
        rm -r ${BS_STASH}
    fi
    echo "Beat-synchronizing the CQTs..."
    for ((idx=0; idx<NUM_FOLDS; idx++))
    do
        for split in valid test train
        do
            python ${SRC}/chords/beat_sync_entities.py \
${STASH}/${idx}/${split}.hdf5 \
${BEAT_TIMES} \
${BS_STASH}/${idx}/${split}.hdf5 \
--subdivide=${SUBDIVIDE}
        done
    done
fi

if [ "$1" == "beatsync-stats" ] || [ "$1" == "all" ]; then
    echo "Computing dataset statistics..."
    for ((idx=0; idx<NUM_FOLDS; idx++))
    do
        for split in valid test train
        do
            python ${SRC}/chords/compute_dataset_stats.py \
${BS_STASH}/${idx}/${split}.hdf5 \
${BS_STASH}/${idx}/${split}.json
        done
    done
fi
