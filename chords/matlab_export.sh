#!/bin/bash
#
# Train a set of end-to-end classifiers and sweep over the checkpointed
#    parameters to identify the early stopping point.

BASEDIR=/media/attic/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/dl4mir

# Directory of optimus data files, divided by index and split, like
METADATA=${BASEDIR}/metadata
OUTPUTS=${BASEDIR}/outputs
FEATURES=${BASEDIR}/features

if [ -z "$1" ]; then
    echo "Usage:"
    echo "train.sh driver trial {[0-4]|*all}"
    echo $'\tdriver - Name of the training driver.'
    echo $'\tfold# - Number of the training fold, default=all.'
    exit 0
fi

DRIVER=$1
TRIAL=$2

if [ "$3" == "all" ] || [ -z "$3" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$3
fi

for idx in ${FOLD_IDXS}
do
    for split in valid test train
    do
        echo "Synchronizing ${OUTPUTS}/${DRIVER}/${idx}/${TRIAL}/${split}.hdf5"
        python ${SRC}/chords/beat_sync_entities.py \
${OUTPUTS}/${DRIVER}/${idx}/${TRIAL}/${split}.hdf5 \
${METADATA}/beat_times.json \
${OUTPUTS}/${DRIVER}/${idx}/${TRIAL}/${split}-beatsync.hdf5

        echo "Exporting ${OUTPUTS}/${DRIVER}/${idx}/${TRIAL}/${split}-beatsync.hdf5"
        python ${SRC}/chords/export_chroma_to_mats.py \
${OUTPUTS}/${DRIVER}/${idx}/${TRIAL}/${split}-beatsync.hdf5 \
${METADATA}/chord_codes.json \
${FEATURES}/${DRIVER}/${idx}/${TRIAL}
    done
done
