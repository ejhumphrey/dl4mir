#!/bin/bash
#
# Train a set of end-to-end classifiers and sweep over the checkpointed
#    parameters to identify the early stopping point.

BASEDIR=/media/attic/dl4mir/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/ejhumphrey/dl4mir

# Directory of optimus data files, divided by index and split, like
METADATA=${BASEDIR}/metadata
OUTPUTS=${BASEDIR}/outputs
MATS=${BASEDIR}/mats

if [ -z "$1" ]; then
    echo "Usage:"
    echo "train.sh {driver|all} {[0-4]|*all} {name|all}"
    echo $'\tdriver - Name of the training driver.'
    echo $'\tfold# - Number of the training fold, default=all.'
    exit 0
fi

if [ "$1" == "all" ]
then
    echo "Setting all known drivers..."
    DRIVER="chroma-L05 "\
"tonnetz-L05 "\
"classifier-L05-V157 "\
"classifier-L10-V157 "\
"classifier-L20-V157 "\
"classifier-L40-V157 "\
"classifier-L80-V157"
else
    DRIVER="$1"
fi

if [ "$2" == "all" ] || [ -z "$2" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$2
fi

TRIAL_NAME=$3

for drv in ${DRIVER}
do
    for idx in ${FOLD_IDXS}
    do
        for split in valid test train
        do
            echo "Synchronizing ${OUTPUTS}/${drv}/${idx}/${TRIAL_NAME}/${split}.hdf5"
            python ${SRC}/chords/beat_sync_entities.py \
${OUTPUTS}/${drv}/${idx}/${TRIAL_NAME}/${split}.hdf5 \
${METADATA}/beat_times.json \
${OUTPUTS}/${drv}/${idx}/${TRIAL_NAME}/${split}-beatsync.hdf5

            echo "Exporting ${OUTPUTS}/${drv}/${idx}/${TRIAL_NAME}/${split}-beatsync.hdf5"
            python ${SRC}/chords/export_chroma_to_mats.py \
${OUTPUTS}/${drv}/${idx}/${TRIAL_NAME}/${split}-beatsync.hdf5 \
${METADATA}/chord_codes.json \
${MATS}/${drv}/${idx}/${TRIAL_NAME}
        done
    done
done
