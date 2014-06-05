#!/bin/bash
BASEDIR=~/dl4mir_test/chord_estimation
SRC=~/Dropbox/NYU/marldev/src/ejhumphrey/dl4mir

# Flat directory of all audio
LABS=${BASEDIR}/labs
META=${BASEDIR}/metadata
# Directory of optimus data files, divided by index and split, like
#   ${DATA}/${FOLD_IDX}/${SPLIT_NAME}.hdf5
OPTFILES=${BASEDIR}/optfiles

PARAM_FILES=${AUDIO}/filelist.txt
CQT_FILES=${CQTS}/filelist.txt
CQT_PARAMS=${META}/cqt_params.json

LCNCQT_FILES=${LCNCQTS}/filelist.txt
LCN_DIM0=11
LCN_DIM1=45

NUM_FOLDS=5
VALID_RATIO=0.15
SPLIT_FILE=${META}/data_splits.json

if [ -z "$1" ]; then
    echo "Usage:"
    echo "build.sh {clean|cqt|lcn|splits|optimus|all}"
    echo $'\tclean - Cleans the directory structure'
    echo $'\tcqt - Builds the CQTs'
    echo $'\tlcn - Applies LCN to the CQTs (assumes the exist)'
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

# -- Parameter Selection --
if [ "$1" == "cqt" ] || [ "$1" == "all" ]; then
    echo "Computing CQTs..."

    SPLIT=train
    for (( idx=0; idx< 1; idx++ ))
    do
        for drv in DRIVER
        do
            python ${SRC}/chords/drivers/${DRIVER}.py \
${OPTFILES}/${idx}/${SPLIT}.hdf5 \
${MODELS}/${DRIVER}/${idx} \
${TRIAL_NAME} \
${TRANSFORM_NAME}
        done
    done
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


# -- Predict & Score --
MEDFILT=0
for PENALTY in 0 5 10 20
do
    python {$SRC}/chords/posteriors_to_labeled_intervals.py \
{$OUTPUTS}/${DRIVER}/${FOLD}/${NAME}/${SPLIT}.hdf5 \
${PENALTY} \
${MEDFILT} \
${META}/cqt_params.json \
${PREDICTIONS}/${DRIVER}/${FOLD}/${NAME}/${SPLIT}-V${PENALTY}-M${MEDFILT}.json

    python ${SRC}/chords/score_estimations.py \
${PREDICTIONS}/${DRIVER}/${FOLD}/${NAME}/${SPLIT}-V${PENALTY}-M${MEDFILT}.json \
${META}/reference_chords.json \
${RESULTS}/${DRIVER}/${FOLD}/${NAME}/${SPLIT}-V${PENALTY}-M${MEDFILT}.json
done

