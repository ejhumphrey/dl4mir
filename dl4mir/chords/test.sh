#!/bin/bash
#
# Train a set of end-to-end classifiers and sweep over the checkpointed
#    parameters to identify the early stopping point.


TRIAL_NAME="test123"
VALIDATOR_NAME="validator"
TRANSFORM_NAME="transform"
PARAM_TEXTLIST="paramlist.txt"


if [ -z "$1" ]; then
    echo "Usage:"
    echo "train_classifiers.sh {driver|all} {[0-5]|all}"
    echo $'\tdriver - Name of the training driver, or all.'
    echo $'\tfold - Number of the training fold, or all.'
    exit 0
fi

DRIVER="classifier-L05-V157 "\
"classifier-L80-V157 "\
"$1"

if [ "$2" == "all" ] || [ -z "$2" ];
then
    echo "Setting all folds"
    FOLD_IDXS=$(seq 0 4)
else
    FOLD_IDXS=$2
fi

# Train networks
for x in ${FOLD_IDXS}
do
    echo ${x}
done

