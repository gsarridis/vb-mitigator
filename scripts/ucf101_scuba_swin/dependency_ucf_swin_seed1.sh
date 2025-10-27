#!/bin/bash

# Methods and seed
# METHODS=("debian" "jtt" "lff" "sd" "di" "bb" "end" "groupdro" "badd" "flac" "maviasb")
# METHODS=("badd" "flac" "maviasb" "lff" "groupdro")
METHODS=("flacb" "mavias")
SEED=1

# Path to the script that accepts method and seed as arguments
RUN_SCRIPT="./scripts/ucf101_scuba_swin/run_exps_ucf_swin.sh"  # make sure this is your updated script with method & seed arguments

# Loop over each method
for METHOD in "${METHODS[@]}"; do
    echo "Submitting jobs for method: $METHOD"

    # Submit the first job
    jid1=$(sbatch "$RUN_SCRIPT" "$METHOD" "$SEED" | awk '{print $4}')
    echo "  Job 1 submitted: $jid1"

    # Submit dependent jobs
    jid2=$(sbatch --dependency=afterany:$jid1 "$RUN_SCRIPT" "$METHOD" "$SEED" | awk '{print $4}')
    echo "  Job 2 submitted: $jid2"

    jid3=$(sbatch --dependency=afterany:$jid2 "$RUN_SCRIPT" "$METHOD" "$SEED" | awk '{print $4}')
    echo "  Job 3 submitted: $jid3"

    jid4=$(sbatch --dependency=afterany:$jid3 "$RUN_SCRIPT" "$METHOD" "$SEED" | awk '{print $4}')
    echo "  Job 4 submitted: $jid4"

    jid5=$(sbatch --dependency=afterany:$jid4 "$RUN_SCRIPT" "$METHOD" "$SEED" | awk '{print $4}')
    echo "  Job 5 submitted: $jid5"
done

echo "All jobs submitted!"
