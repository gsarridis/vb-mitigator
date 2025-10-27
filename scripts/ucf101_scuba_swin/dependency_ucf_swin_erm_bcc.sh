#!/bin/bash

# Submit a chain of dependent jobs
jid1=$(sbatch ./scripts/ucf101_scuba_swin/run_exps_ucf101_scuba_bcc_hua.sh | awk '{print $4}')
jid2=$(sbatch --dependency=afterany:$jid1 ./scripts/ucf101_scuba_swin/run_exps_ucf101_scuba_bcc_hua.sh | awk '{print $4}')
jid3=$(sbatch --dependency=afterany:$jid2 ./scripts/ucf101_scuba_swin/run_exps_ucf101_scuba_bcc_hua.sh | awk '{print $4}')
jid4=$(sbatch --dependency=afterany:$jid3 ./scripts/ucf101_scuba_swin/run_exps_ucf101_scuba_bcc_hua.sh | awk '{print $4}')
jid5=$(sbatch --dependency=afterany:$jid4 ./scripts/ucf101_scuba_swin/run_exps_ucf101_scuba_bcc_hua.sh | awk '{print $4}')

echo "Final job ID: $jid5"
