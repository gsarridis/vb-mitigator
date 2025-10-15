#!/bin/bash

# Submit a chain of dependent jobs
jid1=$(sbatch ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid2=$(sbatch --dependency=afterany:$jid1 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid3=$(sbatch --dependency=afterany:$jid2 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid4=$(sbatch --dependency=afterany:$jid3 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid5=$(sbatch --dependency=afterany:$jid4 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid6=$(sbatch --dependency=afterany:$jid5 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid7=$(sbatch --dependency=afterany:$jid6 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid8=$(sbatch --dependency=afterany:$jid7 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid9=$(sbatch --dependency=afterany:$jid8 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid10=$(sbatch --dependency=afterany:$jid9 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid11=$(sbatch --dependency=afterany:$jid10 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid12=$(sbatch --dependency=afterany:$jid11 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')
jid13=$(sbatch --dependency=afterany:$jid12 ./scripts/run_exps_ucf101_swin_erm_hua.sh | awk '{print $4}')

echo "Final job ID: $jid13"
