import os
import argparse
import pandas as pd
import re


def find_log_files(method_dir):
    """Find all logs<seed>.csv files and extract seed."""
    log_files = []
    pattern = re.compile(r"logs(\d+)\.csv")

    for f in os.listdir(method_dir):
        match = pattern.match(f)
        if match:
            seed = int(match.group(1))
            log_files.append((os.path.join(method_dir, f), seed))

    return log_files


def process_method(
    dataset_name, method, method_dir, metric1, metric2, selection_metric
):
    results = []

    log_files = find_log_files(method_dir)

    for file_path, seed in log_files:
        df = pd.read_csv(file_path)

        if selection_metric not in df.columns:
            raise ValueError(f"{selection_metric} not found in {file_path}")

        # Select best epoch (maximization assumed)
        best_idx = df[selection_metric].idxmax()
        best_row = df.loc[best_idx]
        if best_row[metric1] < 1 and best_row[metric2] < 1:
            best_row[metric1] *= 100
            best_row[metric2] *= 100

        result = {
            "Dataset": dataset_name,
            "Method": method,
            "Seed": f"seed_{seed}",
            "Best Epoch": int(best_idx),
            metric1: best_row[metric1],
            metric2: best_row[metric2],
        }

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--metric1", required=True)
    parser.add_argument("--metric2", required=True)
    parser.add_argument(
        "--selection_metric", required=True, help="Metric used to select best epoch"
    )

    args = parser.parse_args()

    all_results = []

    for method in args.methods:
        method_dir = os.path.join(args.results_dir, method)

        if not os.path.exists(method_dir):
            print(f"Warning: {method_dir} not found, skipping.")
            continue

        method_results = process_method(
            args.dataset,
            method,
            method_dir,
            args.metric1,
            args.metric2,
            args.selection_metric,
        )

        all_results.extend(method_results)

    df = pd.DataFrame(all_results)

    output_file = f"full_results_{args.dataset}.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()

"""
python export_full_results_from_logs.py \
    --dataset bias_in_bios \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/bias_in_bios_baselines/dev \
    --methods bias_ensemble bpa george nsf \
    --metric1 test_overall \
    --metric2 test_worst_group_accuracy \
    --selection_metric test_worst_group_accuracy

    
python export_full_results_from_logs.py \
    --dataset speech_accent_archive \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/speech_accent_archive_baselines/dev \
    --methods bias_ensemble bpa george nsf \
    --metric1 test_overall \
    --metric2 test_worst_group_accuracy \
    --selection_metric test_worst_group_accuracy

python export_full_results_from_logs.py \
    --dataset jigsaw_toxic_comments \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/jigsaw_toxic_comments_baselines/dev \
    --methods bias_ensemble bpa george nsf \
    --metric1 val_accuracy \
    --metric2 test_accuracy \
    --selection_metric val_accuracy

python export_full_results_from_logs.py \
    --dataset urbansounds58 \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/urbansounds58_baselines/dev \
    --methods bias_ensemble bpa george nsf \
    --metric1 test_overall \
    --metric2 test_worst_group_accuracy \
    --selection_metric test_worst_group_accuracy

python export_full_results_from_logs.py \
    --dataset chexpert_nih \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/chexpert_nih_baselines/dev \
    --methods bias_ensemble bpa george nsf \
    --metric1 test_overall \
    --metric2 test_worst_group_accuracy \
    --selection_metric test_worst_group_accuracy


    run this to download from hua server the results for ucf 
    root@ITI-731:~# rsync -av   --include='*/'   --include='*.csv'   --exclude='*' isarridis@10.100.59.229:/home/isarridis/projects/vb-mitigator/output/ucf101_baselines/dev_in_out_scuba_swin_mixed /mnt/c/Users/gsarridis/Desktop/

    
python export_full_results_from_logs.py \
    --dataset ucf101 \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/ucf101_baselines/dev_in_out_scuba_swin_mixed \
    --methods bias_ensemble bpa george nsf \
    --metric1 test_accuracy \
    --metric2 test_accuracy \
    --selection_metric test_accuracy

    
and for the original methods 
python export_full_results_from_logs.py \
    --dataset ucf101 \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/ucf101_baselines/dev_in_out_scuba_swin_mixed \
    --methods erm debian jtt lff sd di bb end groupdro flac maviasb badd \
    --metric1 test_accuracy \
    --metric2 test_accuracy \
    --selection_metric test_accuracy

and for the original scuba exps
python export_full_results_from_logs.py \
    --dataset ucf101_org \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/ucf101_baselines/dev_in_out_scuba_swin \
    --methods erm debian jtt lff sd di bb end groupdro flac maviasb badd bias_ensemble bpa george\
    --metric1 test_accuracy \
    --metric2 test_accuracy \
    --selection_metric test_accuracy


and for natural images 
python export_full_results_from_logs.py \
    --dataset celeba \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/celeba_baselines/blonde \
    --methods bias_ensemble bpa george nsf \
    --metric1 test_overall \
    --metric2 test_worst_group_accuracy \
    --selection_metric test_worst_group_accuracy

python export_full_results_from_logs.py \
    --dataset urbancars \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/urbancars_baselines/dev \
    --methods bias_ensemble bpa george nsf \
    --metric1 test_overall \
    --metric2 test_worst_group_accuracy \
    --selection_metric test_worst_group_accuracy

python export_full_results_from_logs.py \
    --dataset waterbirds \
    --results_dir /mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/waterbirds_baselines/dev \
    --methods bias_ensemble bpa george nsf \
    --metric1 test_overall \
    --metric2 test_worst_group_accuracy \
    --selection_metric test_worst_group_accuracy
    """
