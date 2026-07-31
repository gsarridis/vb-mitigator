How to reproduce the paper results


collect all the results from logs to get full_results_<dataset>.csv using  _cross_domain_study/organized_revision/full_results_new_methods/export_full_results_from_logs.py


update _cross_domain_study/ranking_diou/comp_analysis/{performances,method_attr,data_attr}.csv based on new methods 

compute ranking csv
pre: full_results_<dataset>.csv
run: _cross_domain_study/ucf101/rankings/compute_rankings.py
out: _cross_domain_study/ucf101/rankings/ranking_<dataset>.csv

Table 7
pre: _cross_domain_study/ranking_diou/ranking_<dataset,..,>.csv
run: _cross_domain_study/ranking_diou/analysis.py
out: _cross_domain_study/ranking_diou/method_ranks_{visual,others}.csv


Figure 1
pre: _cross_domain_study/ranking_diou/comp_analysis/performances.csv (actually the results from tables)
run: _cross_domain_study/ranking_diou/comp_analysis/plot_binary_gains_drops.py
our: _cross_domain_study/ranking_diou/comp_analysis/plot_binary_gains_drops_05.pdf


Figure 2
pre: _cross_domain_study/ranking_diou/comp_analysis/{performances,method_attr,data_attr}.csv
run: _cross_domain_study/ranking_diou/comp_analysis/plot_bla_blu.py
our: _cross_domain_study/ranking_diou/comp_analysis/perf_vs_dataset_bias_labels_vs_not.pdf


Figure 3
pre: _cross_domain_study/ranking_diou/method_ranks_{visual,others}.csv
run: _cross_domain_study/ranking_diou/plot_ranking_visual_vs_others.py
out: _cross_domain_study/ranking_diou/ranking_comparison.pdf


Figure 4
pre: _cross_domain_study/ranking_diou/ranking_<dataset,..,>.csv 
run: _cross_domain_study/ranking_diou/comp_analysis/plot_ranks_bootstarp.py
out: _cross_domain_study/ranking_diou/comp_analysis/ranks_bootstarp_{all,other,visual}.pdf















download results from other server:
rsync -av   --include='*/'   --include='*.csv'   --exclude='*' isarridis@10.100.59.229:/home/isarridis/projects/vb-mitigator/output/ucf101_baselines/dev_in_out_scuba_swin /mnt/c/Users/gsarridis/Desktop/                                      
upload here

collect all the results from logs to get full_results_<dataset>.csv using  _cross_domain_study/organized_revision/full_results_new_methods/export_full_results_from_logs.py
DONE

write a script for geting the mean values, std, for the new methods - add them to the paper tables 
Done 


compute ranking csv
pre: full_results_<dataset>.csv modify to work for ucf101 
run: _cross_domain_study/ucf101/rankings/compute_rankings.py
out: _cross_domain_study/ucf101/rankings/ranking_<dataset>.csv

Table 7
pre: _cross_domain_study/ranking_diou/ranking_<dataset,..,>.csv
run: _cross_domain_study/ranking_diou/analysis.py
out: _cross_domain_study/ranking_diou/method_ranks_{visual,others}.csv


Figure 1
pre: _cross_domain_study/ranking_diou/comp_analysis/performances.csv (actually the results from tables)
run: _cross_domain_study/ranking_diou/comp_analysis/plot_binary_gains_drops.py
our: _cross_domain_study/ranking_diou/comp_analysis/plot_binary_gains_drops_05.pdf


Figure 2
pre: _cross_domain_study/ranking_diou/comp_analysis/{performances,method_attr,data_attr}.csv
run: _cross_domain_study/ranking_diou/comp_analysis/plot_bla_blu.py
our: _cross_domain_study/ranking_diou/comp_analysis/perf_vs_dataset_bias_labels_vs_not.pdf


Figure 3
pre: _cross_domain_study/ranking_diou/method_ranks_{visual,others}.csv
run: _cross_domain_study/ranking_diou/plot_ranking_visual_vs_others.py
out: _cross_domain_study/ranking_diou/ranking_comparison.pdf


Figure 4
pre: _cross_domain_study/ranking_diou/ranking_<dataset,..,>.csv 
run: _cross_domain_study/ranking_diou/comp_analysis/plot_ranks_bootstarp.py
out: _cross_domain_study/ranking_diou/comp_analysis/ranks_bootstarp_{all,other,visual}.pdf


results for sensitivity analysis node

update _cross_domain_study/ranking_diou/comp_analysis/{performances,method_attr,data_attr}.csv based on new methods 

create categories - and plots for these categories 


Total ERM pairs found: 14
mavias           0.000125
badd             0.003592
flac             0.037729
bpa              0.092006
di               0.162037
bb               0.399941
george           0.711643
lff              0.729748
debian           0.879864
end              0.901884
sd               0.937572
jtt              0.998145
groupdro         0.999997
bias_ensemble    0.999999
dtype: float64

=== Average Ranks (lower is better) ===
method
mavias            2.333333
badd              3.888889
flac              5.111111
bpa               5.777778
di                6.222222
bb                7.111111
george            7.888889
lff               8.000000
debian            8.555556
end               8.666667
sd                8.888889
jtt              10.000000
groupdro         11.000000
bias_ensemble    11.111111
erm              12.444444
dtype: float64


statistical test for other domains only 
Total ERM pairs found: 14
maviasb          0.023783
badd             0.193799
flac             0.225275
bpa              0.714693
di               0.919873
lff              0.968844
bb               0.988495
debian           0.998104
end              0.998983
george           0.999270
sd               0.999892
groupdro         0.999984
bias_ensemble    1.000000
jtt              1.000000
dtype: float64


B. By how the bias signal is acquired (the most informative new axis)
B1 — Bias labels supplied by the user
GroupDRO, DI, EnD, BB, BAdd
B2 — Bias inferred from a separate auxiliary/bias-capturing model
LfF (GCE-trained biased classifier), FLAC (bias-capturing features for MI minimization), BAdd (uses a separately trained bias-capturing model — note BAdd straddles A1 and B2 because the auxiliary is itself trained on the bias label), MAVias (originally a vision-language tagger; in your paper, the same bias-capturing features as FLAC/BAdd), DebiAN (an alternating auxiliary that searches for Equal Opportunity violations), BiasEnsemble (an ensemble of N=5 GCE-biased models that votes on which samples are bias-aligned)
B3 — Bias inferred from the main model's own training dynamics
JTT (misclassified samples after a short ERM warmup), Sebra (rounds in which a sample gets "learned" via UpweightedTrainingLoss), BPA (per-cluster loss share from the main model during training)
B4 — Bias inferred from clustering in feature space
George (per-class GMM/k-means + UMAP on penultimate features), BPA (per-class cosine k-means on a frozen base model's features), NSF (nearest-prototype on features from a frozen ERM)
B5 — No explicit bias signal at all (purely architectural / regularization)
SD (spectral decoupling regularizer on logits), ERM (baseline)
Note: B2/B3/B4 are not mutually exclusive. BPA and NSF rely on B4 (clustering/prototypes), but BPA also continuously updates per-sample weights using B3 (training-dynamics signals from the main model).

C. By what the algorithm does once it has the bias signal
C1 — Per-sample loss reweighting
GroupDRO (worst-group weighted loss), LfF (GCE-derived sample weights), JTT (2× upweight on the error set), DebiAN (continuous per-sample weights from an auxiliary), BPA (cluster-size × cluster-loss-share weights with EMA smoothing), Sebra Stage 1 (p_y^β upweighting of easy samples — opposite direction to LfF), BiasEnsemble (LfF-style sample selection inside the loop)
C2 — Logit-space adjustment
BB (subtracts log p(α|y) from logits to remove the bias prior), SD (L2 regularizer on logits)
C3 — Feature-space manipulation
EnD (orthogonality + parallelism constraints on the Gram matrix), BAdd (adds bias-capturing features into the main forward pass), MAVias (alignment loss between biased and main logits), FLAC (mutual information minimization between features and bias-capturing features), NSF (learned affine feature transformation that pulls features toward neutralized class centers)
C4 — Architectural separation
DI (separate classification head per bias domain)
C5 — Contrastive / metric learning
Sebra Stage 2 (supervised contrastive loss with rank-based positives/negatives)
C6 — Sample selection / re-grouping
JTT (select error set, retrain), BiasEnsemble (select samples by ensemble agreement, then run LfF on the rest), George (re-group by cluster, then run GroupDRO over discovered groups)