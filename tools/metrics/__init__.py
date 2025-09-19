from .acc import acc_dict, acc, acc_per_class
from .unb_bc_ba import unb_bc_ba_dict, unb_bc_ba
from .wg_ovr import wg_ovr_dict, wg_ovr
from .wg_ovr_tags import wg_ovr_tags, wg_ovr_tags_dict
from .wg_ovr_std import wg_ovr_std, wg_ovr_std_dict
from .wg_ovr_analytic import wg_ovr_analytic, wg_ovr_analytic_dict

metrics_dicts = {
    "acc": acc_dict,
    "unb_bc_ba": unb_bc_ba_dict,
    "wg_ovr": wg_ovr_dict,
    "wg_ovr_tags": wg_ovr_tags_dict,
    "acc_per_class": acc_dict,
    "wg_ovr_std": wg_ovr_std_dict,
    "wg_ovr_analytic": wg_ovr_analytic_dict,
}

get_performance = {
    "acc": acc,
    "unb_bc_ba": unb_bc_ba,
    "wg_ovr": wg_ovr,
    "wg_ovr_tags": wg_ovr_tags,
    "acc_per_class": acc_per_class,
    "wg_ovr_std": wg_ovr_std,
    "wg_ovr_analytic": wg_ovr_analytic,
}
