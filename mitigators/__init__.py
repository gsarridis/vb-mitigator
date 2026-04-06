from .erm import ERMTrainer
from .flac import FLACTrainer
from .badd import BAddTrainer
from .mavias import MAVIASTrainer
from .groupdro import GroupDROTrainer
from .debian import DebiANTrainer
from .domain_independent import DomainIndependentTrainer
from .spectral_decouple import SpectralDecoupleTrainer
from .lff import LfFTrainer
from .bb import BBTrainer
from .end import EndTrainer
from .erm_tags import ERMTagsTrainer
from .flacb import FLACBTrainer
from .jtt import JTTTrainer
from .softcon import SoftConTrainer
from .erm_bcc import ERMBCCTrainer
from .maviasb import MAVIASBTrainer
from .flac_aida import FLACAIDATrainer
from .recurrent import RecurrentTrainer
from .erm_dev import ERMDevTrainer
from .sae import SAETrainer
from .sae_steering import SAESteeringTrainer
from .sae_neuron_classifier import SAENeuronClassifierTrainer
from .sae_augmentation import SAEAugmentationTrainer
from .sae_weighted_classifier import SAEWeightedClassifierTrainer
from .zero_shot_vlm import ZeroShotVLMTrainer
from .zero_shot_vlm_sae import SAEFilteredZeroShotTrainer
from .zero_shot_vlm_sae_text import SAETextAlignedZeroShotTrainer

# from .tag_supervised_sae import TagSupervisedSAETrainer
from .tag_sae import TagSAETrainer
from .sae_debias_cls import SAEDebiasClassifierTrainer
from .tag_only_sae import TagOnlySAETrainer
from .tag_only_debias_cls import TagOnlyDebiasClassifierTrainer
from .tag_only_sae_v2 import TagOnlySAEv2Trainer
from .tag_extraction import TagExtractionTrainer
from .clip_distillation import CLIPDistillationTrainer
from .openclip_classifier import OpenCLIPClassifierTrainer

method_to_trainer = {
    "erm": ERMTrainer,
    "flac": FLACTrainer,
    "flacb": FLACBTrainer,
    "badd": BAddTrainer,
    "mavias": MAVIASTrainer,
    "groupdro": GroupDROTrainer,
    "debian": DebiANTrainer,
    "di": DomainIndependentTrainer,
    "sd": SpectralDecoupleTrainer,
    "lff": LfFTrainer,
    "bb": BBTrainer,
    "end": EndTrainer,
    "erm_tags": ERMTagsTrainer,
    "jtt": JTTTrainer,
    "softcon": SoftConTrainer,
    "erm_bcc": ERMBCCTrainer,
    "maviasb": MAVIASBTrainer,
    "flac_aida": FLACAIDATrainer,
    "recurrent": RecurrentTrainer,
    "erm_dev": ERMDevTrainer,
    "sae": SAETrainer,
    "sae_steering": SAESteeringTrainer,
    "sae_neuron_classifier": SAENeuronClassifierTrainer,
    "sae_augmentation": SAEAugmentationTrainer,
    "sae_weighted_classifier": SAEWeightedClassifierTrainer,
    "zero_shot_vlm": ZeroShotVLMTrainer,
    "sae_filtered_zero_shot": SAEFilteredZeroShotTrainer,
    "sae_text_aligned_zero_shot": SAETextAlignedZeroShotTrainer,
    # "tag_supervised_sae": TagSupervisedSAETrainer,
    "tag_sae": TagSAETrainer,
    "sae_debias_cls": SAEDebiasClassifierTrainer,
    "tag_only_sae": TagOnlySAETrainer,
    "tag_only_debias_cls": TagOnlyDebiasClassifierTrainer,
    "tag_only_sae_v2": TagOnlySAEv2Trainer,
    "tag_extraction": TagExtractionTrainer,
    "clip_distillation": CLIPDistillationTrainer,
    "openclip_classifier": OpenCLIPClassifierTrainer,
}
