import sys
from yacs.config import CfgNode as CN
from tools.utils import log_msg


def show_cfg(cfg, logger):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.MODEL = cfg.MODEL
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.MITIGATOR = cfg.MITIGATOR
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    if cfg.MITIGATOR.TYPE in cfg:
        dump_cfg.update({cfg.MITIGATOR.TYPE: cfg.get(cfg.MITIGATOR.TYPE)})
    log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO", logger)


CFG = CN()

# Experiment
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = "biased_mnist"
CFG.EXPERIMENT.NAME = "dev"
CFG.EXPERIMENT.TAG = "vanilla"
CFG.EXPERIMENT.GPU = "cuda:0"  # or cpu
CFG.EXPERIMENT.SEED = 1
CFG.EXPERIMENT.EVAL = False
CFG.EXPERIMENT.EPOCH_STEPS = sys.maxsize
CFG.EXPERIMENT.EVAL_STEP = 1
CFG.EXPERIMENT.PLACEHOLDER_STEPS = sys.maxsize
CFG.EXPERIMENT.PROGRESS_BAR = True
# Model
CFG.MODEL = CN()
CFG.MODEL.TYPE = "resnet"
CFG.MODEL.PRETRAINED = True
CFG.MODEL.FREEZE_BACKBONE = False
CFG.MODEL.PATH = "best"

# Solver
CFG.SOLVER = CN()
CFG.SOLVER.BATCH_SIZE = 64
CFG.SOLVER.EPOCHS = 240
CFG.SOLVER.LR = 0.001
CFG.SOLVER.WEIGHT_DECAY = 0.0001
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.TYPE = "SGD"
CFG.SOLVER.CRITERION = "CE"
CFG.SOLVER.SCHEDULER = CN()
CFG.SOLVER.SCHEDULER.TYPE = "MultiStepLR"
CFG.SOLVER.SCHEDULER.LR_DECAY_STAGES = [150, 180, 210]
CFG.SOLVER.SCHEDULER.LR_DECAY_RATE = 0.1
CFG.SOLVER.SCHEDULER.LINEAR_WARMUP = 0.0
# Log
CFG.LOG = CN()
CFG.LOG.TENSORBOARD_FREQ = 500
CFG.LOG.SAVE_CHECKPOINT_FREQ = 40
CFG.LOG.PREFIX = "./output"
CFG.LOG.WANDB = False
CFG.LOG.TRAIN_PERFORMANCE = False
CFG.LOG.SAVE_CRITERION = "test"

CFG.METRIC = "acc"
CFG.METRIC_TAGS = "wg_ovr_tags"
# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "cifar100"
CFG.DATASET.ROOT = "./data"
CFG.DATASET.NUM_WORKERS = 8
CFG.DATASET.TEST = CN()
CFG.DATASET.TEST.BATCH_SIZE = 64
CFG.DATASET.BIASES = [
    "color",
    "blonde",
    "makeup",
    "race",
    "age",
    "background",
    "foreground",
]


# Dataset specific arguements
CFG.DATASET.BIASED_MNIST = CN()
CFG.DATASET.BIASED_MNIST.RATIO = 0
CFG.DATASET.BIASED_MNIST.CORR = 0.99
CFG.DATASET.BIASED_MNIST.ROOT = "./data/biased_mnist"
CFG.DATASET.BIASED_MNIST.IMAGE_SIZE = 28


# Dataset specific arguements
CFG.DATASET.FB_BIASED_MNIST = CN()
CFG.DATASET.FB_BIASED_MNIST.RATIO = 0
CFG.DATASET.FB_BIASED_MNIST.CORR_BG = 0.9
CFG.DATASET.FB_BIASED_MNIST.CORR_FG = 0.9
CFG.DATASET.FB_BIASED_MNIST.ROOT = "./data/fb_biased_mnist"
CFG.DATASET.FB_BIASED_MNIST.IMAGE_SIZE = 28


# Dataset specific arguements
CFG.DATASET.UTKFACE = CN()
CFG.DATASET.UTKFACE.BIAS = "race"  # or age
CFG.DATASET.UTKFACE.ROOT = "./data/utkface"
CFG.DATASET.UTKFACE.RATIO = 0
CFG.DATASET.UTKFACE.IMAGE_SIZE = 64
CFG.DATASET.UTKFACE.BIAS_ALIGNED = [(1, 1), (0, 0)]

CFG.DATASET.WATERBIRDS = CN()
CFG.DATASET.WATERBIRDS.ROOT = "./data/waterbirds"
CFG.DATASET.WATERBIRDS.IMAGE_SIZE = 224

CFG.DATASET.CELEBA = CN()
CFG.DATASET.CELEBA.ROOT = "./data/celeba"
CFG.DATASET.CELEBA.BIAS = "gender"
CFG.DATASET.CELEBA.TARGET = "blonde"  # or "makeup"
CFG.DATASET.CELEBA.RATIO = 0
CFG.DATASET.CELEBA.IMAGE_SIZE = 224
CFG.DATASET.CELEBA.BIAS_ALIGNED = [(0, 0), (1, 1)]

CFG.DATASET.IMAGENET9 = CN()
CFG.DATASET.IMAGENET9.ROOT_IMAGENET = "/home/isarridis/datasets/imagenet/"  # you should manually download ImageNet and define the root directory here.
CFG.DATASET.IMAGENET9.ROOT_IMAGENET_BG = "./data/imagenet9"
CFG.DATASET.IMAGENET9.IMAGE_SIZE = 224
CFG.DATASET.IMAGENET9.BIAS = "unknown"
CFG.DATASET.IMAGENET9.BENCHMARK_VAL = "mixed_rand"  # choices: mixed_rand, mixed_next, mixed_same, no_fg, only_bg_b, only_bg_t, only_fg, original
CFG.DATASET.IMAGENET9.BENCHMARK_TEST = "original"  # choices: mixed_rand, mixed_next, mixed_same, no_fg, only_bg_b, only_bg_t, only_fg, original

CFG.DATASET.UCF101 = CN()
CFG.DATASET.UCF101.VIDEO_PATH = "/mnt/cephfs/home/common/datasets/UCF101/UCF-101-jpg"
CFG.DATASET.UCF101.ANNOTATION_PATH = (
    "/mnt/cephfs/home/common/datasets/UCF101/ucf101_01.json"
)
CFG.DATASET.UCF101.ANNOTATION_PATH_SCUBA = "/mnt/cephfs/home/gsarridis/projects/StillMix/main_network/mmaction2/data/UCF101-24/lists_generated/testlist01.txt"
# CFG.DATASET.UCF101.VIDEO_PATH_SCUBA = "/mnt/cephfs/home/gsarridis/projects/StillMix/main_network/mmaction2/data/UCF101-24/generated"
CFG.DATASET.UCF101.VIDEO_PATH_SCUBA = (
    "/mnt/cephfs/home/common/datasets/UCF101/scuba/generated_videos"
)
CFG.DATASET.UCF101.TEST_BENCHMARK = "scuba"

CFG.DATASET.UCF101.BIAS_TYPE = "indoor_outdoor"
CFG.DATASET.UCF101.BIAS_TH = 0.00

CFG.DATASET.JIGSAW_TOXIC_COMMENTS = CN()
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.ROOT = "./data/jigsaw_toxic_comments/data"
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.BIAS = "bias"
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.TARGET = "identity_hate"
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.TRAIN_SET = "train_identity_hate_biases.csv"
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.VAL_SET = "test_identity_hate_biases.csv"
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.TEST_SET = "test_chatgpt_biases.csv"
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.TEXT_ENCODER = "all-MiniLM-L6-v2"
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.CLASSIFICATION_HEAD = CN()
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.CLASSIFICATION_HEAD.INPUT_DIM = 368
CFG.DATASET.JIGSAW_TOXIC_COMMENTS.CLASSIFICATION_HEAD.HIDDEN_DIM = 256


CFG.DATASET.BIAS_IN_BIOS = CN()
CFG.DATASET.BIAS_IN_BIOS.ROOT = "./data/bias_in_bios"
CFG.DATASET.BIAS_IN_BIOS.BIAS = "gender"
CFG.DATASET.BIAS_IN_BIOS.TARGET = "profession"
CFG.DATASET.BIAS_IN_BIOS.TEXT_ENCODER = "all-MiniLM-L6-v2"
CFG.DATASET.BIAS_IN_BIOS.CLASSIFICATION_HEAD = CN()
CFG.DATASET.BIAS_IN_BIOS.CLASSIFICATION_HEAD.INPUT_DIM = 368
CFG.DATASET.BIAS_IN_BIOS.CLASSIFICATION_HEAD.HIDDEN_DIM = 256

CFG.DATASET.SPEECH_ACCENT_ARCHIVE = CN()
CFG.DATASET.SPEECH_ACCENT_ARCHIVE.ROOT = "./data/speech_accent_archive"
CFG.DATASET.SPEECH_ACCENT_ARCHIVE.AUDIO_ENCODER = "RealAudioEncoder"
CFG.DATASET.SPEECH_ACCENT_ARCHIVE.CLASSIFICATION_HEAD = CN()
CFG.DATASET.SPEECH_ACCENT_ARCHIVE.CLASSIFICATION_HEAD.INPUT_DIM = 768
CFG.DATASET.SPEECH_ACCENT_ARCHIVE.CLASSIFICATION_HEAD.HIDDEN_DIM = 256

CFG.DATASET.URBANSOUNDS = CN()
CFG.DATASET.URBANSOUNDS.ROOT = "./data/urbansounds/UrbanSound8K"
CFG.DATASET.URBANSOUNDS.AUDIO_ENCODER = "MFCCEncoder"
CFG.DATASET.URBANSOUNDS.CLASSIFICATION_HEAD = CN()
CFG.DATASET.URBANSOUNDS.CLASSIFICATION_HEAD.INPUT_DIM = 768
CFG.DATASET.URBANSOUNDS.CLASSIFICATION_HEAD.HIDDEN_DIM = 256

CFG.DATASET.IMAGENET = CN()
CFG.DATASET.IMAGENET.ROOT = "/home/isarridis/datasets/imagenet/"  # you should manually download ImageNet and define the root directory here.
CFG.DATASET.IMAGENET.IMAGE_SIZE = 224
CFG.DATASET.IMAGENET.BIAS = "unknown"

CFG.DATASET.CIFAR10 = CN()
CFG.DATASET.CIFAR10.ROOT = "./data/cifar10"  # you should manually download ImageNet and define the root directory here.
CFG.DATASET.CIFAR10.IMAGE_SIZE = 32
CFG.DATASET.CIFAR10.BIAS = "unknown"

CFG.DATASET.CIFAR100 = CN()
CFG.DATASET.CIFAR100.ROOT = "./data/cifar100"  # you should manually download ImageNet and define the root directory here.
CFG.DATASET.CIFAR100.IMAGE_SIZE = 32
CFG.DATASET.CIFAR100.BIAS = "unknown"

CFG.DATASET.STANFORD_DOGS = CN()
CFG.DATASET.STANFORD_DOGS.ROOT = "./data/stanford-dogs-dataset"  # you should manually download ImageNet and define the root directory here.
CFG.DATASET.STANFORD_DOGS.IMAGE_SIZE = 224
CFG.DATASET.STANFORD_DOGS.BIAS = "unknown"

CFG.DATASET.URBANCARS = CN()
CFG.DATASET.URBANCARS.ROOT = "./data/urbancars"  # you should manually download ImageNet and define the root directory here.
CFG.DATASET.URBANCARS.IMAGE_SIZE = 224
CFG.DATASET.URBANCARS.BIAS = "bg_cooc_obj"

# MITIGATOR
CFG.MITIGATOR = CN()
CFG.MITIGATOR.TYPE = "erm"  # Vanilla as default


CFG.MITIGATOR.GCOS = CN()
CFG.MITIGATOR.GCOS.LAMBDA = 1.0


CFG.MITIGATOR.SELFKD = CN()
CFG.MITIGATOR.SELFKD.TEACHER_PATH = "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/output/utkface_baselines/race_r18/erm/best"

# MAVias CFG
CFG.MITIGATOR.MAVIAS = CN()
CFG.MITIGATOR.MAVIAS.TAGGING_MODEL = CN()
CFG.MITIGATOR.MAVIAS.TAGGING_MODEL.TYPE = "ram"
CFG.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE = 384
CFG.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE = 16
CFG.MITIGATOR.MAVIAS.ENCODER = CN()
CFG.MITIGATOR.MAVIAS.ENCODER.TYPE = "clip"
CFG.MITIGATOR.MAVIAS.ENCODER.SIZE = 768
CFG.MITIGATOR.MAVIAS.LLM = CN()
CFG.MITIGATOR.MAVIAS.LLM.TYPE = "llama3"
CFG.MITIGATOR.MAVIAS.LLM.BATCH_SIZE = 100
CFG.MITIGATOR.MAVIAS.LOSS = CN()
CFG.MITIGATOR.MAVIAS.LOSS.ALPHA = 0.1
CFG.MITIGATOR.MAVIAS.LOSS.LAMBDA = 0.8
CFG.MITIGATOR.MAVIAS.PROJNET = CN()
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM = CN()

CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM.LR = 0.001
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM.WEIGHT_DECAY = 5e-4
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM.MOMENTUM = 0.9
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM.TYPE = "SGD"


CFG.MITIGATOR.MAVIASB = CN()
CFG.MITIGATOR.MAVIASB.LOSS = CN()
CFG.MITIGATOR.MAVIASB.LOSS.ALPHA = 0.1
CFG.MITIGATOR.MAVIASB.LOSS.LAMBDA = 0.8
CFG.MITIGATOR.MAVIASB.PROJNET = CN()
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM = CN()
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM.LR = 0.001
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM.WEIGHT_DECAY = 5e-4
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM.MOMENTUM = 0.9
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM.TYPE = "SGD"
CFG.MITIGATOR.MAVIASB.BCC_PATH = ""


CFG.MITIGATOR.MHMAVIAS = CN()
CFG.MITIGATOR.MHMAVIAS.TAGGING_MODEL = CN()
CFG.MITIGATOR.MHMAVIAS.TAGGING_MODEL.TYPE = "ram"
CFG.MITIGATOR.MHMAVIAS.TAGGING_MODEL.IMG_SIZE = 384
CFG.MITIGATOR.MHMAVIAS.TAGGING_MODEL.BATCH_SIZE = 16
CFG.MITIGATOR.MHMAVIAS.ENCODER = CN()
CFG.MITIGATOR.MHMAVIAS.ENCODER.TYPE = "clip"
CFG.MITIGATOR.MHMAVIAS.ENCODER.SIZE = 768
CFG.MITIGATOR.MHMAVIAS.LLM = CN()
CFG.MITIGATOR.MHMAVIAS.LLM.TYPE = "llama3"
CFG.MITIGATOR.MHMAVIAS.LLM.BATCH_SIZE = 100
CFG.MITIGATOR.MHMAVIAS.LOSS = CN()
CFG.MITIGATOR.MHMAVIAS.LOSS.ALPHA = 0.1
CFG.MITIGATOR.MHMAVIAS.LOSS.LAMBDA = 0.8


CFG.MITIGATOR.ARC = CN()
CFG.MITIGATOR.ARC.BIAS_DISCOVERY_EPOCHS = 5
CFG.MITIGATOR.ARC.DETECTION_OPTIMIZER = CN()
CFG.MITIGATOR.ARC.DETECTION_OPTIMIZER.LR = 0.001
CFG.MITIGATOR.ARC.DETECTION_OPTIMIZER.MOMENTUM = 0.9
CFG.MITIGATOR.ARC.DETECTION_OPTIMIZER.WEIGHT_DECAY = 0.0001

CFG.MITIGATOR.JTT = CN()
CFG.MITIGATOR.JTT.BIAS_DISCOVERY_EPOCHS = 50
CFG.MITIGATOR.JTT.UPWEIGHT = 100
CFG.MITIGATOR.JTT.BCC_PATH = ""

# FLAC CFG
CFG.MITIGATOR.FLAC = CN()
CFG.MITIGATOR.FLAC.LOSS = CN()
CFG.MITIGATOR.FLAC.LOSS.ALPHA = 110.0
CFG.MITIGATOR.FLAC.LOSS.DELTA = 1.0
CFG.MITIGATOR.FLAC.LOSS.CE_WEIGHT = 1.0
CFG.MITIGATOR.FLAC.BCC_PATH = ""

# FLAC-B CFG
CFG.MITIGATOR.FLACB = CN()
CFG.MITIGATOR.FLACB.BCC_PATH = ""
CFG.MITIGATOR.FLACB.LOSS = CN()
CFG.MITIGATOR.FLACB.LOSS.ALPHA = 110.0
CFG.MITIGATOR.FLACB.LOSS.DELTA = 1.0
CFG.MITIGATOR.FLACB.LOSS.CE_WEIGHT = 1.0

# SOFTCON CFG
CFG.MITIGATOR.SOFTCON = CN()
CFG.MITIGATOR.SOFTCON.BCC_PATH = ""
CFG.MITIGATOR.SOFTCON.WEIGHT = 1000

# BADD CFG
CFG.MITIGATOR.BADD = CN()
CFG.MITIGATOR.BADD.M = 1.0
CFG.MITIGATOR.BADD.BCC_PATH = ""


# GROUPDRO CFG
CFG.MITIGATOR.GROUPDRO = CN()
CFG.MITIGATOR.GROUPDRO.ROBUST_STEP_SIZE = 0.01

# SPECTRAL DECOUPLE
CFG.MITIGATOR.SD = CN()
CFG.MITIGATOR.SD.COEF = 0.1


# END
CFG.MITIGATOR.END = CN()
CFG.MITIGATOR.END.ALPHA = 1
CFG.MITIGATOR.END.BETA = 1
CFG.MITIGATOR.END.WEIGHT = 1

# SAE
CFG.MITIGATOR.SAE = CN()

# Path to pretrained checkpoint (e.g., ERM model)
CFG.MITIGATOR.SAE.CHECKPOINT_PATH = ""

# SAE Architecture Type
# Options: "standard", "topk", "batch_topk", "jumprelu"
CFG.MITIGATOR.SAE.TYPE = "standard"

# Expansion factor: dictionary_size = feature_dim * expansion_factor
# Typical values: 4, 8, 16, 32, 64
CFG.MITIGATOR.SAE.EXPANSION_FACTOR = 8

# Training Parameters
CFG.MITIGATOR.SAE.STEPS = 10000
CFG.MITIGATOR.SAE.BATCH_SIZE = 256
CFG.MITIGATOR.SAE.LR = 0.001
CFG.MITIGATOR.SAE.WARMUP_STEPS = 500

# Standard SAE Parameters
CFG.MITIGATOR.SAE.L1_PENALTY = 0.001
CFG.MITIGATOR.SAE.RESAMPLE_STEPS = 2500  # Set to 0 to disable

# TopK SAE Parameters
CFG.MITIGATOR.SAE.K = 32
CFG.MITIGATOR.SAE.AUXK_ALPHA = 0.03

# JumpReLU SAE Parameters
CFG.MITIGATOR.SAE.BANDWIDTH = 0.001
CFG.MITIGATOR.SAE.SPARSITY_PENALTY = 0.1

# Analysis Parameters
CFG.MITIGATOR.SAE.TOP_K_IMAGES = 16  # Top-k images per neuron
CFG.MITIGATOR.SAE.NUM_VISUALIZE = 100  # Number of neurons to visualize

# Clustering Parameters
CFG.MITIGATOR.SAE.CLUSTERING = CN()
CFG.MITIGATOR.SAE.CLUSTERING.ENABLED = True
CFG.MITIGATOR.SAE.CLUSTERING.N_CLUSTERS = -1  # -1 means use number of data groups
CFG.MITIGATOR.SAE.CLUSTERING.PERPLEXITY = 30  # t-SNE perplexity
CFG.MITIGATOR.SAE.CLUSTERING.UMAP_NEIGHBORS = 15  # UMAP n_neighbors
CFG.MITIGATOR.SAE.CLUSTERING.UMAP_MIN_DIST = 0.1  # UMAP min_dist


CFG.MODEL.OPENCLIP = CN()

# Model architecture
# Options: 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336',
#          'ViT-H-14', 'ViT-G-14', 'ViT-bigG-14', 'RN50', 'RN101', etc.
CFG.MODEL.OPENCLIP.ARCH = "ViT-B-32"

# Pretrained weights
# Options: 'openai', 'laion2b_s34b_b79k', 'laion400m_s13b_b51k',
#          'datacomp_xl_s13b_b90k', etc.
# Use empty string "" for random initialization
CFG.MODEL.OPENCLIP.PRETRAINED = "openai"

# Whether to use the CLIP projection layer or raw transformer output
CFG.MODEL.OPENCLIP.USE_PROJECTION = True

# Classification head type: 'linear' or 'mlp'
CFG.MODEL.OPENCLIP.HEAD_TYPE = "linear"

# MLP head hidden dimension (only used if HEAD_TYPE='mlp')
CFG.MODEL.OPENCLIP.HEAD_HIDDEN_DIM = 512

# MLP head dropout rate (only used if HEAD_TYPE='mlp')
CFG.MODEL.OPENCLIP.HEAD_DROPOUT = 0.1

# Image size (model-specific, but can be overridden)
CFG.MODEL.OPENCLIP.IMAGE_SIZE = 224


CFG.MITIGATOR.SAE_STEERING = CN()

# Path to trained SAE checkpoint (ae.pt)
CFG.MITIGATOR.SAE_STEERING.SAE_CHECKPOINT_PATH = ""

# Path to SAE analysis results (analysis_results.json from SAE trainer)
CFG.MITIGATOR.SAE_STEERING.SAE_ANALYSIS_PATH = ""

# Purity threshold: only keep neurons with class purity >= this value
# 1.0 = only keep neurons where ALL top-k images are from the same class
# 0.9 = keep neurons where >= 90% of top-k images are from the same class
# 0.5 = keep neurons where >= 50% of top-k images are from the same class (majority)
CFG.MITIGATOR.SAE_STEERING.PURITY_THRESHOLD = 1.0

# Value to use for suppressed (polysemantic) neurons
# 0.0 = completely deactivate these neurons
# 0.5 = reduce their contribution by half
# Negative values can be used to invert their effect (experimental)
CFG.MITIGATOR.SAE_STEERING.SUPPRESSION_VALUE = 0.0

# Whether to use sparse SAE features directly for classification
# False = decode back to original feature space, then classify
# True = classify directly from sparse (and masked) SAE features
CFG.MITIGATOR.SAE_STEERING.USE_SAE_FEATURES = False

# Whether to decode SAE features back to original space
# True = decode after masking (for USE_SAE_FEATURES=False)
# This affects the input dimension to the classification head
CFG.MITIGATOR.SAE_STEERING.USE_DECODE = True


CFG.MITIGATOR.SAE_NEURON_CLASSIFIER = CN()

# Path to trained SAE checkpoint (ae.pt)
CFG.MITIGATOR.SAE_NEURON_CLASSIFIER.SAE_CHECKPOINT_PATH = ""

# Path to SAE analysis results (analysis_results.json)
CFG.MITIGATOR.SAE_NEURON_CLASSIFIER.SAE_ANALYSIS_PATH = ""

# Purity threshold: only use neurons with class purity >= this value
# 1.0 = only neurons where ALL top-k images are from same class
CFG.MITIGATOR.SAE_NEURON_CLASSIFIER.PURITY_THRESHOLD = 1.0

# Activation threshold: minimum activation value to consider a neuron as "active"
# Neurons with activation below this threshold are treated as 0 (not contributing)
# This filters out weak/noisy activations that may not be meaningful
CFG.MITIGATOR.SAE_NEURON_CLASSIFIER.ACTIVATION_THRESHOLD = 0.1

# How to aggregate neuron activations for each class
# 'sum' = sum all activations (favors classes with more neurons)
# 'mean' = average activations (normalizes for neuron count)
# 'max' = max activation (only strongest neuron matters)
CFG.MITIGATOR.SAE_NEURON_CLASSIFIER.AGGREGATION = "sum"

# Temperature for softmax (higher = softer predictions)
# Only affects probability distribution, not argmax prediction
CFG.MITIGATOR.SAE_NEURON_CLASSIFIER.TEMPERATURE = 1.0

# Whether to learn optimal temperature on training set
# If True, runs a few epochs to optimize temperature via cross-entropy
CFG.MITIGATOR.SAE_NEURON_CLASSIFIER.LEARN_TEMPERATURE = False


CFG.MITIGATOR.SAE_AUGMENTATION = CN()

# Path to trained SAE checkpoint (ae.pt)
CFG.MITIGATOR.SAE_AUGMENTATION.SAE_CHECKPOINT_PATH = ""

# Path to SAE analysis results (analysis_results.json)
CFG.MITIGATOR.SAE_AUGMENTATION.SAE_ANALYSIS_PATH = ""

# Purity threshold: only use neurons with class purity >= this value
# 1.0 = only neurons where ALL top-k images are from same class
CFG.MITIGATOR.SAE_AUGMENTATION.PURITY_THRESHOLD = 1.0

# Probability of applying augmentation to each sample (0.0 to 1.0)
# 0.0 = no augmentation, 1.0 = always augment
CFG.MITIGATOR.SAE_AUGMENTATION.AUGMENT_PROB = 0.5

# Percentage of ACTIVE class neurons to DEACTIVATE (set to 0)
# 0.2 = deactivate 20% of currently active class-specific neurons
CFG.MITIGATOR.SAE_AUGMENTATION.DEACTIVATE_PCT = 0.2

# Percentage of INACTIVE class neurons to ACTIVATE (add activation_value)
# 0.3 = activate 30% of currently inactive class-specific neurons
CFG.MITIGATOR.SAE_AUGMENTATION.ACTIVATE_PCT = 0.3

# Constant activation value to add to activated neurons
# Higher values = stronger augmentation effect
CFG.MITIGATOR.SAE_AUGMENTATION.ACTIVATION_VALUE = 1.0

# Input to classifier: 'sparse' or 'decoded'
# 'sparse' = classify directly from SAE latents (larger dim, more interpretable)
# 'decoded' = decode back to original feature space first
CFG.MITIGATOR.SAE_AUGMENTATION.CLASSIFIER_INPUT = "sparse"


# ============================================
# SAE Weighted Classifier Configuration
# Add this section to configs/cfg.py
# ============================================

CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER = CN()

# Path to trained SAE checkpoint (ae.pt)
CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER.SAE_CHECKPOINT_PATH = ""

# Path to SAE analysis results (analysis_results.json)
CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER.SAE_ANALYSIS_PATH = ""

# Purity threshold for including neurons
# Can be lower than 1.0 since we weight by purity anyway
CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER.PURITY_THRESHOLD = 0.8

# Weight mode: 'purity' or 'learnable'
# 'purity' = weight by neuron purity (no learning, training-free)
# 'learnable' = learn per-neuron weights (minimal parameters)
CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER.WEIGHT_MODE = "learnable"

# If True and mode='learnable', initialize weights with purity values
# Otherwise initialize with ones
CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER.INIT_WITH_PURITY = True

# Initial temperature for score scaling
CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER.TEMPERATURE = 1.0

# Whether to learn the temperature parameter
CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER.LEARN_TEMPERATURE = True

# Whether to learn per-class bias terms
# Helps compensate for class imbalance or unequal neuron counts
CFG.MITIGATOR.SAE_WEIGHTED_CLASSIFIER.LEARN_CLASS_BIAS = True


# ============================================
# VLM Encoder Configuration Defaults
# Add to configs/cfg.py
# ============================================

# ---- VLM Encoder Settings ----
CFG.MODEL.VLM = CN()

# Encoder type: "openclip", "siglip", "perception_encoder"
CFG.MODEL.VLM.ENCODER_TYPE = "openclip"

# Model architecture name
# OpenCLIP: "ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14", "ViT-G-14"
# SigLIP: "siglip-base-patch16-224", "siglip-so400m-patch14-384"
# PE: "PE-Core-B16-224", "PE-Core-L14-336", "PE-Core-G14-448"
CFG.MODEL.VLM.MODEL_NAME = "ViT-B-32"

# Pretrained weights source
# OpenCLIP: "openai", "laion2b_s34b_b79k", "laion400m_e32"
CFG.MODEL.VLM.PRETRAINED = "openai"

# Backend for Perception Encoder: "open_clip", "timm", "native"
CFG.MODEL.VLM.PE_BACKEND = "open_clip"

# Classification head type: "linear", "mlp", "zero_shot"
CFG.MODEL.VLM.HEAD_TYPE = "linear"

# Hidden dim for MLP head
CFG.MODEL.VLM.HEAD_HIDDEN_DIM = 512

# Dropout for MLP head
CFG.MODEL.VLM.HEAD_DROPOUT = 0.1


# ---- Zero-Shot VLM Classifier Settings ----
CFG.MITIGATOR.ZERO_SHOT_VLM = CN()

# Encoder type
CFG.MITIGATOR.ZERO_SHOT_VLM.ENCODER_TYPE = "openclip"

# Model name
CFG.MITIGATOR.ZERO_SHOT_VLM.MODEL_NAME = "ViT-L-14"

# Pretrained weights
CFG.MITIGATOR.ZERO_SHOT_VLM.PRETRAINED = "openai"

# Class name variant for dataset
# Options depend on dataset:
# - UTKFace: "default", "short", "detailed"
# - Waterbirds: "default", "species", "detailed"
# - CelebA: "default", "blond_hair", "male", "young", "smiling"
# - UrbanCars: "default", "car_types", "detailed"
# Can also be a custom list of class name strings
CFG.MITIGATOR.ZERO_SHOT_VLM.CLASS_NAME_VARIANT = "default"

# Temperature scaling for similarity scores
CFG.MITIGATOR.ZERO_SHOT_VLM.TEMPERATURE = 100.0

# Whether to evaluate all available prompt variants
CFG.MITIGATOR.ZERO_SHOT_VLM.EVALUATE_ALL_VARIANTS = True


# ============================================
# Dataset-specific class names (reference)
# ============================================

# UTKFace (gender)
# - default: ["a photo of a male person", "a photo of a female person"]
# - short: ["male", "female"]
# - detailed: ["a photograph of a man", "a photograph of a woman"]

# Waterbirds
# - default: ["a photo of a waterbird", "a photo of a landbird"]
# - species: ["a photo of an albatross, a seabird...", "a photo of a warbler, a small songbird"]
# - detailed: ["a photo of a seabird near water...", "a photo of a forest bird on land..."]

# CelebA (use attribute-specific variants)
# - blond_hair: ["person with dark hair", "person with blond hair"]
# - male: ["female person", "male person"]

# UrbanCars
# - default: ["car in urban environment", "car in rural environment"]
# - car_types: ["sedan, hatchback, or city car", "SUV, pickup truck, or off-road vehicle"]


# ============================================
# SAE-Filtered Zero-Shot Configuration
# Add to configs/cfg.py
# ============================================

CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT = CN()

# ---- VLM Encoder Settings ----
# Encoder type: "openclip", "siglip", "perception_encoder"
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.ENCODER_TYPE = "openclip"

# Model architecture
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.MODEL_NAME = "ViT-L-14"

# Pretrained weights
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.PRETRAINED = "openai"

# ---- SAE Settings ----
# Path to trained SAE checkpoint (ae.pt)
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.SAE_CHECKPOINT_PATH = ""

# Path to SAE analysis results (analysis_results.json)
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.SAE_ANALYSIS_PATH = ""

# Purity threshold for monosemantic neurons
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.PURITY_THRESHOLD = 1.0

# Activation threshold for filtering neurons
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.ACTIVATION_THRESHOLD = 0.1

# ---- Filtering Strategy ----
# Filter mode:
# - "per_class": Use each class's neurons to compute that class's score
# - "all_mono": Keep all monosemantic neurons regardless of class
# - "target_only": Keep only neurons for a specific target class
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.FILTER_MODE = "per_class"

# Target class for "target_only" mode
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.TARGET_CLASS = 0

# ---- Zero-Shot Settings ----
# Class name variant for dataset prompts
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.CLASS_NAME_VARIANT = "default"

# Temperature for similarity scoring
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.TEMPERATURE = 100.0

# ---- Evaluation ----
# Compare with unfiltered baseline
CFG.MITIGATOR.SAE_FILTERED_ZERO_SHOT.COMPARE_WITH_UNFILTERED = True


# ============================================
# SAE Text-Aligned Zero-Shot Configuration
# Add to configs/cfg.py
# ============================================

CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT = CN()

# ---- VLM Encoder Settings ----
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.ENCODER_TYPE = "openclip"
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.MODEL_NAME = "ViT-L-14"
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.PRETRAINED = "openai"

# ---- SAE Settings ----
# Only need the SAE checkpoint - no analysis_results.json needed!
# Neuron-class assignments come from text alignment instead
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.SAE_CHECKPOINT_PATH = ""

# ---- Alignment Settings ----
# How to compute neuron-class alignment:
# - "decoder": Use SAE decoder columns (neuron directions in feature space)
# - "encoder": Use SAE encoder rows (what each neuron responds to)
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.ALIGNMENT_METHOD = "decoder"

# Minimum cosine similarity between neuron direction and text embedding
# to assign the neuron to that class
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.ALIGNMENT_THRESHOLD = 0.1

# Minimum margin between best and second-best class alignment
# Ensures neuron is clearly associated with one class
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.ALIGNMENT_MARGIN = 0.05

# ---- Zero-Shot Settings ----
# Class name variant from DATASET_CLASS_NAMES
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.CLASS_NAME_VARIANT = "default"

# Temperature for similarity scoring
CFG.MITIGATOR.SAE_TEXT_ALIGNED_ZERO_SHOT.TEMPERATURE = 100.0


# ============================================
# SAE Combined Dataset Configuration
# Add to configs/cfg.py
# ============================================

CFG.DATASET.SAE_COMBINED = CN()

# ---- Dataset Roots ----
# CUB-200-2011: All bird images
CFG.DATASET.SAE_COMBINED.CUB200_ROOT = ""

# Stanford Cars: All car images
CFG.DATASET.SAE_COMBINED.STANFORD_CARS_ROOT = ""

# Places365: Selected scene categories
CFG.DATASET.SAE_COMBINED.PLACES365_ROOT = ""

# LVIS: Annotations directory
CFG.DATASET.SAE_COMBINED.LVIS_ROOT = ""

# COCO: Images directory (used with LVIS annotations)
CFG.DATASET.SAE_COMBINED.COCO_IMAGES_ROOT = ""

# ---- Category Selections ----

# Places365 categories to include
# Default: forests, water, roads, urban scenes
CFG.DATASET.SAE_COMBINED.PLACES365_CATEGORIES = [
    "bamboo_forest",
    "forest/broadleaf",
    "ocean",
    "lake/natural",
    "alley",
    "crosswalk",
    "downtown",
    "gas_station",
    "garage/outdoor",
    "driveway",
    "forest_road",
    "field_road",
    "desert_road",
]

# LVIS categories to include (uses full images containing these objects)
# Default: street objects and animals
CFG.DATASET.SAE_COMBINED.LVIS_CATEGORIES = [
    "fire_hydrant",  # fireplug
    "stop_sign",
    "street_sign",
    "parking_meter",
    "traffic_light",
    "cow",
    "horse",
    "sheep",
]

# ---- Image Settings ----
CFG.DATASET.SAE_COMBINED.IMAGE_SIZE = 224

# Whether to use train, val, or all splits
CFG.DATASET.SAE_COMBINED.SPLIT = "all"

CFG.DATASET.SAE_COMBINED.BALANCE_SOURCES = True

CFG.MITIGATOR.SAE.PRECOMPUTED_FEATURES_PATH = ""

CFG.MITIGATOR.SAE.PRETRAINED_SAE_PATH = ""


# CFG.MITIGATOR.TAG_SAE = CN()

# # Path to tags CSV from MAVIAS pipeline
# CFG.MITIGATOR.TAG_SAE.TAGS_CSV_PATH = "train_tags.csv"
# CFG.MITIGATOR.TAG_SAE.TAG_COLUMN = "irrelevant_tags"
# CFG.MITIGATOR.TAG_SAE.TAG_SEPARATOR = " | "
# CFG.MITIGATOR.TAG_SAE.MIN_TAG_FREQUENCY = 10

# # Model checkpoint (optional)
# CFG.MITIGATOR.TAG_SAE.CHECKPOINT_PATH = ""

# # Precomputed features (optional)
# CFG.MITIGATOR.TAG_SAE.PRECOMPUTED_FEATURES_PATH = ""

# # SAE Architecture
# CFG.MITIGATOR.TAG_SAE.EXPANSION_FACTOR = 8
# CFG.MITIGATOR.TAG_SAE.NUM_FREE_NEURONS = 0  # 0 = use expansion factor

# # Training
# CFG.MITIGATOR.TAG_SAE.STEPS = 20000
# CFG.MITIGATOR.TAG_SAE.BATCH_SIZE = 256
# CFG.MITIGATOR.TAG_SAE.LR = 1e-3

# # Loss Weights
# CFG.MITIGATOR.TAG_SAE.LAMBDA_RECONSTRUCTION = 1.0
# CFG.MITIGATOR.TAG_SAE.LAMBDA_SPARSITY = 1e-3
# CFG.MITIGATOR.TAG_SAE.LAMBDA_TAG = 1.0

# # Tag Supervision Config
# CFG.MITIGATOR.TAG_SAE.TAG_LOSS_TYPE = "bce"  # "bce", "hinge", "mse"
# CFG.MITIGATOR.TAG_SAE.POSITIVE_WEIGHT = 1.0
# CFG.MITIGATOR.TAG_SAE.NEGATIVE_WEIGHT = 0.5
# CFG.MITIGATOR.TAG_SAE.USE_NEGATIVE_SUPERVISION = True
# CFG.MITIGATOR.TAG_SAE.MARGIN = 0.5  # For hinge loss
# CFG.MITIGATOR.TAG_SAE.TARGET_ACTIVATION = 1.0  # For MSE loss

CFG.MITIGATOR.TAG_SAE = CN()

# ---- Tags from MAVIAS Pipeline ----
# Path to tags CSV (e.g., train_tags.csv from MAVIAS)
CFG.MITIGATOR.TAG_SAE.TAGS_CSV_PATH = "train_tags.csv"

# Column containing tags to use for supervision
# Options: "tags" (all tags), "irrelevant_tags" (bias tags only)
CFG.MITIGATOR.TAG_SAE.TAG_COLUMN = "irrelevant_tags"

# Separator between tags in the CSV
CFG.MITIGATOR.TAG_SAE.TAG_SEPARATOR = " | "

# Minimum frequency for a tag to be included
# Tags appearing less than this are ignored
CFG.MITIGATOR.TAG_SAE.MIN_TAG_FREQUENCY = 10

# ---- Model Checkpoints ----
# Path to pretrained model checkpoint (e.g., ERM model)
CFG.MITIGATOR.TAG_SAE.CHECKPOINT_PATH = ""

# Path to precomputed features (skip extraction if provided)
CFG.MITIGATOR.TAG_SAE.PRECOMPUTED_FEATURES_PATH = ""

# ---- SAE Architecture ----
# Expansion factor: dict_size = feature_dim * expansion_factor
# Only used if NUM_FREE_NEURONS is 0
CFG.MITIGATOR.TAG_SAE.EXPANSION_FACTOR = 8

# Number of additional free neurons beyond tag neurons
# If > 0, dict_size = num_tags + NUM_FREE_NEURONS
# If 0, uses EXPANSION_FACTOR instead
CFG.MITIGATOR.TAG_SAE.NUM_FREE_NEURONS = 0

# ---- Training Parameters ----
CFG.MITIGATOR.TAG_SAE.STEPS = 20000
CFG.MITIGATOR.TAG_SAE.BATCH_SIZE = 256
CFG.MITIGATOR.TAG_SAE.LR = 1e-3

# ---- Loss Weights ----
# Weight for reconstruction loss
CFG.MITIGATOR.TAG_SAE.LAMBDA_RECONSTRUCTION = 1.0

# Weight for sparsity loss (L1 on latents)
CFG.MITIGATOR.TAG_SAE.LAMBDA_SPARSITY = 1e-3

# Weight for tag supervision loss
CFG.MITIGATOR.TAG_SAE.LAMBDA_TAG = 1.0


# ---- Energy Balancing (NEW) ----
# Weight for energy balance loss (encourages free neurons to be active)
# Set > 0 to prevent tag neurons from dominating
# Recommended: 0.1 - 1.0
CFG.MITIGATOR.TAG_SAE.LAMBDA_ENERGY_BALANCE = 0.8

# Minimum ratio of energy that should be in free neurons
# E.g., 0.3 means at least 30% of total energy should be in free neurons
CFG.MITIGATOR.TAG_SAE.MIN_FREE_ENERGY_RATIO = 0.7

# ---- Orthogonality Constraint (NEW) ----
# Weight for orthogonality loss (makes free neurons learn different info than tag neurons)
# Set > 0 to encourage free neurons to capture non-tag information
# Recommended: 0.01 - 0.1
CFG.MITIGATOR.TAG_SAE.LAMBDA_ORTHOGONALITY = 1.0

# ---- Training Schedule (NEW) ----
# Number of warmup steps before applying full tag supervision
# During warmup, SAE focuses on reconstruction, then gradually adds tag supervision
# Recommended: 1000 - 5000
CFG.MITIGATOR.TAG_SAE.TAG_WARMUP_STEPS = 0


# ---- Tag Supervision Configuration ----
# Loss type for tag supervision
# Options:
#   - "bce": Binary cross-entropy (soft, differentiable)
#   - "hinge": Margin-based (enforce minimum activation gap)
#   - "mse": Mean squared error to target activation level
CFG.MITIGATOR.TAG_SAE.TAG_LOSS_TYPE = "bce"

# Weight for positive examples (tag present, should activate)
CFG.MITIGATOR.TAG_SAE.POSITIVE_WEIGHT = 1.0

# Weight for negative examples (tag absent, should not activate)
CFG.MITIGATOR.TAG_SAE.NEGATIVE_WEIGHT = 0.5

# Whether to penalize activation when tag is absent
# If False, only encourages activation when tag is present
CFG.MITIGATOR.TAG_SAE.USE_NEGATIVE_SUPERVISION = True

# Margin for hinge loss (activation should be > margin when tag present)
CFG.MITIGATOR.TAG_SAE.MARGIN = 0.5

# Target activation level for MSE loss (when tag is present)
CFG.MITIGATOR.TAG_SAE.TARGET_ACTIVATION = 1.0

CFG.MITIGATOR.TAG_SAE.SAE_CHECKPOINT_PATH = ""


CFG.MITIGATOR.SAE_DEBIAS_CLS = CN()

# ---- Tag-SAE Checkpoint ----
# Path to trained Tag-SAE checkpoint (required)
CFG.MITIGATOR.SAE_DEBIAS_CLS.SAE_CHECKPOINT_PATH = ""

# ---- Debiasing Mode ----
# How to debias features:
# - "all": Zero ALL tag neurons (removes all bias info)
# - "specific": Zero only specified tags (selective removal)
# - "none": No debiasing (baseline comparison)
CFG.MITIGATOR.SAE_DEBIAS_CLS.DEBIAS_MODE = "all"

# Tags to remove when mode="specific"
# Example: ["water", "forest", "ocean"]
CFG.MITIGATOR.SAE_DEBIAS_CLS.TAGS_TO_REMOVE = []

# ---- Classifier Architecture ----
# Type of classifier head
# - "linear": Simple linear layer (faster, less overfitting)
# - "mlp": MLP with one hidden layer (more capacity)
CFG.MITIGATOR.SAE_DEBIAS_CLS.CLASSIFIER_TYPE = "linear"

# Hidden dimension for MLP classifier
CFG.MITIGATOR.SAE_DEBIAS_CLS.HIDDEN_DIM = 256

# Dropout rate
CFG.MITIGATOR.SAE_DEBIAS_CLS.DROPOUT = 0.1

# ---- Training ----
CFG.MITIGATOR.SAE_DEBIAS_CLS.EPOCHS = 50
CFG.MITIGATOR.SAE_DEBIAS_CLS.LR = 1e-3
CFG.MITIGATOR.SAE_DEBIAS_CLS.WEIGHT_DECAY = 1e-4


CFG.MITIGATOR.TAG_ONLY_SAE = CN()

# ---- Tags Configuration ----
# Path to tags CSV (from MAVIAS pipeline)
CFG.MITIGATOR.TAG_ONLY_SAE.TAGS_CSV_PATH = "train_tags.csv"

# Column with ALL tags (used for SAE training)
CFG.MITIGATOR.TAG_ONLY_SAE.ALL_TAGS_COLUMN = "tags"

# Column with irrelevant/bias tags (used for debiasing)
CFG.MITIGATOR.TAG_ONLY_SAE.IRRELEVANT_TAGS_COLUMN = "irrelevant_tags"

# Separator between tags in CSV
CFG.MITIGATOR.TAG_ONLY_SAE.TAG_SEPARATOR = " | "

# Minimum tag frequency (filter rare tags)
CFG.MITIGATOR.TAG_ONLY_SAE.MIN_TAG_FREQUENCY = 10

# ---- Checkpoints ----
# Path to pretrained SAE checkpoint (for eval mode)
CFG.MITIGATOR.TAG_ONLY_SAE.SAE_CHECKPOINT_PATH = ""

# Path to precomputed features (skip extraction if provided)
CFG.MITIGATOR.TAG_ONLY_SAE.PRECOMPUTED_FEATURES_PATH = ""

# ---- Training ----
CFG.MITIGATOR.TAG_ONLY_SAE.STEPS = 20000
CFG.MITIGATOR.TAG_ONLY_SAE.BATCH_SIZE = 256
CFG.MITIGATOR.TAG_ONLY_SAE.LR = 1e-3

# ---- Loss Weights ----
CFG.MITIGATOR.TAG_ONLY_SAE.LAMBDA_RECONSTRUCTION = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE.LAMBDA_SPARSITY = 1e-3
CFG.MITIGATOR.TAG_ONLY_SAE.LAMBDA_TAG = 1.0

# ---- Tag Supervision ----
# Loss type: "bce", "hinge", "mse"
CFG.MITIGATOR.TAG_ONLY_SAE.TAG_LOSS_TYPE = "bce"
CFG.MITIGATOR.TAG_ONLY_SAE.POSITIVE_WEIGHT = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE.NEGATIVE_WEIGHT = 0.5
CFG.MITIGATOR.TAG_ONLY_SAE.USE_NEGATIVE_SUPERVISION = True
CFG.MITIGATOR.TAG_ONLY_SAE.MARGIN = 0.5
CFG.MITIGATOR.TAG_ONLY_SAE.TARGET_ACTIVATION = 1.0


CFG.MITIGATOR.TAG_ONLY_DEBIAS_CLS = CN()

# Path to trained Tag-Only SAE checkpoint (required)
CFG.MITIGATOR.TAG_ONLY_DEBIAS_CLS.SAE_CHECKPOINT_PATH = ""

# Classifier architecture
# - "linear": Simple linear layer
# - "mlp": MLP with one hidden layer
CFG.MITIGATOR.TAG_ONLY_DEBIAS_CLS.CLASSIFIER_TYPE = "linear"
CFG.MITIGATOR.TAG_ONLY_DEBIAS_CLS.HIDDEN_DIM = 256  # For MLP
CFG.MITIGATOR.TAG_ONLY_DEBIAS_CLS.DROPOUT = 0.1

# Training
CFG.MITIGATOR.TAG_ONLY_DEBIAS_CLS.EPOCHS = 50
CFG.MITIGATOR.TAG_ONLY_DEBIAS_CLS.LR = 1e-3
CFG.MITIGATOR.TAG_ONLY_DEBIAS_CLS.WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------
# TAG-ONLY SAE V2 (with Bias and Target Class Neurons)
# ---------------------------------------------------------------------
CFG.MITIGATOR.TAG_ONLY_SAE_V2 = CN()

# ---- Tags Configuration ----
CFG.MITIGATOR.TAG_ONLY_SAE_V2.TAGS_CSV_PATH = "train_tags.csv"
CFG.MITIGATOR.TAG_ONLY_SAE_V2.ALL_TAGS_COLUMN = "tags"
CFG.MITIGATOR.TAG_ONLY_SAE_V2.IRRELEVANT_TAGS_COLUMN = "irrelevant_tags"
CFG.MITIGATOR.TAG_ONLY_SAE_V2.TAG_SEPARATOR = " | "
CFG.MITIGATOR.TAG_ONLY_SAE_V2.MIN_TAG_FREQUENCY = 10

# ---- Checkpoints ----
CFG.MITIGATOR.TAG_ONLY_SAE_V2.SAE_CHECKPOINT_PATH = ""
CFG.MITIGATOR.TAG_ONLY_SAE_V2.PRECOMPUTED_FEATURES_PATH = ""

# ---- Training ----
CFG.MITIGATOR.TAG_ONLY_SAE_V2.STEPS = 20000
CFG.MITIGATOR.TAG_ONLY_SAE_V2.BATCH_SIZE = 256
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LR = 1e-3

# ---- Loss Weights ----
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_RECONSTRUCTION = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_SPARSITY = 1e-3
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_TAG = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_BIAS = 1.0  # Weight for bias label supervision
CFG.MITIGATOR.TAG_ONLY_SAE_V2.LAMBDA_TARGET = 1.0  # Weight for target class supervision

# ---- Tag Supervision ----
CFG.MITIGATOR.TAG_ONLY_SAE_V2.TAG_LOSS_TYPE = "bce"
CFG.MITIGATOR.TAG_ONLY_SAE_V2.POSITIVE_WEIGHT = 1.0
CFG.MITIGATOR.TAG_ONLY_SAE_V2.NEGATIVE_WEIGHT = 0.5
CFG.MITIGATOR.TAG_ONLY_SAE_V2.USE_NEGATIVE_SUPERVISION = True
CFG.MITIGATOR.TAG_ONLY_SAE_V2.MARGIN = 0.5
CFG.MITIGATOR.TAG_ONLY_SAE_V2.TARGET_ACTIVATION = 1.0


# ---------------------------------------------------------------------
# TAG EXTRACTION (Comprehensive Semantic Extraction)
# ---------------------------------------------------------------------
CFG.MITIGATOR.TAG_EXTRACTION = CN()
CFG.MITIGATOR.TAG_EXTRACTION.TASK_DESCRIPTION = ""
CFG.MITIGATOR.TAG_EXTRACTION.VLM_MODEL = "llava"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_DEVICE = "cuda"
CFG.MITIGATOR.TAG_EXTRACTION.VLM_BATCH_SIZE = 1
CFG.MITIGATOR.TAG_EXTRACTION.LLM_MODEL = "ollama"
CFG.MITIGATOR.TAG_EXTRACTION.LLM_MODEL_PATH = ""
CFG.MITIGATOR.TAG_EXTRACTION.LLM_API_KEY = ""
CFG.MITIGATOR.TAG_EXTRACTION.LLM_BASE_URL = ""
CFG.MITIGATOR.TAG_EXTRACTION.LLM_TAG_BATCH_SIZE = 100
CFG.MITIGATOR.TAG_EXTRACTION.MIN_TAG_FREQUENCY = 5
CFG.MITIGATOR.TAG_EXTRACTION.ENABLE_HUMAN_REVIEW = False
CFG.MITIGATOR.TAG_EXTRACTION.OUTPUT_DIR = ""
CFG.MITIGATOR.TAG_EXTRACTION.RESUME_FROM_STAGE = 1


CFG.MITIGATOR.CLIP_DISTILLATION = CN()
CFG.MITIGATOR.CLIP_DISTILLATION.TEACHER_ARCH = "ViT-L-14"
CFG.MITIGATOR.CLIP_DISTILLATION.TEACHER_PRETRAINED = "openai"
CFG.MITIGATOR.CLIP_DISTILLATION.STUDENT_ARCH = "ViT-B-32"
CFG.MITIGATOR.CLIP_DISTILLATION.STUDENT_PRETRAINED = "openai"
CFG.MITIGATOR.CLIP_DISTILLATION.STEPS = 10000
CFG.MITIGATOR.CLIP_DISTILLATION.BATCH_SIZE = 256
CFG.MITIGATOR.CLIP_DISTILLATION.LR = 1e-3
CFG.MITIGATOR.CLIP_DISTILLATION.WEIGHT_DECAY = 0.0
CFG.MITIGATOR.CLIP_DISTILLATION.PROJECTION_HIDDEN = None
CFG.MITIGATOR.CLIP_DISTILLATION.CLASS_NAME_VARIANT = "default"
CFG.MITIGATOR.CLIP_DISTILLATION.TEMPERATURE = 100.0
CFG.MITIGATOR.CLIP_DISTILLATION.PROJECTION_PATH = ""
CFG.MITIGATOR.CLIP_DISTILLATION.BIAS_TEXTS = []


CFG.MITIGATOR.OPENCLIP_CLASSIFIER = CN()
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.ARCH = "ViT-B-32"
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.PRETRAINED = "openai"
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.CLASSIFIER_TYPE = "linear"
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.HIDDEN_DIM = 512
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.NUM_LAYERS = 1
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.DROPOUT = 0.1
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.EPOCHS = 50
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.LR = 1e-3
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.WEIGHT_DECAY = 1e-4
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.BATCH_SIZE = 256
CFG.MITIGATOR.OPENCLIP_CLASSIFIER.PRECOMPUTE_FEATURES = True
