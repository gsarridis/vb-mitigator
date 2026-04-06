#!/usr/bin/env python3
"""
Run ImageNet zero-shot classification analysis across ALL pretrained
OpenCLIP models. For each model:
  1. Load model, build zero-shot classifier weights
  2. Compute top-1 and top-5 accuracy on ImageNet validation set
  3. Delete model from GPU/CPU and clear cache
  4. Append results to a master table

Outputs:
  - results/master_results.csv — one row per model with all metrics
  - results/master_results.json — same data in JSON
  - results/summary_plots/  — cross-model comparison plots

Usage:
    python run_all_models_imagenet.py
    python run_all_models_imagenet.py --filter "ViT-B"        # only models matching this
    python run_all_models_imagenet.py --exclude "RN"          # skip ResNet models
    python run_all_models_imagenet.py --resume                # skip already completed models

Requirements:
    pip install torch open-clip-torch Pillow matplotlib numpy pandas tqdm torchvision
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip
from open_clip import (
    build_zero_shot_classifier,
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
)


# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = "/mnt/cephfs/home/common/datasets/ImageNet/ILSVRC/Data/CLS-LOC/val"
DEFAULT_LABELS_FILE = "/mnt/cephfs/home/common/datasets/ImageNet/LOC_val_solution.csv"
DEFAULT_BATCH_SIZE = 16
DEFAULT_OUTPUT_DIR = "./results_imagenet"
DEFAULT_NUM_WORKERS = 4


# ═══════════════════════════════════════════════════════════════════
# Synset to class index mapping (ImageNet 1000 classes)
# ═══════════════════════════════════════════════════════════════════

# This is the standard ImageNet synset ordering (same as torchvision)
IMAGENET_SYNSETS = [
    "n01440764",
    "n01443537",
    "n01484850",
    "n01491361",
    "n01494475",
    "n01496331",
    "n01498041",
    "n01514668",
    "n01514859",
    "n01518878",
    "n01530575",
    "n01531178",
    "n01532829",
    "n01534433",
    "n01537544",
    "n01558993",
    "n01560419",
    "n01580077",
    "n01582220",
    "n01592084",
    "n01601694",
    "n01608432",
    "n01614925",
    "n01616318",
    "n01622779",
    "n01629819",
    "n01630670",
    "n01631663",
    "n01632458",
    "n01632777",
    "n01641577",
    "n01644373",
    "n01644900",
    "n01664065",
    "n01665541",
    "n01667114",
    "n01667778",
    "n01669191",
    "n01675722",
    "n01677366",
    "n01682714",
    "n01685808",
    "n01687978",
    "n01688243",
    "n01689811",
    "n01692333",
    "n01693334",
    "n01694178",
    "n01695060",
    "n01697457",
    "n01698640",
    "n01704323",
    "n01728572",
    "n01728920",
    "n01729322",
    "n01729977",
    "n01734418",
    "n01735189",
    "n01737021",
    "n01739381",
    "n01740131",
    "n01742172",
    "n01744401",
    "n01748264",
    "n01749939",
    "n01751748",
    "n01753488",
    "n01755581",
    "n01756291",
    "n01768244",
    "n01770081",
    "n01770393",
    "n01773157",
    "n01773549",
    "n01773797",
    "n01774384",
    "n01774750",
    "n01775062",
    "n01776313",
    "n01784675",
    "n01795545",
    "n01796340",
    "n01797886",
    "n01798484",
    "n01806143",
    "n01806567",
    "n01807496",
    "n01817953",
    "n01818515",
    "n01819313",
    "n01820546",
    "n01824575",
    "n01828970",
    "n01829413",
    "n01833805",
    "n01843065",
    "n01843383",
    "n01847000",
    "n01855032",
    "n01855672",
    "n01860187",
    "n01871265",
    "n01872401",
    "n01873310",
    "n01877812",
    "n01882714",
    "n01883070",
    "n01910747",
    "n01914609",
    "n01917289",
    "n01924916",
    "n01930112",
    "n01943899",
    "n01944390",
    "n01945685",
    "n01950731",
    "n01955084",
    "n01968897",
    "n01978287",
    "n01978455",
    "n01980166",
    "n01981276",
    "n01983481",
    "n01984695",
    "n01985128",
    "n01986214",
    "n01990800",
    "n02002556",
    "n02002724",
    "n02006656",
    "n02007558",
    "n02009229",
    "n02009912",
    "n02011460",
    "n02012849",
    "n02013706",
    "n02017213",
    "n02018207",
    "n02018795",
    "n02025239",
    "n02027492",
    "n02028035",
    "n02033041",
    "n02037110",
    "n02051845",
    "n02056570",
    "n02058221",
    "n02066245",
    "n02071294",
    "n02074367",
    "n02077923",
    "n02085620",
    "n02085782",
    "n02085936",
    "n02086079",
    "n02086240",
    "n02086646",
    "n02086910",
    "n02087046",
    "n02087394",
    "n02088094",
    "n02088238",
    "n02088364",
    "n02088466",
    "n02088632",
    "n02089078",
    "n02089867",
    "n02089973",
    "n02090379",
    "n02090622",
    "n02090721",
    "n02091032",
    "n02091134",
    "n02091244",
    "n02091467",
    "n02091635",
    "n02091831",
    "n02092002",
    "n02092339",
    "n02093256",
    "n02093428",
    "n02093647",
    "n02093754",
    "n02093859",
    "n02093991",
    "n02094114",
    "n02094258",
    "n02094433",
    "n02095314",
    "n02095570",
    "n02095889",
    "n02096051",
    "n02096177",
    "n02096294",
    "n02096437",
    "n02096585",
    "n02097047",
    "n02097130",
    "n02097209",
    "n02097298",
    "n02097474",
    "n02097658",
    "n02098105",
    "n02098286",
    "n02098413",
    "n02099267",
    "n02099429",
    "n02099601",
    "n02099712",
    "n02099849",
    "n02100236",
    "n02100583",
    "n02100735",
    "n02100877",
    "n02101006",
    "n02101388",
    "n02101556",
    "n02102040",
    "n02102177",
    "n02102318",
    "n02102480",
    "n02102973",
    "n02104029",
    "n02104365",
    "n02105056",
    "n02105162",
    "n02105251",
    "n02105412",
    "n02105505",
    "n02105641",
    "n02105855",
    "n02106030",
    "n02106166",
    "n02106382",
    "n02106550",
    "n02106662",
    "n02107142",
    "n02107312",
    "n02107574",
    "n02107683",
    "n02107908",
    "n02108000",
    "n02108089",
    "n02108422",
    "n02108551",
    "n02108915",
    "n02109047",
    "n02109525",
    "n02109961",
    "n02110063",
    "n02110185",
    "n02110341",
    "n02110627",
    "n02110806",
    "n02110958",
    "n02111129",
    "n02111277",
    "n02111500",
    "n02111889",
    "n02112018",
    "n02112137",
    "n02112350",
    "n02112706",
    "n02113023",
    "n02113186",
    "n02113624",
    "n02113712",
    "n02113799",
    "n02113978",
    "n02114367",
    "n02114548",
    "n02114712",
    "n02114855",
    "n02115641",
    "n02115913",
    "n02116738",
    "n02117135",
    "n02119022",
    "n02119789",
    "n02120079",
    "n02120505",
    "n02123045",
    "n02123159",
    "n02123394",
    "n02123597",
    "n02124075",
    "n02125311",
    "n02127052",
    "n02128385",
    "n02128757",
    "n02128925",
    "n02129165",
    "n02129604",
    "n02130308",
    "n02132136",
    "n02133161",
    "n02134084",
    "n02134418",
    "n02137549",
    "n02138441",
    "n02165105",
    "n02165456",
    "n02167151",
    "n02168699",
    "n02169497",
    "n02172182",
    "n02174001",
    "n02177972",
    "n02190166",
    "n02206856",
    "n02219486",
    "n02226429",
    "n02229544",
    "n02231487",
    "n02233338",
    "n02236044",
    "n02256656",
    "n02259212",
    "n02264363",
    "n02268443",
    "n02268853",
    "n02276258",
    "n02277742",
    "n02279972",
    "n02280649",
    "n02281406",
    "n02281787",
    "n02317335",
    "n02319095",
    "n02321529",
    "n02325366",
    "n02326432",
    "n02328150",
    "n02342885",
    "n02346627",
    "n02356798",
    "n02361337",
    "n02363005",
    "n02364673",
    "n02389026",
    "n02391049",
    "n02395406",
    "n02396427",
    "n02397096",
    "n02398521",
    "n02403003",
    "n02408429",
    "n02410509",
    "n02412080",
    "n02415577",
    "n02417914",
    "n02422106",
    "n02422699",
    "n02423022",
    "n02437312",
    "n02437616",
    "n02441942",
    "n02442845",
    "n02443114",
    "n02443484",
    "n02444819",
    "n02445715",
    "n02447366",
    "n02454379",
    "n02457408",
    "n02480495",
    "n02480855",
    "n02481823",
    "n02483362",
    "n02483708",
    "n02484975",
    "n02486261",
    "n02486410",
    "n02487347",
    "n02488291",
    "n02488702",
    "n02489166",
    "n02490219",
    "n02492035",
    "n02492660",
    "n02493509",
    "n02493793",
    "n02494079",
    "n02497673",
    "n02500267",
    "n02504013",
    "n02504458",
    "n02509815",
    "n02510455",
    "n02514041",
    "n02526121",
    "n02536864",
    "n02606052",
    "n02607072",
    "n02640242",
    "n02641379",
    "n02643566",
    "n02655020",
    "n02666196",
    "n02667093",
    "n02669723",
    "n02672831",
    "n02676566",
    "n02687172",
    "n02690373",
    "n02692877",
    "n02699494",
    "n02701002",
    "n02704792",
    "n02708093",
    "n02727426",
    "n02730930",
    "n02747177",
    "n02749479",
    "n02769748",
    "n02776631",
    "n02777292",
    "n02782093",
    "n02783161",
    "n02786058",
    "n02787622",
    "n02788148",
    "n02790996",
    "n02791124",
    "n02791270",
    "n02793495",
    "n02794156",
    "n02795169",
    "n02797295",
    "n02799071",
    "n02802426",
    "n02804414",
    "n02804610",
    "n02807133",
    "n02808304",
    "n02808440",
    "n02814533",
    "n02814860",
    "n02815834",
    "n02817516",
    "n02823428",
    "n02823750",
    "n02825657",
    "n02834397",
    "n02835271",
    "n02837789",
    "n02840245",
    "n02841315",
    "n02843684",
    "n02859443",
    "n02860847",
    "n02865351",
    "n02869837",
    "n02870880",
    "n02871525",
    "n02877765",
    "n02879718",
    "n02883205",
    "n02892201",
    "n02892767",
    "n02894605",
    "n02895154",
    "n02906734",
    "n02909870",
    "n02910353",
    "n02916936",
    "n02917067",
    "n02927161",
    "n02930766",
    "n02939185",
    "n02948072",
    "n02950826",
    "n02951358",
    "n02951585",
    "n02963159",
    "n02965783",
    "n02966193",
    "n02966687",
    "n02971356",
    "n02974003",
    "n02977058",
    "n02978881",
    "n02979186",
    "n02980441",
    "n02981792",
    "n02988304",
    "n02992211",
    "n02992529",
    "n02999410",
    "n03000134",
    "n03000247",
    "n03000684",
    "n03014705",
    "n03016953",
    "n03017168",
    "n03018349",
    "n03026506",
    "n03028079",
    "n03032252",
    "n03041632",
    "n03042490",
    "n03045698",
    "n03047690",
    "n03062245",
    "n03063599",
    "n03063689",
    "n03065424",
    "n03075370",
    "n03085013",
    "n03089624",
    "n03095699",
    "n03100240",
    "n03109150",
    "n03110669",
    "n03124043",
    "n03124170",
    "n03125729",
    "n03126707",
    "n03127747",
    "n03127925",
    "n03131574",
    "n03133878",
    "n03134739",
    "n03141823",
    "n03146219",
    "n03160309",
    "n03179701",
    "n03180011",
    "n03187595",
    "n03188531",
    "n03196217",
    "n03197337",
    "n03201208",
    "n03207743",
    "n03207941",
    "n03208938",
    "n03216828",
    "n03218198",
    "n03220513",
    "n03223299",
    "n03240683",
    "n03249569",
    "n03250847",
    "n03255030",
    "n03259280",
    "n03271574",
    "n03272010",
    "n03272562",
    "n03290653",
    "n03291819",
    "n03297495",
    "n03314780",
    "n03325584",
    "n03337140",
    "n03344393",
    "n03345487",
    "n03347037",
    "n03355925",
    "n03372029",
    "n03376595",
    "n03379051",
    "n03384352",
    "n03388043",
    "n03388183",
    "n03388549",
    "n03393912",
    "n03394916",
    "n03400231",
    "n03404251",
    "n03417042",
    "n03424325",
    "n03425413",
    "n03443371",
    "n03444034",
    "n03445777",
    "n03445924",
    "n03447447",
    "n03447721",
    "n03450230",
    "n03452741",
    "n03457902",
    "n03459775",
    "n03461385",
    "n03467068",
    "n03476684",
    "n03476991",
    "n03478589",
    "n03481172",
    "n03482405",
    "n03483316",
    "n03485407",
    "n03485794",
    "n03492542",
    "n03494278",
    "n03495258",
    "n03496892",
    "n03498962",
    "n03527444",
    "n03529860",
    "n03530642",
    "n03532672",
    "n03534580",
    "n03535780",
    "n03538406",
    "n03544143",
    "n03584254",
    "n03584829",
    "n03590841",
    "n03594734",
    "n03594945",
    "n03595614",
    "n03598930",
    "n03599486",
    "n03602883",
    "n03617480",
    "n03623198",
    "n03627232",
    "n03630383",
    "n03633091",
    "n03637318",
    "n03642806",
    "n03649909",
    "n03657121",
    "n03658185",
    "n03661043",
    "n03662601",
    "n03666591",
    "n03670208",
    "n03673027",
    "n03676483",
    "n03680355",
    "n03690938",
    "n03691459",
    "n03692522",
    "n03697007",
    "n03706229",
    "n03709823",
    "n03710193",
    "n03710637",
    "n03710721",
    "n03717622",
    "n03720891",
    "n03721384",
    "n03724870",
    "n03729826",
    "n03733131",
    "n03733281",
    "n03733805",
    "n03742115",
    "n03743016",
    "n03759954",
    "n03761084",
    "n03763968",
    "n03764736",
    "n03769881",
    "n03770439",
    "n03770679",
    "n03773504",
    "n03775071",
    "n03775546",
    "n03776460",
    "n03777568",
    "n03777754",
    "n03781244",
    "n03782006",
    "n03785016",
    "n03786901",
    "n03787032",
    "n03788195",
    "n03788365",
    "n03791053",
    "n03792782",
    "n03792972",
    "n03793489",
    "n03794056",
    "n03796401",
    "n03803284",
    "n03804744",
    "n03814639",
    "n03814906",
    "n03825788",
    "n03832673",
    "n03837869",
    "n03838899",
    "n03840681",
    "n03841143",
    "n03843555",
    "n03854065",
    "n03857828",
    "n03866082",
    "n03868242",
    "n03868863",
    "n03871628",
    "n03873416",
    "n03874293",
    "n03874599",
    "n03876231",
    "n03877472",
    "n03877845",
    "n03884397",
    "n03887697",
    "n03888257",
    "n03888605",
    "n03891251",
    "n03891332",
    "n03895866",
    "n03899768",
    "n03902125",
    "n03903868",
    "n03908618",
    "n03908714",
    "n03916031",
    "n03920288",
    "n03924679",
    "n03929660",
    "n03929855",
    "n03930313",
    "n03930630",
    "n03933933",
    "n03935335",
    "n03937543",
    "n03938244",
    "n03942813",
    "n03944341",
    "n03947888",
    "n03950228",
    "n03954731",
    "n03956157",
    "n03958227",
    "n03961711",
    "n03967562",
    "n03970156",
    "n03976467",
    "n03976657",
    "n03977966",
    "n03980874",
    "n03982430",
    "n03983396",
    "n03991062",
    "n03992509",
    "n03995372",
    "n03998194",
    "n04004767",
    "n04005630",
    "n04008634",
    "n04009552",
    "n04019541",
    "n04023962",
    "n04026417",
    "n04033901",
    "n04033995",
    "n04037443",
    "n04039381",
    "n04040759",
    "n04041544",
    "n04044716",
    "n04049303",
    "n04065272",
    "n04067472",
    "n04069434",
    "n04070727",
    "n04074963",
    "n04081281",
    "n04086273",
    "n04090263",
    "n04099969",
    "n04111531",
    "n04116512",
    "n04118538",
    "n04118776",
    "n04120489",
    "n04125021",
    "n04127249",
    "n04131690",
    "n04133789",
    "n04136333",
    "n04141076",
    "n04141327",
    "n04141975",
    "n04146614",
    "n04147183",
    "n04149813",
    "n04152593",
    "n04153751",
    "n04154565",
    "n04162706",
    "n04179913",
    "n04192698",
    "n04200800",
    "n04201297",
    "n04204238",
    "n04204347",
    "n04208210",
    "n04209133",
    "n04209239",
    "n04228054",
    "n04229816",
    "n04235860",
    "n04238763",
    "n04239074",
    "n04243546",
    "n04251144",
    "n04252077",
    "n04252225",
    "n04254120",
    "n04254680",
    "n04254777",
    "n04258138",
    "n04259630",
    "n04263257",
    "n04264628",
    "n04265275",
    "n04266014",
    "n04270147",
    "n04273569",
    "n04275548",
    "n04277352",
    "n04285008",
    "n04286575",
    "n04296562",
    "n04310018",
    "n04311004",
    "n04311174",
    "n04317175",
    "n04325704",
    "n04326547",
    "n04328186",
    "n04330267",
    "n04332243",
    "n04335435",
    "n04336792",
    "n04344873",
    "n04346328",
    "n04347754",
    "n04350905",
    "n04355338",
    "n04355933",
    "n04356056",
    "n04357314",
    "n04366367",
    "n04367480",
    "n04370456",
    "n04371430",
    "n04371774",
    "n04372370",
    "n04376876",
    "n04380533",
    "n04389033",
    "n04392985",
    "n04398044",
    "n04399382",
    "n04404412",
    "n04409515",
    "n04417672",
    "n04418357",
    "n04423845",
    "n04428191",
    "n04429376",
    "n04435653",
    "n04442312",
    "n04443257",
    "n04447861",
    "n04456115",
    "n04458633",
    "n04461696",
    "n04462240",
    "n04465501",
    "n04467665",
    "n04476259",
    "n04479046",
    "n04482393",
    "n04483307",
    "n04485082",
    "n04486054",
    "n04487081",
    "n04487394",
    "n04493381",
    "n04501370",
    "n04505470",
    "n04507155",
    "n04509417",
    "n04515003",
    "n04517823",
    "n04522168",
    "n04523525",
    "n04525038",
    "n04525305",
    "n04532106",
    "n04532670",
    "n04536866",
    "n04540053",
    "n04542943",
    "n04548280",
    "n04548362",
    "n04550184",
    "n04552348",
    "n04553703",
    "n04554684",
    "n04557648",
    "n04560804",
    "n04562935",
    "n04579145",
    "n04579432",
    "n04584207",
    "n04589890",
    "n04590129",
    "n04591157",
    "n04591713",
    "n04592741",
    "n04596742",
    "n04597913",
    "n04599235",
    "n04604644",
    "n04606251",
    "n04612504",
    "n04613696",
    "n06359193",
    "n06596364",
    "n06785654",
    "n06794110",
    "n06874185",
    "n07248320",
    "n07565083",
    "n07579787",
    "n07583066",
    "n07584110",
    "n07590611",
    "n07613480",
    "n07614500",
    "n07615774",
    "n07684084",
    "n07693725",
    "n07695742",
    "n07697313",
    "n07697537",
    "n07711569",
    "n07714571",
    "n07714990",
    "n07715103",
    "n07716358",
    "n07716906",
    "n07717410",
    "n07717556",
    "n07718472",
    "n07718747",
    "n07720875",
    "n07730033",
    "n07734744",
    "n07742313",
    "n07745940",
    "n07747607",
    "n07749582",
    "n07753113",
    "n07753275",
    "n07753592",
    "n07754684",
    "n07760859",
    "n07768694",
    "n07802026",
    "n07831146",
    "n07836838",
    "n07860988",
    "n07871810",
    "n07873807",
    "n07875152",
    "n07880968",
    "n07892512",
    "n07920052",
    "n07930864",
    "n07932039",
    "n09193705",
    "n09229709",
    "n09246464",
    "n09256479",
    "n09288635",
    "n09332890",
    "n09399592",
    "n09421951",
    "n09428293",
    "n09468604",
    "n09472597",
    "n09835506",
    "n10148035",
    "n10565667",
    "n11879895",
    "n11939491",
    "n12057211",
    "n12144580",
    "n12267677",
    "n12620546",
    "n12768682",
    "n12985857",
    "n12998815",
    "n13037406",
    "n13040303",
    "n13044778",
    "n13052670",
    "n13054560",
    "n13133613",
    "n15075141",
]

# Build synset -> index mapping
SYNSET_TO_IDX = {synset: idx for idx, synset in enumerate(IMAGENET_SYNSETS)}


# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════


def cleanup_model(model, device: str) -> None:
    """Aggressively free model memory."""
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def delete_cached_weights(model_name: str, pretrained_tag: str) -> None:
    """
    Delete downloaded model weights from all known cache directories.
    """
    home = os.path.expanduser("~")
    deleted = []

    # Hugging Face hub cache
    hf_cache = os.path.join(home, ".cache", "huggingface", "hub")
    if os.path.isdir(hf_cache):
        for entry in os.listdir(hf_cache):
            entry_path = os.path.join(hf_cache, entry)
            if not os.path.isdir(entry_path):
                continue
            entry_lower = entry.lower()
            model_parts = model_name.lower().replace("-", "").replace("_", "")
            tag_parts = pretrained_tag.lower().replace("-", "").replace("_", "")
            entry_clean = entry_lower.replace("-", "").replace("_", "")
            if model_parts in entry_clean and (
                tag_parts in entry_clean or "clip" in entry_lower
            ):
                try:
                    shutil.rmtree(entry_path)
                    deleted.append(entry_path)
                except Exception as e:
                    print(f"    [warn] Could not delete {entry_path}: {e}")

    # OpenCLIP / CLIP cache
    for cache_name in [".cache/clip", ".cache/open_clip"]:
        cache_dir = os.path.join(home, cache_name)
        if os.path.isdir(cache_dir):
            for entry in os.listdir(cache_dir):
                entry_path = os.path.join(cache_dir, entry)
                if os.path.isfile(entry_path) and entry.endswith(
                    (".pt", ".pth", ".bin", ".safetensors")
                ):
                    try:
                        os.remove(entry_path)
                        deleted.append(entry_path)
                    except Exception as e:
                        print(f"    [warn] Could not delete {entry_path}: {e}")

    # Torch hub cache
    torch_cache = os.path.join(home, ".cache", "torch", "hub", "checkpoints")
    if os.path.isdir(torch_cache):
        for entry in os.listdir(torch_cache):
            entry_path = os.path.join(torch_cache, entry)
            entry_lower = entry.lower()
            model_parts = model_name.lower().replace("-", "_")
            if model_parts in entry_lower or pretrained_tag.lower() in entry_lower:
                try:
                    os.remove(entry_path)
                    deleted.append(entry_path)
                except Exception as e:
                    print(f"    [warn] Could not delete {entry_path}: {e}")

    if deleted:
        print(f"    [cleanup] Removed {len(deleted)} cached file(s)/dir(s) from disk.")
    else:
        print(f"    [cleanup] No cached weight files found to delete.")


def parse_model_info(model_name: str, pretrained_tag: str) -> dict:
    """Extract architecture details from the model/pretrained names."""
    info = {
        "model_name": model_name,
        "pretrained_tag": pretrained_tag,
        "model_id": f"{model_name}_{pretrained_tag}",
    }

    # Architecture family
    if model_name.startswith("RN"):
        info["arch_family"] = "ResNet"
        m = re.match(r"RN(\d+)(x\d+)?", model_name)
        if m:
            info["resnet_depth"] = int(m.group(1))
            info["resnet_multiplier"] = m.group(2) or "x1"
        info["patch_size"] = None
    elif "ViT" in model_name or "vit" in model_name.lower():
        info["arch_family"] = "ViT"
        size_m = re.search(r"ViT-([a-zA-Z]+)-(\d+)", model_name)
        if size_m:
            info["vit_size"] = size_m.group(1)
            info["patch_size"] = int(size_m.group(2))
        else:
            info["vit_size"] = None
            info["patch_size"] = None
    elif "coca" in model_name.lower():
        info["arch_family"] = "CoCa"
        size_m = re.search(r"ViT-([a-zA-Z]+)-(\d+)", model_name)
        if size_m:
            info["vit_size"] = size_m.group(1)
            info["patch_size"] = int(size_m.group(2))
    elif "convnext" in model_name.lower():
        info["arch_family"] = "ConvNeXt"
        info["patch_size"] = None
    elif "EVA" in model_name:
        info["arch_family"] = "EVA"
        size_m = re.search(r"-(\d+)", model_name)
        if size_m:
            info["patch_size"] = int(size_m.group(1))
    elif "MobileCLIP" in model_name:
        info["arch_family"] = "MobileCLIP"
        info["patch_size"] = None
    elif "SigLIP" in model_name or "siglip" in model_name.lower():
        info["arch_family"] = "SigLIP"
        info["patch_size"] = None
    else:
        info["arch_family"] = "Other"
        info["patch_size"] = None

    # Image size from model name
    img_m = re.search(r"(\d{3,4})$", model_name.replace("-", ""))
    if img_m and int(img_m.group(1)) >= 200:
        info["image_size"] = int(img_m.group(1))
    else:
        info["image_size"] = None

    # Training dataset from pretrained tag
    tag = pretrained_tag.lower()
    if "openai" in tag:
        info["training_data"] = "OpenAI WIT (400M)"
    elif "laion2b" in tag:
        info["training_data"] = "LAION-2B"
    elif "laion400m" in tag:
        info["training_data"] = "LAION-400M"
    elif "datacomp" in tag:
        info["training_data"] = "DataComp"
    elif "metaclip" in tag:
        info["training_data"] = "MetaCLIP"
    elif "dfn" in tag:
        info["training_data"] = "DFN"
    elif "commonpool" in tag:
        info["training_data"] = "CommonPool"
    elif "yfcc" in tag:
        info["training_data"] = "YFCC-15M"
    elif "cc12m" in tag:
        info["training_data"] = "CC-12M"
    elif "webli" in tag:
        info["training_data"] = "WebLI"
    elif "merged" in tag:
        info["training_data"] = "Merged"
    elif "mscoco" in tag:
        info["training_data"] = "MS-COCO (finetuned)"
    else:
        info["training_data"] = pretrained_tag

    info["quickgelu"] = "quickgelu" in model_name.lower()

    return info


# ═══════════════════════════════════════════════════════════════════
# ImageNet Dataset (flat structure with LOC_val_solution.csv)
# ═══════════════════════════════════════════════════════════════════


class ImageNetValDataset(torch.utils.data.Dataset):
    """
    ImageNet validation dataset with flat structure.

    Expected structure:
        data_root/ILSVRC2012_val_00000001.JPEG
        data_root/ILSVRC2012_val_00000002.JPEG
        ...

    Labels file (LOC_val_solution.csv):
        ImageId,PredictionString
        ILSVRC2012_val_00000001,n01751748 ...
        ...

    The synset ID (e.g., n01751748) is mapped to class index (0-999).
    """

    def __init__(self, data_root: str, labels_file: str, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []  # list of (image_path, class_idx)

        # Parse labels file
        print(f"  Loading labels from {labels_file}...")
        labels_df = pd.read_csv(labels_file)

        # Build image_id -> synset mapping
        image_to_synset = {}
        for _, row in labels_df.iterrows():
            image_id = row["ImageId"]
            # PredictionString format: "n03995372 85 1 499 272" - first token is synset
            pred_str = row["PredictionString"]
            synset = pred_str.split()[0]
            image_to_synset[image_id] = synset

        # Find all images in data_root
        all_images = sorted(
            [
                f
                for f in os.listdir(data_root)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        )

        missing_labels = 0
        unknown_synsets = set()

        for img_name in all_images:
            # Extract image_id (e.g., "ILSVRC2012_val_00000001" from "ILSVRC2012_val_00000001.JPEG")
            image_id = os.path.splitext(img_name)[0]

            if image_id not in image_to_synset:
                missing_labels += 1
                continue

            synset = image_to_synset[image_id]

            if synset not in SYNSET_TO_IDX:
                unknown_synsets.add(synset)
                continue

            class_idx = SYNSET_TO_IDX[synset]
            img_path = os.path.join(data_root, img_name)
            self.samples.append((img_path, class_idx))

        print(f"  Found {len(self.samples)} images with valid labels")
        if missing_labels > 0:
            print(f"  WARNING: {missing_labels} images missing from labels file")
        if unknown_synsets:
            print(
                f"  WARNING: {len(unknown_synsets)} unknown synsets: {list(unknown_synsets)[:5]}..."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# ═══════════════════════════════════════════════════════════════════
# Cross-model summary plots
# ═══════════════════════════════════════════════════════════════════


def plot_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Generate cross-model comparison plots from the master DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    if df.empty:
        return

    # ── 1. Top-1 accuracy bar chart ────────────────────────────
    df_sorted = df.sort_values("top1_accuracy", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) * 0.35)))
    y_pos = np.arange(len(df_sorted))
    colors = plt.cm.RdYlGn(df_sorted["top1_accuracy"].values / 100)
    ax.barh(
        y_pos,
        df_sorted["top1_accuracy"],
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model_id"], fontsize=6)
    ax.set_xlabel("Top-1 Accuracy (%)")
    ax.set_title("ImageNet Zero-Shot Top-1 Accuracy — All OpenCLIP Models")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_models_top1_accuracy.png"), dpi=150)
    plt.close(fig)

    # ── 2. Top-5 accuracy bar chart ────────────────────────────
    df_sorted_top5 = df.sort_values("top5_accuracy", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted_top5) * 0.35)))
    y_pos = np.arange(len(df_sorted_top5))
    colors = plt.cm.RdYlGn(df_sorted_top5["top5_accuracy"].values / 100)
    ax.barh(
        y_pos,
        df_sorted_top5["top5_accuracy"],
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted_top5["model_id"], fontsize=6)
    ax.set_xlabel("Top-5 Accuracy (%)")
    ax.set_title("ImageNet Zero-Shot Top-5 Accuracy — All OpenCLIP Models")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "all_models_top5_accuracy.png"), dpi=150)
    plt.close(fig)

    # ── 3. Top-1 vs Top-5 scatter ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        df["top1_accuracy"],
        df["top5_accuracy"],
        c="steelblue",
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        s=40,
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["model_id"],
            (row["top1_accuracy"], row["top5_accuracy"]),
            fontsize=4,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("Top-1 Accuracy (%)")
    ax.set_ylabel("Top-5 Accuracy (%)")
    ax.set_title("Top-1 vs Top-5 Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "top1_vs_top5.png"), dpi=150)
    plt.close(fig)

    # ── 4. Grouped by architecture family ────────────────────────
    if "arch_family" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, col, title in zip(
            axes,
            ["top1_accuracy", "top5_accuracy"],
            ["Top-1 Accuracy", "Top-5 Accuracy"],
        ):
            grouped = df.groupby("arch_family")[col].agg(["mean", "std", "count"])
            grouped = grouped.sort_values("mean", ascending=True)
            ax.barh(
                grouped.index,
                grouped["mean"],
                xerr=grouped["std"],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xlabel(f"{title} (%)")
            ax.set_title(f"{title} by Architecture")
            ax.grid(True, alpha=0.3, axis="x")
            for i, (idx, row) in enumerate(grouped.iterrows()):
                ax.text(
                    row["mean"] + 1,
                    i,
                    f"n={int(row['count'])}",
                    va="center",
                    fontsize=8,
                )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "accuracy_by_architecture.png"), dpi=150)
        plt.close(fig)

    # ── 5. Params vs Top-1 accuracy ────────────────────────────
    if "total_params_M" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            df["total_params_M"],
            df["top1_accuracy"],
            c="steelblue",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            s=40,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("Model Size vs Top-1 Accuracy")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "params_vs_top1.png"), dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Single-model analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_single_model(
    model_name: str,
    pretrained_tag: str,
    dataset: ImageNetValDataset,
    device: str,
    batch_size: int,
    num_workers: int,
    output_dir: str,
) -> dict | None:
    """
    Run ImageNet zero-shot evaluation for one model.
    Returns a results dict or None on failure.
    """
    model_id = f"{model_name}_{pretrained_tag}"
    info = parse_model_info(model_name, pretrained_tag)

    print(f"\n{'═' * 70}")
    print(f"  MODEL: {model_name} / {pretrained_tag}")
    print(f"{'═' * 70}")

    model = None
    try:
        # ── Load model ───────────────────────────────────────────
        t0 = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_tag, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        load_time = time.time() - t0

        # Get actual model config details
        if hasattr(model, "visual"):
            visual = model.visual
            if hasattr(visual, "image_size"):
                img_size = visual.image_size
                if isinstance(img_size, (list, tuple)):
                    info["image_size"] = img_size[0]
                else:
                    info["image_size"] = img_size
            if hasattr(visual, "output_dim"):
                info["embed_dim"] = visual.output_dim
            elif hasattr(model, "embed_dim"):
                info["embed_dim"] = model.embed_dim

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        info["total_params_M"] = round(total_params / 1e6, 1)

        print(
            f"  Loaded in {load_time:.1f}s | {info['total_params_M']}M params | "
            f"img_size={info.get('image_size', '?')} | embed_dim={info.get('embed_dim', '?')}"
        )

        # ── Build zero-shot classifier ───────────────────────────
        print("  Building zero-shot classifier...")
        t1 = time.time()
        with torch.no_grad(), torch.amp.autocast(device):
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )
        classifier_time = time.time() - t1
        print(
            f"  Classifier built in {classifier_time:.1f}s | shape={classifier.shape}"
        )

        # ── Create dataloader with this model's preprocess ───────
        # We need to update the transform for each model
        dataset.transform = preprocess
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # ── Evaluate ─────────────────────────────────────────────
        print("  Evaluating...")
        t2 = time.time()

        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="  Eval", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                with torch.amp.autocast(device):
                    image_features = model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100.0 * image_features @ classifier

                # Compute accuracy
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                batch_size_actual = labels.size(0)
                top1_correct += acc1 * batch_size_actual / 100
                top5_correct += acc5 * batch_size_actual / 100
                total += batch_size_actual

        eval_time = time.time() - t2

        top1_acc = 100 * top1_correct / total
        top5_acc = 100 * top5_correct / total

        print(
            f"\n  ✓ Top-1: {top1_acc:.2f}%  Top-5: {top5_acc:.2f}%  "
            f"({total} images in {eval_time:.1f}s)"
        )

        # Build result row
        result = {
            **info,
            "top1_accuracy": round(top1_acc, 3),
            "top5_accuracy": round(top5_acc, 3),
            "total_images": total,
            "load_time_s": round(load_time, 1),
            "classifier_time_s": round(classifier_time, 1),
            "eval_time_s": round(eval_time, 1),
        }

        # ── Save per-model JSON ──────────────────────────────────
        model_dir = os.path.join(output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None

    finally:
        # ── CLEANUP: delete model from GPU/CPU memory ────────────
        if model is not None:
            cleanup_model(model, device)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  [cleanup] Model memory freed.")

        # ── CLEANUP: delete cached weights from disk ─────────────
        delete_cached_weights(model_name, pretrained_tag)
        print(f"  [cleanup] Done.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Run ImageNet zero-shot analysis across all OpenCLIP pretrained models."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Path to ImageNet validation set (flat folder with ILSVRC2012_val_*.JPEG)",
    )
    parser.add_argument(
        "--labels-file",
        default=DEFAULT_LABELS_FILE,
        help="Path to LOC_val_solution.csv with ground truth labels",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run models whose name contains this string.",
    )
    parser.add_argument(
        "--exclude", default=None, help="Skip models whose name contains this string."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models that already have results.json.",
    )
    parser.add_argument(
        "--max-params",
        type=float,
        default=None,
        help="Skip models with more than this many million parameters.",
    )
    parser.add_argument(
        "--no-delete-cache",
        action="store_true",
        help="Don't delete model weights from cache after evaluation.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Verify data ──────────────────────────────────────────────
    if not os.path.isdir(args.data_root):
        print(f"ERROR: Data root not found: {args.data_root}")
        sys.exit(1)

    if not os.path.isfile(args.labels_file):
        print(f"ERROR: Labels file not found: {args.labels_file}")
        sys.exit(1)

    # Count images
    all_images = [
        f
        for f in os.listdir(args.data_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    print(f"Found {len(all_images)} images in {args.data_root}")

    if len(all_images) != 50000:
        print(f"WARNING: Expected 50000 images, found {len(all_images)}")

    # ── Create dataset (without transform initially) ─────────────
    # Transform will be set per-model in analyze_single_model
    dataset = ImageNetValDataset(args.data_root, args.labels_file, transform=None)

    # ── Enumerate all pretrained models ──────────────────────────
    all_pretrained = open_clip.list_pretrained()
    print(
        f"\nOpenCLIP has {len(all_pretrained)} pretrained model/weight combinations.\n"
    )

    # Apply filters
    filtered = []
    for model_name, pretrained_tag in all_pretrained:
        model_id = f"{model_name}_{pretrained_tag}"

        if args.filter and args.filter.lower() not in model_id.lower():
            continue
        if args.exclude and args.exclude.lower() in model_id.lower():
            continue
        if args.resume:
            result_path = os.path.join(args.output_dir, model_id, "results.json")
            if os.path.exists(result_path):
                print(f"  [skip] {model_id} — already completed.")
                continue
        filtered.append((model_name, pretrained_tag))

    print(f"Models to run: {len(filtered)}")
    if not filtered:
        print("Nothing to do.")
        if args.resume:
            _generate_summary_from_existing(args.output_dir)
        return

    # ── Run analysis for each model ──────────────────────────────
    all_results = []

    # Load existing results if resuming
    if args.resume:
        all_results = _load_existing_results(args.output_dir)
        print(f"Loaded {len(all_results)} existing results.\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, (model_name, pretrained_tag) in enumerate(filtered):
        print(f"\n[{idx + 1}/{len(filtered)}] ", end="")

        result = analyze_single_model(
            model_name=model_name,
            pretrained_tag=pretrained_tag,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
        )

        if result is not None:
            all_results.append(result)

            # Incrementally save master table
            df = pd.DataFrame(all_results)
            df = df.sort_values("top1_accuracy", ascending=False)
            df.to_csv(os.path.join(args.output_dir, "master_results.csv"), index=False)
            with open(os.path.join(args.output_dir, "master_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Final summary ────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values("top1_accuracy", ascending=False)
        df.to_csv(os.path.join(args.output_dir, "master_results.csv"), index=False)
        with open(os.path.join(args.output_dir, "master_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        summary_dir = os.path.join(args.output_dir, "summary_plots")
        plot_summary(df, summary_dir)

        # Print top-level summary
        print(f"\n\n{'═' * 80}")
        print("FINAL SUMMARY")
        print(f"{'═' * 80}")
        print(f"Models analyzed: {len(df)}")
        print(
            f"Failed models: {len(filtered) - len([r for r in all_results if r is not None])}"
        )

        print(f"\nTop 10 by Top-1 Accuracy:")
        cols = [
            "model_id",
            "arch_family",
            "total_params_M",
            "image_size",
            "training_data",
            "top1_accuracy",
            "top5_accuracy",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].head(10).to_string(index=False))

        print(f"\nTop 10 by Top-5 Accuracy:")
        print(
            df.sort_values("top5_accuracy", ascending=False)[cols]
            .head(10)
            .to_string(index=False)
        )

        print(f"\nResults saved to:")
        print(f"  {args.output_dir}/master_results.csv")
        print(f"  {args.output_dir}/master_results.json")
        print(f"  {summary_dir}/ (comparison plots)")


def _load_existing_results(output_dir: str) -> list[dict]:
    """Load all existing results.json files from model subdirectories."""
    results = []
    if not os.path.isdir(output_dir):
        return results
    for entry in os.listdir(output_dir):
        rpath = os.path.join(output_dir, entry, "results.json")
        if os.path.isfile(rpath):
            try:
                with open(rpath) as f:
                    results.append(json.load(f))
            except Exception:
                pass
    return results


def _generate_summary_from_existing(output_dir: str) -> None:
    """Regenerate summary from existing per-model results."""
    results = _load_existing_results(output_dir)
    if not results:
        print("No existing results found.")
        return
    df = pd.DataFrame(results)
    df = df.sort_values("top1_accuracy", ascending=False)
    df.to_csv(os.path.join(output_dir, "master_results.csv"), index=False)
    with open(os.path.join(output_dir, "master_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    summary_dir = os.path.join(output_dir, "summary_plots")
    plot_summary(df, summary_dir)
    print(f"Regenerated summary from {len(results)} existing results.")


if __name__ == "__main__":
    main()
