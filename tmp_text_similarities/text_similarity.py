#!/usr/bin/env python3
"""
Extract features from 3 texts and 2 images using OpenCLIP,
then print all pairwise cosine similarities.

Usage:
    python text_similarity.py
    python text_similarity.py --texts "cat" "dog" "car" --images img1.jpg img2.jpg
    python text_similarity.py --model ViT-L-14 --pretrained laion2b_s32b_b82k

Requirements:
    pip install torch open-clip-torch Pillow
"""

import argparse
from itertools import combinations

import torch
import torch.nn.functional as F
from PIL import Image

import open_clip


# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_TEXTS = [
    "a photo of country environment",
    "a photo of an urban environment",
    "a photo of a pickup truck car",
    "a photo of a sports car",
    "a photo of an animal",
    "a photo of a fireplug",
    # "a picture of a stop sign | traffic sign | stop sign pole | tra ffic light | traffic light pole | power lines | electrical wires | building | brick building | stop | street | urban",
    # "a picture of a mountain | hill | dirt road | grass | sheep | goat | bird | mountain range | sunset | sky | clouds | dusk | evening | mountain top",
]
# DEFAULT_IMAGES = [
#     "../data/urbancars/bg-0.5_co_occur_obj-0.5/test/obj-urban_bg-urban_co_occur_obj-urban/008.jpg",
#     "../data/urbancars/bg-0.5_co_occur_obj-0.5/test/obj-urban_bg-country_co_occur_obj-country/007.jpg",
# ]
# DEFAULT_TEXTS = [
#     "a photo of a jeep",
#     "a photograph of a AM General Hummer SUV 2000, Aston Martin V8 Vantage Convertible 2012, BMW X5 SUV 2007, BMW X6 SUV 2012, BMW X3 SUV 2012, Buick Rainier SUV 2007, Buick Enclave SUV 2012, Cadillac SRX SUV 2012, Cadillac Escalade EXT Crew Cab 2007, Chevrolet Silverado 1500 Hybrid Crew Cab 2012, Chevrolet Traverse SUV 2012, Chevrolet HHR SS 2010, Chevrolet Tahoe Hybrid SUV 2012, Chevrolet Express Cargo Van 2007, Chevrolet Avalanche Crew Cab 2012, Chevrolet TrailBlazer SS 2009, Chevrolet Silverado 2500HD Regular Cab 2012, Chevrolet Silverado 1500 Classic Extended Cab 2007, Chevrolet Express Van 2007, Chevrolet Silverado 1500 Extended Cab 2012, Chevrolet Silverado 1500 Regular Cab 2012, Chrysler Aspen SUV 2009, Chrysler Town and Country Minivan 2012, Dodge Caravan Minivan 1997, Dodge Ram Pickup 3500 Crew Cab 2010, Dodge Ram Pickup 3500 Quad Cab 2009, Dodge Sprinter Cargo Van 2009, Dodge Journey SUV 2012, Dodge Dakota Crew Cab 2010, Dodge Dakota Club Cab 2007, Dodge Durango SUV 2012, Dodge Durango SUV 2007, Ford F-450 Super Duty Crew Cab 2012, Ford Freestar Minivan 2007, Ford Expedition EL SUV 2009, Ford Edge SUV 2012, Ford Ranger SuperCab 2011, Ford F-150 Regular Cab 2012, Ford F-150 Regular Cab 2007, Ford E-Series Wagon Van 2012, GMC Terrain SUV 2012, GMC Savana Van 2012, GMC Yukon Hybrid SUV 2012, GMC Acadia SUV 2012, GMC Canyon Extended Cab 2012, HUMMER H3T Crew Cab 2010, HUMMER H2 SUT Crew Cab 2009, Honda Odyssey Minivan 2012, Honda Odyssey Minivan 2007, Hyundai Santa Fe SUV 2012, Hyundai Tucson SUV 2012, Hyundai Veracruz SUV 2012, Infiniti QX56 SUV 2011, Isuzu Ascender SUV 2008, Jeep Patriot SUV 2012, Jeep Wrangler SUV 2012, Jeep Liberty SUV 2012, Jeep Grand Cherokee SUV 2012, Jeep Compass SUV 2012, Land Rover Range Rover SUV 2012, Land Rover LR2 SUV 2012, Mazda Tribute SUV 2011, Mercedes-Benz Sprinter Van 2012, Nissan NV Passenger Van 2012, Ram C/V Cargo Van Minivan 2012, Toyota Sequoia SUV 2012, Toyota 4Runner SUV 2012, Volvo XC90 SUV 2007",
#     "a photograph of a Acura RL Sedan 2012, Acura TL Sedan 2012, Acura TL Type-S 2008, Acura TSX Sedan 2012, Acura Integra Type R 2001, Acura ZDX Hatchback 2012, Aston Martin V8 Vantage Coupe 2012, Aston Martin Virage Convertible 2012, Aston Martin Virage Coupe 2012, Audi RS 4 Convertible 2008, Audi A5 Coupe 2012, Audi TTS Coupe 2012, Audi R8 Coupe 2012, Audi V8 Sedan 1994, Audi 100 Sedan 1994, Audi 100 Wagon 1994, Audi TT Hatchback 2011, Audi S6 Sedan 2011, Audi S5 Convertible 2012, Audi S5 Coupe 2012, Audi S4 Sedan 2012, Audi S4 Sedan 2007, Audi TT RS Coupe 2012, BMW ActiveHybrid 5 Sedan 2012, BMW 1 Series Convertible 2012, BMW 1 Series Coupe 2012, BMW 3 Series Sedan 2012, BMW 3 Series Wagon 2012, BMW 6 Series Convertible 2007, BMW M3 Coupe 2012, BMW M5 Sedan 2010, BMW M6 Convertible 2010, BMW Z4 Convertible 2012, Bentley Continental Supersports Conv. Convertible 2012, Bentley Arnage Sedan 2009, Bentley Mulsanne Sedan 2011, Bentley Continental GT Coupe 2012, Bentley Continental GT Coupe 2007, Bentley Continental Flying Spur Sedan 2007, Bugatti Veyron 16.4 Convertible 2009, Bugatti Veyron 16.4 Coupe 2009, Buick Regal GS 2012, Buick Verano Sedan 2012, Cadillac CTS-V Sedan 2012, Chevrolet Corvette Convertible 2012, Chevrolet Corvette ZR1 2012, Chevrolet Corvette Ron Fellows Edition Z06 2007, Chevrolet Camaro Convertible 2012, Chevrolet Impala Sedan 2007, Chevrolet Sonic Sedan 2012, Chevrolet Cobalt SS 2010, Chevrolet Malibu Hybrid Sedan 2010, Chevrolet Monte Carlo Coupe 2007, Chevrolet Malibu Sedan 2007, Chrysler Sebring Convertible 2010, Chrysler 300 SRT-8 2010, Chrysler Crossfire Convertible 2008, Chrysler PT Cruiser Convertible 2008, Daewoo Nubira Wagon 2002, Dodge Caliber Wagon 2012, Dodge Caliber Wagon 2007, Dodge Magnum Wagon 2008, Dodge Challenger SRT8 2011, Dodge Charger Sedan 2012, Dodge Charger SRT-8 2009, Eagle Talon Hatchback 1998, FIAT 500 Abarth 2012, FIAT 500 Convertible 2012, Ferrari FF Coupe 2012, Ferrari California Convertible 2012, Ferrari 458 Italia Convertible 2012, Ferrari 458 Italia Coupe 2012, Fisker Karma Sedan 2012, Ford Mustang Convertible 2007, Ford GT Coupe 2006, Ford Focus Sedan 2007, Ford Fiesta Sedan 2012, Geo Metro Convertible 1993, Honda Accord Coupe 2012, Honda Accord Sedan 2012, Hyundai Veloster Hatchback 2012, Hyundai Sonata Hybrid Sedan 2012, Hyundai Elantra Sedan 2007, Hyundai Accent Sedan 2012, Hyundai Genesis Sedan 2012, Hyundai Sonata Sedan 2012, Hyundai Elantra Touring Hatchback 2012, Hyundai Azera Sedan 2012, Infiniti G Coupe IPL2012, Jaguar XK XKR 2012, Lamborghini Reventon Coupe 2008, Lamborghini Aventador Coupe 2012, Lamborghini Gallardo LP 570-4 Superleggera 2012, Lamborghini Diablo Coupe 2001, Lincoln Town Car Sedan 2011, MINI Cooper Roadster Convertible 2012, Maybach Landaulet Convertible 2012, McLaren MP4-12C Coupe 2012, Mercedes-Benz 300-Class Convertible 1993, Mercedes-Benz C-Class Sedan 2012, Mercedes-Benz SL-Class Coupe 2009, Mercedes-Benz E-Class Sedan 2012, Mercedes-Benz S-Class Sedan 2012, Mitsubishi Lancer Sedan 2012, Nissan Leaf Hatchback 2012, Nissan Juke Hatchback 2012, Nissan 240SX Coupe 1998, Plymouth Neon Coupe 1999, Porsche Panamera Sedan 2012, Rolls-Royce Phantom Drophead Coupe Convertible 2012, Rolls-Royce Ghost Sedan 2012, Rolls-Royce Phantom Sedan 2012, Scion xD Hatchback 2012, Spyker C8 Convertible 2009, Spyker C8 Coupe 2009, Suzuki Aerio Sedan 2007, Suzuki Kizashi Sedan 2012, Suzuki SX4 Hatchback 2012, Suzuki SX4 Sedan 2012, Tesla Model S Sedan 2012, Toyota Camry Sedan 2012, Toyota Corolla Sedan 2012, Volkswagen Golf Hatchback 2012, Volkswagen Golf Hatchback 1991, Volkswagen Beetle Hatchback 2012, Volvo C30 Hatchback 2012, Volvo 240 Sedan 1993, smart fortwo Convertible 2012",
# ]
# DEFAULT_IMAGES = [
#     "../data/urbancars/bg-0.5_co_occur_obj-0.5/test/obj-country_bg-country_co_occur_obj-country/122.jpg",
#     "../data/urbancars/bg-0.5_co_occur_obj-0.5/test/obj-country_bg-urban_co_occur_obj-urban/071.jpg",
# ]
DEFAULT_IMAGES = [
    "../data/urbancars/bg-0.5_co_occur_obj-0.5/test/obj-urban_bg-urban_co_occur_obj-urban/075.jpg",
    "../data/urbancars/bg-0.5_co_occur_obj-0.5/test/obj-urban_bg-country_co_occur_obj-country/023.jpg",
]


# DEFAULT_MODEL = "ViT-B-32"
# DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_MODEL = "ViT-B-16"
DEFAULT_PRETRAINED = "openai"


def encode_texts(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """Tokenize and encode texts into normalized feature vectors."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        features = model.encode_text(tokens)
    return F.normalize(features, dim=-1)


def encode_images(
    model, preprocess, image_paths: list[str], device: str
) -> torch.Tensor:
    """Load, preprocess, and encode images into normalized feature vectors."""
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(preprocess(img))
    batch = torch.stack(images).to(device)
    with torch.no_grad(), torch.amp.autocast(device):
        features = model.encode_image(batch)
    return F.normalize(features, dim=-1)


def print_similarity_matrix(labels: list[str], sim: torch.Tensor) -> None:
    """Pretty-print the full similarity matrix."""
    n = len(labels)
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    for i in range(n):
        row = f"{labels[i]:<{col_w}}" + "".join(
            f"{sim[i, j].item():>{col_w}.4f}" for j in range(n)
        )
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise cosine similarity of 3 texts + 2 images via OpenCLIP."
    )
    parser.add_argument(
        "--texts",
        nargs=4,
        default=DEFAULT_TEXTS,
        help="Three input texts (default: built-in examples).",
    )
    parser.add_argument(
        "--images",
        nargs=2,
        default=DEFAULT_IMAGES,
        help="Two image file paths (default: image1.jpg, image2.jpg).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenCLIP architecture (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help=f"Pretrained weights tag (default: {DEFAULT_PRETRAINED})",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : {args.model} / {args.pretrained}\n")

    # ── Load model ───────────────────────────────────────────────
    print("Loading model …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    # ── Encode texts ─────────────────────────────────────────────
    print("Encoding texts …")
    text_features = encode_texts(model, tokenizer, args.texts, device)

    # ── Encode images ────────────────────────────────────────────
    print("Encoding images …\n")
    image_features = encode_images(model, preprocess, args.images, device)

    # ── Combine all features ─────────────────────────────────────
    # Order: T0, T1, T2, I0, I1
    all_features = torch.cat([text_features, image_features], dim=0)
    labels = [f"T{i}" for i in range(len(args.texts))] + [
        f"I{i}" for i in range(len(args.images))
    ]
    descriptions = [f'"{t}"' for t in args.texts] + list(args.images)

    n = len(labels)
    sim_matrix = all_features @ all_features.T

    # ── Print inputs ─────────────────────────────────────────────
    print(f"Feature dim: {all_features.shape[1]}-d")
    print(f"Inputs ({n}):")
    for lbl, desc in zip(labels, descriptions):
        print(f"  {lbl} : {desc}")
    print()

    # ── Pairwise similarities ────────────────────────────────────
    print("Pairwise cosine similarities:")
    print("-" * 55)

    # Text ↔ Text
    print("\n  Text ↔ Text:")
    for i, j in combinations(range(len(args.texts)), 2):
        s = sim_matrix[i, j].item()
        print(
            f"    {labels[i]} ↔ {labels[j]} : {s:.4f}  ({descriptions[i]}  ↔  {descriptions[j]})"
        )

    # Image ↔ Image
    ti = len(args.texts)  # offset for image indices
    print("\n  Image ↔ Image:")
    for i, j in combinations(range(len(args.images)), 2):
        gi, gj = ti + i, ti + j
        s = sim_matrix[gi, gj].item()
        print(
            f"    {labels[gi]} ↔ {labels[gj]} : {s:.4f}  ({descriptions[gi]}  ↔  {descriptions[gj]})"
        )

    # Text ↔ Image (cross-modal)
    print("\n  Text ↔ Image:")
    for i in range(len(args.texts)):
        for j in range(len(args.images)):
            gj = ti + j
            s = sim_matrix[i, gj].item()
            print(
                f"    {labels[i]} ↔ {labels[gj]} : {s:.4f}  ({descriptions[i]}  ↔  {descriptions[gj]})"
            )

    # ── Full matrix ──────────────────────────────────────────────
    print("\nFull similarity matrix:\n")
    print_similarity_matrix(labels, sim_matrix)


if __name__ == "__main__":
    main()
