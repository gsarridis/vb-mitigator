#!/bin/bash

# List of methods
methods=("erm" "badd" "flac" "maviasb" "bb" "debian" "di" "end" "groupdro" "jtt" "lff" "sd")

# Loop over methods
for method in "${methods[@]}"
do
    cfg_file="configs/ucf101/${method}/dev.yaml"
    echo "Running method: $method with config $cfg_file"
    
    python tools/train.py --cfg "$cfg_file" --eval

    echo "Finished method: $method"
    echo "--------------------------"
done
