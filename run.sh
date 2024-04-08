#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

datasets=("hESC" "hHep" "mDC" "mESC" "mHSC-E" "mHSC-GM" "mHSC-L")
values=("500" "1000")
tasks=("specific" "non_specific" "STRING")

for task in "${tasks[@]}"; do
    for value in "${values[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "Processing dataset: $dataset with value: $value and task: $task"

            output_dir="./Output/output_${value}_${dataset}"
            mkdir -p $output_dir

            if [[ $dataset == h* ]]; then
                prior_network_path="./Prior_network/network_human.csv"
            else
                prior_network_path="./Prior_network/network_mouse.csv"
            fi

            python main.py --dataset_name ${dataset} --data_path "./Dataset/${dataset}/tfs+${value}/${task}/ExpressionData.csv" \
            --gt_path "./Dataset/${dataset}/tfs+${value}/${task}/network.csv" --prior_network ${prior_network_path} --time_info "./Dataset/${dataset}/PseudoTime.csv" \
            --output_dir ${output_dir}
        done
    done
done