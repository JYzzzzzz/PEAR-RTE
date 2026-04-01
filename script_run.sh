#!/bin/bash

dataset_dir="dataset/CMIM23-NOM1-RA"
GPU_ID="0"

loss_list=("lsr" )
lsr_eps_list=(0.2 )
output_dir_list=(
"outputs/PEAR241018"
)

# Multiple experiments conducted in series
echo "${output_dir_list[*]}"
echo ""
for i in "${!output_dir_list[@]}"
do
  echo "-- shell: start python3 03_run_ner_crf.py $i-th"
  echo "-- shell: lsr_eps = ${lsr_eps_list[$((i))]}"
  echo "-- shell: output_dir = ${output_dir_list[$((i))]}"

  # train and get dev_dataset outputs
  python3 run.py \
      --loss_type ${loss_list[$((i))]} \
      --lsr_eps ${lsr_eps_list[$((i))]} \
      --output_dir ${output_dir_list[$((i))]} \
      --save_epochs 1.0 --eval_epochs 1.0 \
      --data_dir ${dataset_dir} \
      --num_train_epochs 80.0 \
      --gpu_id "${GPU_ID}" \
      --do_train \

  # score for dev_dataset predictions
  python3 get_score.py \
      --OUTPUT_DIR ${output_dir_list[$((i))]} \
      --DATASET_DIR ${dataset_dir}

  # only predict test_dataset
  python3 run.py \
      --loss_type ${loss_list[$((i))]} \
      --output_dir "${output_dir_list[$((i))]}" \
      --data_dir ${dataset_dir} \
      --gpu_id "${GPU_ID}" \
      --do_predict_only
  # score for test_dataset prediction. Exact Match & GOUGE Match
  python3 get_score.py \
      --OUTPUT_DIR "${output_dir_list[$((i))]}/best" \
      --DATASET_DIR ${dataset_dir}
  python3 get_score.py \
      --OUTPUT_DIR "${output_dir_list[$((i))]}/best" \
      --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \
      --DATASET_DIR ${dataset_dir}


done

