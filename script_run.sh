#!/bin/bash

#ssh gpu9
#source ~/.bashrc  ### 初始化环境变量
#source  /opt/app/anaconda3/bin/activate python3_10
#cd /home/u2021110308/jyz_projects/ner_code_2311/ner_code_231117/

dataset_dir="dataset/CMIM23-NOM1-RA"
GPU_ID="0"

#lstm_layer_list=(2 )
#lstm_hidden_list=(768 )
#lstm_bi_list=(False )
#bilstm_len_list=("none" )
#indep_bw_lstm_h_list=(0 )
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
#  echo "-- shell: lstm_num_layers = ${lstm_layer_list[$((i))]}"
#  echo "-- shell: lstm_hidden_size = ${lstm_hidden_list[$((i))]}"
#  echo "-- shell: lstm_bidirectional = ${lstm_bi_list[$((i))]}"
  echo "-- shell: lsr_eps = ${lsr_eps_list[$((i))]}"
  echo "-- shell: output_dir = ${output_dir_list[$((i))]}"

  # -------------------- train and get dev_dataset outputs
  python3 run.py \
      --loss_type ${loss_list[$((i))]} \
      --lsr_eps ${lsr_eps_list[$((i))]} \
      --output_dir ${output_dir_list[$((i))]} \
      --save_epochs 1.0 --eval_epochs 1.0 \
      --data_dir ${dataset_dir} \
      --num_train_epochs 80.0 \
      --gpu_id "${GPU_ID}" \
      --do_train \
#      --ignore_mid_epoch_eval True \
#      --lstm_num_layers ${lstm_layer_list[$((i))]} \
#      --lstm_hidden_size ${lstm_hidden_list[$((i))]} \
#      --lstm_bidirectional ${lstm_bi_list[$((i))]} \
#      --bilstm_len ${bilstm_len_list[$((i))]} \
#      --indep_bw_lstm_h ${indep_bw_lstm_h_list[$((i))]} \

  # -------------------- score for dev_dataset predictions
  python3 get_score.py \
      --OUTPUT_DIR ${output_dir_list[$((i))]} \
      --DATASET_DIR ${dataset_dir}
#  python3 get_score.py \
#      --OUTPUT_DIR ${output_dir_list[$((i))]} \
#      --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \
#      --DATASET_DIR ${dataset_dir}

  # -------------------- only predict test_dataset
  python3 run.py \
      --loss_type ${loss_list[$((i))]} \
      --output_dir "${output_dir_list[$((i))]}" \
      --data_dir ${dataset_dir} \
      --gpu_id "${GPU_ID}" \
      --do_predict_only
  # -------------------- score for test_dataset prediction. Exact Match & GOUGE Match
  python3 get_score.py \
      --OUTPUT_DIR "${output_dir_list[$((i))]}/best" \
      --DATASET_DIR ${dataset_dir}
  python3 get_score.py \
      --OUTPUT_DIR "${output_dir_list[$((i))]}/best" \
      --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \
      --DATASET_DIR ${dataset_dir}

#  # -------------------- delete all model files in output_dir
#  echo "delete model in ${output_dir_list[$((i))]}"
#  find ${output_dir_list[$((i))]} -type f -name 'pytorch_model.bin' | xargs rm -f

done

