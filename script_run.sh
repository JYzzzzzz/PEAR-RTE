#!/bin/bash

#source ~/.bashrc  ### 初始化环境变量
#source  /opt/app/anaconda3/bin/activate python3_10
#cd /home/u2021110308/jyz_projects/ner_code_231117

dataset_dir="dataset/CMIM23-NOM1-RA"

lstm_layer_list=(2 )
lstm_hidden_list=(768 )
lstm_bi_list=(True )
bilstm_len_list=("none" )
indep_bw_lstm_h_list=(0 )

loss_list=("lsr" )
lsr_eps_list=(0.2 )

output_dir_list=(
"outputs/241006"
)

# Multiple experiments conducted in series
echo "${output_dir_list[*]}"
echo ""
for i in "${!output_dir_list[@]}"
do
  echo "-- shell: start python3 03_run_ner_crf.py $i-th"
  echo "-- shell: lstm_num_layers = ${lstm_layer_list[$((i))]}"
  echo "-- shell: lstm_hidden_size = ${lstm_hidden_list[$((i))]}"
  echo "-- shell: lstm_bidirectional = ${lstm_bi_list[$((i))]}"
  echo "-- shell: lsr_eps = ${lsr_eps_list[$((i))]}"
  echo "-- shell: output_dir = ${output_dir_list[$((i))]}"

  python3 run.py \
      --lstm_num_layers ${lstm_layer_list[$((i))]} \
      --lstm_hidden_size ${lstm_hidden_list[$((i))]} \
      --lstm_bidirectional ${lstm_bi_list[$((i))]} \
      --bilstm_len ${bilstm_len_list[$((i))]} \
      --indep_bw_lstm_h ${indep_bw_lstm_h_list[$((i))]} \
      --loss_type ${loss_list[$((i))]} \
      --lsr_eps ${lsr_eps_list[$((i))]} \
      --output_dir ${output_dir_list[$((i))]} \
      --save_epochs 1.0 --eval_epochs 1.0 \
      --data_dir ${dataset_dir} \
      --gpu_id "1" \
  #    --max_origin_sent_token_len__eval_ratio 1.5 \
  #    --num_train_epochs 50.0 \
#      --lsr_eps ${lsr_eps_list[$((i))]} \
#      --ignore_mid_epoch_eval True \

  python3 get_score.py \
      --OUTPUT_DIR ${output_dir_list[$((i))]} \
      --DATASET_DIR ${dataset_dir}

  python3 get_score.py \
      --OUTPUT_DIR ${output_dir_list[$((i))]} \
      --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \
      --DATASET_DIR ${dataset_dir}

#  # 用于删除模型
#  echo "delete model in ${output_dir_list[$((i))]}"
#  find ${output_dir_list[$((i))]} -type f -name 'pytorch_model.bin' | xargs rm -f

done
