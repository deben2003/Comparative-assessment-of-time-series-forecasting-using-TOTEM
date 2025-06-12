gpu=1
#!/bin/bash

# General config
seq_len=96
pred_len=96
random_seed=2021
gpu=0

# Absolute path for Colab with spaces handled
root_path_name="/content/drive/MyDrive/Colab Notebooks/TOTEM_1-master/data/Forecasting"
data_path_name="ETTh1.csv"
data_name="ETTh1"

# Step 1: Save revin data
python3 -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "$root_path_name" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7 \
  --gpu $gpu \
  --save_path "forecasting/data/ETTh1"

# Step 2: Train VQ-VAE
python3 forecasting/train_vqvae.py \
  --config_path forecasting/scripts/ETTh1.json \
  --model_init_num_gpus $gpu \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_ETTh1 \
  --save_path "forecasting/saved_models/ETTh1/" \
  --base_path "forecasting/data" \
  --batchsize 4096

# Step 3: Extract forecasting data
for pred_len in 96 192 336 720
do
python3 -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "$root_path_name" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7 \
  --gpu $gpu \
  --save_path "forecasting/data/ETTh1/Tin${seq_len}_Tout${pred_len}/" \
  --trained_vqvae_model_path "path/to/your/trained/vqvae_model.pth" \
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

# Step 4: Train the forecaster
Tin=96
datatype="ETTh1"
gpu=0
for seed in 2021 1 13
do
  for Tout in 96 192 336 720
  do
    python3 forecasting/train_forecaster.py \
      --data-type $datatype \
      --Tin $Tin \
      --Tout $Tout \
      --cuda-id $gpu \
      --seed $seed \
      --data_path "forecasting/data/${datatype}/Tin${Tin}_Tout${Tout}" \
      --codebook_size 256 \
      --checkpoint \
      --checkpoint_path "forecasting/saved_models/${datatype}/forecaster_checkpoints/${datatype}_Tin${Tin}_Tout${Tout}_seed${seed}" \
      --file_save_path "forecasting/results/${datatype}/"
  done
done
