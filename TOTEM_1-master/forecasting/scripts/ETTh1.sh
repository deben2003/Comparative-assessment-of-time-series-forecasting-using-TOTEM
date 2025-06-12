#!/bin/bash

# Define base paths
ROOT_PATH="forecasting/data/ETTh1"
DATA_PATH="ETTh1.csv"
SAVE_PATH="forecasting/data/ETTh1/Tin96_Tout96"
MODEL_PATH="forecasting/save/VQ-VAE/ETTh1/Tin96_Tout96"

# # Colab-native version of ETTh1.sh script for TOTEM Forecasting

import os

# ✅ Set base paths
base_path = "/content/drive/MyDrive/Colab Notebooks/TOTEM_1-master/TOTEM_1-master"
forecasting_path = os.path.join(base_path, "forecasting")

# Step 1: Train VQ-VAE model
!python3 "{forecasting_path}/train_vqvae.py" \
  --root_path forecasting/data/ETTh1 \
  --data_path ETTh1.csv \
  --save_path forecasting/save/VQ-VAE/ETTh1/Tin96_Tout96 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 8 \
  --d_model 64 \
  --num_workers 4 \
  --itr 1

# Step 2: Save transformed data (REVIN processed)
!python3 "{forecasting_path}/save_revin_data.py" \
  --root_path forecasting/data/ETTh1 \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --model VQ \
  --trained_vqvae_model_path forecasting/save/VQ-VAE/ETTh1/Tin96_Tout96 \
  --save_path forecasting/data/ETTh1/Tin96_Tout96 \
  --batch_size 8 \
  --num_workers 4

# Step 3: Extract forecasting data using VQ-VAE codebook
!python3 "{forecasting_path}/extract_forecasting_data.py" \
  --root_path forecasting/data/ETTh1 \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --model VQ \
  --trained_vqvae_model_path forecasting/save/VQ-VAE/ETTh1/Tin96_Tout96 \
  --save_path forecasting/data/ETTh1/Tin96_Tout96 \
  --batch_size 8 \
  --num_workers 4

# Step 4: Train the forecaster (e.g., Transformer-based)
!python3 "{forecasting_path}/train_forecaster.py" \
  --root_path forecasting/data/ETTh1/Tin96_Tout96 \
  --data_path codebook.npy \
  --features S \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 512 \
  --d_model 256 \
  --e_layers 2 \
  --n_heads 4 \
  --batch_size 8 \
  --itr 1 \
  --run_train \
  --run_test

