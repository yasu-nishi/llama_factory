#!/bin/bash

python ../../../src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset oasst2_33k_ja,oasst1_21k_ja,ichikara,oasst2_33k_en,dolly_ja,answercarefully \
    --dataset_dir ../../../data \
    --template llama3_ja \
    --finetuning_type full \
    --use_galore \
    --galore_layerwise \
    --galore_target mlp,self_attn \
    --galore_rank 128 \
    --galore_scale 2.0 \
    --output_dir /content/drive/MyDrive/GENIAC/llama3_8B_galore \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1536 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --max_samples 5000 \
    --plot_loss \
    --pure_bf16 \
    --flash_attn fa2 \
    --gradient_checkpointing \
    --export_hub_model_id Yasusan/llama3-ja-sft-galore   ## the Hugging Face hub ID to upload model

