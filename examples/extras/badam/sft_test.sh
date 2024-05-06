#!/bin/bash

python ../../../src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset oasst2_33k_ja,oasst1_21k_ja,ichikara,oasst2_33k_en,dolly_ja \
    --dataset_dir ../../../data \
    --template llama3_ja \
    --finetuning_type full \
    --use_badam \
    --badam_switch_mode descending \
    --badam_switch_interval 50 \
    --badam_verbose 2 \
    --output_dir ../../../sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --val_size 0.1 \
    --plot_loss \
    --pure_bf16 \
    --export_hub_model_id Yasusan/llama3-ja-sft-badam    ## the Hugging Face hub ID to upload model

#max_samples #For debugging purposes, truncate the number of examples for each dataset
#optim paged_adamw_8bit \
#--template llama2_ja \