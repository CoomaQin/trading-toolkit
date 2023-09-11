python -u train.py \
    --dataset_path "./data/tesla_22to23_500" \
    --lora_rank 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 1e-6 \
    --remove_unused_columns false \
    --logging_steps 10 \
    --eval_steps 100 \
    --load_best_model_at_end true \
    --evaluation_strategy "steps" \
    --output_dir "./teslagpt"