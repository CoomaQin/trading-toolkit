PRE_SEQ_LEN=500
CHECKPOINT=tesla22to23-chatglm-6b-pt-500-2e-2
STEP=1000

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --validation_file data/tesla_22to23_500_val.json \
    --test_file data/tesla_22to23_500_val.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length $PRE_SEQ_LEN \
    --max_target_length $PRE_SEQ_LEN \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4