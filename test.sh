PRE_SEQ_LEN=500

CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint output/tesla22to23-chatglm-6b-pt-500-2e-2/checkpoint-1000 \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 