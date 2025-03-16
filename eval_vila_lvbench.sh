export HF_ENDPOINT="https://hf-mirror.com"
# export no_proxy=hf-mirror.com
export HF_HOME=/home/gaohuan03/ouyangkun/.cache/huggingface
# export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=3
# export HF_DATASETS_OFFLINE=1
python3 -m accelerate.commands.launch \
    --num_processes 8 \
    -m lmms_eval \
    --model vila \
    --model_args pretrained="Efficient-Large-Model/VILA1.5-40b",attn_implementation="flash_attention_2",max_frames_num=2,device_map="auto" \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vila \
    --output_path ./logs/ 
