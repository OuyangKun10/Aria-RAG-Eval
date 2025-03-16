export HF_ENDPOINT="https://hf-mirror.com"
# export no_proxy=hf-mirror.com
export DECORD_EOF_RETRY_MAX=20480s
# export HF_DATASETS_OFFLINE=1
export HF_HOME=/home/gaohuan03/ouyangkun/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,2,3
python3 -m accelerate.commands.launch \
    --num_processes 8 \
    -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained="/pfs/Models/Qwen2-VL-72B-Instruct",use_flash_attention_2=True,max_num_frames=64,device_map="auto" \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2vl \
    --output_path ./logs/
