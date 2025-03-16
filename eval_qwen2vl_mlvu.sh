export HF_ENDPOINT="https://hf-mirror.com"
# export no_proxy=hf-mirror.com
export HF_HOME=/home/gaohuan03/ouyangkun/.cache/huggingface
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# export HF_DATASETS_OFFLINE=1
python3 -m accelerate.commands.launch \
    --num_processes 8 \
    -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained="/pfs/Models/Qwen2-VL-7B-Instruct",use_flash_attention_2=True,max_num_frames=32,max_pixels=3211264,device_map="auto" \
    --tasks mlvu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2vl \
    --output_path ./logs/ 
    