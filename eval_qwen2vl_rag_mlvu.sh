export HF_ENDPOINT="https://hf-mirror.com"
# export no_proxy=hf-mirror.com
export HF_HOME=/home/gaohuan03/ouyangkun/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# export HF_DATASETS_OFFLINE=1
python3  -m accelerate.commands.launch \
    --num_processes 8 \
    -m lmms_eval \
    --model rag_qwen2_vl \
    --model_args pretrained="/pfs/Models/Qwen2-VL-7B-Instruct",use_flash_attention_2=True,max_num_frames=32,device_map="auto",rag_file="/home/gaohuan03/ouyangkun/code/lmmeval/rag_file/mlvu_rag_val_mm_rag_0114_span_with_scores_index_148k_etbench_taskprompt_rag_44k_from_0916_256f_fps1.0_parsed.json" \
    --tasks mlvu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2vl \
    --output_path ./logs/
