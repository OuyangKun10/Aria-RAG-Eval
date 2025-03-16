export HF_ENDPOINT="https://hf-mirror.com"
export no_proxy=hf-mirror.com
# export HF_DATASETS_OFFLINE=1
export HF_HOME=/home/ouyangkun/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
python3 -m accelerate.commands.launch \
    --num_processes 8 \
    -m lmms_eval \
    --model rag_qwen2_vl \
    --model_args pretrained="/home/lishicheng/ckpts/Qwen/Qwen2-VL-7B-Instruct/",use_flash_attention_2=True,max_num_frames=32,device_map="auto",rag_file="/home/ouyangkun/lmmeval/rag_file/longvideobench_rag_val_mm_rag_0114_span_with_scores_index_148k_etbench_taskprompt_rag_44k_from_0916_256f_fps1.0_subtitle_parsed.json" \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2vl \
    --output_path ./logs/
