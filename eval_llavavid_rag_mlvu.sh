export HF_ENDPOINT="https://hf-mirror.com"
export DECORD_EOF_RETRY_MAX=20480s
# export no_proxy=hf-mirror.com
export HF_HOME=/home/gaohuan03/ouyangkun/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m accelerate.commands.launch \
    --num_processes 8 \
    -m lmms_eval \
    --model rag_llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-Video-72B-Qwen2,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=64,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after,device_map="auto",rag_file="/home/gaohuan03/ouyangkun/code/lmmeval/rag_file/mlvu_rag_val_mm_rag_0114_span_with_scores_index_148k_etbench_taskprompt_rag_44k_from_0916_256f_fps1.0_parsed.json" \
    --tasks mlvu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llavavid_rag_mlvu_64_rag \
    --output_path ./logs/ 
