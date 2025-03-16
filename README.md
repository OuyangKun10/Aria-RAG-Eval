# Aria-RAG-Eval
Aria-RAG evaluation code

For example, to evaluate Qwen2VL on MLVU, you can run:

bash eval_qwen2vl_mlvu.sh

To evaluate Qwen2VL-w-Aria-RAG:

bash eval_qwen2vl_rag_mlvu.sh

To test vila
1)

pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales

git clone git@github.com:NVlabs/VILA.git

cd VILA

python -m pip install -e .

if meet any bugs when evaluating vila directly using lmms-eval, you may find useful solutions on https://github.com/EvolvingLMMs-Lab/lmms-eval.

2) not using lmms-eval

