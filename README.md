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


2) not using lmms-eval

{git clone https://github.com/NVlabs/VILA.git}

{./environment_setup.sh vila}

{conda activate vila}

{python -W ignore server.py \
    --port 8000 \
    --model-path Efficient-Large-Model/VILA1.5-40b \
    --conv-mode auto}

{
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="fake-key",
)
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": {the input question}},
                {
                    "type": "video",
                    "video": { [the frame list, opt: uniform sampling; RAG-frames]
                    },
                },
            ],
        }
    ],
    model="VILA1.5-40b",
)
}
