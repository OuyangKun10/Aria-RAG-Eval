Before evaluation, you need modify the HF_HOME in the .sh file, ensure that you download the dataset in the correct path.

```markdown
# Aria-RAG-Eval

Aria-RAG evaluation code

For example, to evaluate Qwen2VL on MLVU, you can run:

```bash
bash eval_qwen2vl_mlvu.sh
```

**Please first check the path of rag file**

To evaluate Qwen2VL-w-Aria-RAG:

```bash
bash eval_qwen2vl_rag_mlvu.sh
```

**Testing 72B model**

You need load it with multi-gpus, and do the following step:
```
# -m accelerate.commands.launch \
#    --num_processes 8 \
#    --gpu_ids 0,1,2,3,4,5,6,7 \
```

## To test vila

1.

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
git clone git@github.com:NVlabs/VILA.git
cd VILA
python -m pip install -e .
```

2. not using lmms-eval

```bash
git clone https://github.com/NVlabs/VILA.git
./environment_setup.sh vila
conda activate vila
python -W ignore server.py \
    --port 8000 \
    --model-path Efficient-Large-Model/VILA1.5-40b \
    --conv-mode auto
```

```python
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
```
```
