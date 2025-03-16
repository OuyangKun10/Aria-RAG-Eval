import argparse
import json
import logging
import math
import os
import signal
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("lmms-eval")
# import sys;sys.path.append("llava-video")
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.data.dataset import LazySupervisedDataset
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
except ImportError as e:
    eval_logger.debug(f"VILA is not installed. Please install VILA to use this model. Error: {e}")
    
def find_qid(json_file,video_value,question_value):
    """
    按行读取 JSON 文件，查找 video=x 且 question 包含在 question_value 中的唯一元素，并返回对应的 qid。
    假设每行是一个独立的 JSON 对象。

    Args:
        json_file_path (str): JSON 文件的路径。
        video_value (str): 要查找的 video 值。
        question_value (str): 要查找的 question 值（question 需要包含在其中）。

    Returns:
        str: 如果找到唯一的元素，则返回其 qid。 如果找不到或找到多个匹配项，则返回 None。
    """

    matching_elements = []
    video_value=video_value[0]
    video_value=os.path.basename(video_value)
    video_value=video_value.replace(".mp4","")
    # try:
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            # try:

                # print("video_value",video_value)
                # print("question_value",question_value)
                element = json.loads(line)  # 解析每一行
                # print(element)
                if element.get("video_id") == os.path.basename(video_value) and element.get("question") in question_value:
                    matching_elements.append(element)
                # except json.JSONDecodeError:
                #     print(f"警告：跳过无效的 JSON 行：{line.strip()}")
                #     continue  # 跳过此行，继续下一行
                # except Exception as e:
                #     print(f"读取行时发生未知错误：{e}")
                #     return None

    # except FileNotFoundError:
    #     print(f"错误：文件 '{json_file_path}' 未找到。")
    #     return None
    # except Exception as e:
    #     print(f"读取文件时发生未知错误: {e}")
    #     return None


    if len(matching_elements) == 0:
        print(f"未找到 video='{video_value}' 且 question 包含在 '{question_value}' 中的元素。")
        return None
    elif len(matching_elements) > 1:
        print(f"找到多个 video='{video_value}' 且 question 包含在 '{question_value}' 中的元素。")
        return None
    else:
        print("成功找到")
        return matching_elements[0].get("id")

def load_rag_file(file):
    rag = {}
    if file is not None:
        with open(file, "r") as f:
            for line in f:
                item = json.loads(line)
                qid = os.path.basename(item['id'])
                rag[qid] = []
                sim2fid = {}
                for img_path, img_score in zip(item['rag_images'], item['rag_sim']):
                    frame_sec = int(os.path.splitext(os.path.basename(img_path))[0]) / item['fps']
                    sim2fid[img_score] = sim2fid.get(img_score, []) + [frame_sec]
                for img_score, frame_secs in sim2fid.items():
                    random.shuffle(frame_secs)
                sorted_scores = sorted(sim2fid.keys(), reverse=True)
                for score in sorted_scores:
                    rag[qid] += sim2fid[score]
    print("===", len(rag))
    return rag

@register_model("rag_vila")
class RAG_VILA(lmms):
    """
    VILA Model
    """

    def __init__(
        self,
        pretrained: str = "Efficient-Large-Model/VILA1.5-40b",
        max_frames_num: Optional[int] = 100,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="hermes-2",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        video_decode_backend="decord",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.max_frames_num = max_frames_num
        # self._config = AutoConfig.from_pretrained(self.pretrained)

        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, self.model_name, device_map=self.device_map, attn_implementation=attn_implementation)

        self.model.image_processor = self._image_processor

        self._config = self._model.config

        if self._tokenizer.pad_token_id is None:
            if "qwen" in self._tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self._tokenizer.pad_token_id = 151643

        self.video_decode_backend = video_decode_backend
        self.model.eval()
        # self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.rag = load_rag_file(rag_file)
    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_video(self, video_path, max_frames_num,qid):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = np.linspace(0, total_frame_num - 2, max_frames_num, dtype=int)
            if len(frame_idx) > max_frames_num and qid in self.rag and len(self.rag[qid]) > 0:
                rag_frame_idxs = [int(vr.get_avg_fps() * _) for _ in self.rag[qid] if int(vr.get_avg_fps() * _) < total_frame_num]
                frame_idx = sorted(rag_frame_idxs[:max_frames_num])
                spare_frames = vr.get_batch(frame_idx).asnumpy()
                print(True)
            elif len(frame_idx) > max_frames_num:
                sample_fps = max_frames_num
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
                spare_frames = vr.get_batch(uniform_sampled_frames).asnumpy()
            else:
                spare_frames = vr.get_batch(frame_idx).asnumpy()
            print(frame_idx)
            return [Image.fromarray(img) for img in spare_frames]
        except Exception as e:
            eval_logger.error(f"Failed to load video {video_path} with error: {e}")
            return [Image.new("RGB", (448, 448), (0, 0, 0))] * max_frames_num

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video = self.load_video(visual, self.max_frames_num)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                videos.append(video)

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=videos, modalities="video")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            num_video_frames = self.model.config.num_video_frames
            videos = []
            if self.max_frames_num == 0:
                images = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames
                video = process_images(images, self.model.image_processor, self.model.config).half().cuda()
                videos.append(video)
            else:
                for visual in visuals:
                    if self.video_decode_backend == "decord":
                        images = self.load_video(visual, num_video_frames)
                    elif self.video_decode_backend == "pyav":
                        images = read_video_pyav(visual, num_frm=num_video_frames)
                    video = process_images(images, self.model.image_processor, self.model.config).half().cuda()
                    videos.append(video)

            qs = f"<video>\n {contexts}"
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

            # This is much safer for llama3, as we now have some object type in it
            # if "llama_3" in self.conv_template:
            #     conv = copy.deepcopy(conv_templates[self.conv_template])
            # else:
            #     conv = conv_templates[self.conv_template].copy()
            conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            # if "llama_3" in self.conv_template:
            #     pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            # input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            # pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            # input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            # attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]

            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            cur_prompt = contexts

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.2
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    images=videos,
                    attention_mask=attention_masks,
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
                # output_ids_2 = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=False, max_new_tokens=50,stopping_criteria=[stopping_criteria])
                # output_ids = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=50,use_cache=True)

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print("Question: ", cur_prompt)
            print("Answer: ", outputs)
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
