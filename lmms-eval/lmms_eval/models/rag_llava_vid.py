import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.language_model.llava_qwen import LlavaQwenConfig

# eval_logger = logging.getLogger("lmms-eval")
# import sys;sys.path.append("llava-video")
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

# from lmms_eval.models.model_utils.rag_utils import load_rag_file          # FIXME
import os
import json
import random

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
    with open(json_file, 'r') as f:
        for line in f:
            # try:
                # print("video_value",video_value)
                # print("question_value",question_value)
                element = json.loads(line)  # 解析每一行
                # print(element)
                if video_value in element.get("video_id") and element.get("question") in question_value:
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
                # print(item.keys())
                # print(item['id'])
                # try:
                #     qid = os.path.basename(item['id'])
                # except:
                #     qid = item['id']
                qid = item['id']
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



# try:
#     from llavavid.model.builder import load_pretrained_model
#     from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
#     from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
#     from llavavid.conversation import conv_templates, SeparatorStyle
#     from llavavid.mm_utils import tokenizer_image_token_qwen_merge, preprocess_qwen, preprocess_llama3
# except ImportError:
#     import llava
#     import pdb;pdb.set_trace()
#     if "llava-video-old" in llava.__file__:
#         from llava.model.language_model.llava_llama import LlavaConfig
#         from llava.model.language_model.llava_qwen import LlavaQwenConfig
#         from llava.model.builder import load_pretrained_model
#         from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
#         from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
#         from llava.conversation import conv_templates, SeparatorStyle

#         AutoConfig.register("llava_llama", LlavaConfig)
#         AutoConfig.register("llava_qwen", LlavaQwenConfig)
#     else:
#         eval_logger.debug("LLaVA-Video is not installed. Please install LLaVA-Video to use this model.")

# from llavavid.model.language_model.llava_qwen import LlavaQwenConfig
# from llavavid.model.language_model.llava_llama import LlavaConfig

# AutoConfig.register("llava_qwen", LlavaQwenConfig)
# AutoConfig.register("llava_llama", LlavaConfig)


AutoConfig.register("llava_llama", LlavaConfig)
AutoConfig.register("llava_qwen", LlavaQwenConfig)


@register_model("rag_llava_vid")             # FIXME
class RAG_LlavaVid(lmms):             # FIXME
    """
    LlavaVid Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        max_frames_num: int = 20,
        video_fps: int = 1,
        mm_resampler_type: str = "spatial_pool",
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_out_channels: int = 1024,
        mm_spatial_pool_mode: str = "average",
        mm_resampler_location: str = "before",
        mm_newline_position: str = "grid",
        overwrite: bool = True,
        video_decode_backend: str = "decord",
        delay_load: bool = False,
        tie_weights: bool = True,
        force_sample: bool = False,
        add_time_instruction: bool = False,
        add_faster_video: bool = False,
        faster_token_stride: int = 10,
        rag_file=None,             # FIXME
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and (device_map == "auto" or device_map == "balanced_low_0"):
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.video_decode_backend = video_decode_backend
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self.overwrite = overwrite
        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.max_frames_num = int(max_frames_num)
        self.fps = int(video_fps)
        self.mm_resampler_location = mm_resampler_location
        self.delay_load = delay_load
        self.force_sample = force_sample
        self.add_time_instruction = add_time_instruction
        print("force sample:", self.force_sample)
        # self.add_faster_video = add_faster_video
        # self.faster_token_stride = faster_token_stride
        self.torch_dtype = torch_dtype
        if self.overwrite == True:
            overwrite_config = {}
            # overwrite_config["mm_resampler_type"] = self.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
            overwrite_config["mm_pooling_position"] = self.mm_resampler_location
            overwrite_config["mm_newline_position"] = mm_newline_position
            overwrite_config["delay_load"] = self.delay_load
            # overwrite_config["attn_implementation"] = attn_implementation

            if "vicuna" in self.pretrained.lower() or "yi" in self.pretrained.lower():
                cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    eval_logger.info(f"Scaling factor: {scaling_factor}")
                    # print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained, None, self.model_name, device_map=self.device_map, torch_dtype=self.torch_dtype, overwrite_config=overwrite_config, attn_implementation=attn_implementation
            )
        else:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map, torch_dtype=self.torch_dtype, attn_implementation=attn_implementation)

        self._config = self._model.config

        # import pdb;pdb.set_trace()

        if self._tokenizer.pad_token_id is None:
            if "qwen" in self._tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self._tokenizer.pad_token_id = 151643

        self.model.eval()
        if tie_weights:
            self.model.tie_weights()
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
        

        # FIXME
        self.rag = load_rag_file(rag_file)
        self.rag_file = rag_file        
                        


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

    def load_image(self, image_path):
        raise NotImplementedError("TODO: Implement image loading for LLaVAVid")         # FIXME

        frame_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        num_frames_to_sample = 10

        total_frames = len(frame_files)

        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # Read and store the sampled frames
        video = []
        for idx in sampled_indices:
            frame_path = frame_files[idx]
            try:
                with Image.open(frame_path) as img:
                    # Convert the PIL image to a numpy array if needed
                    # frame = np.array(img.convert('RGB'))
                    frame = img.convert("RGB")
                    video.append(frame)
            except IOError:
                print(f"Failed to read frame at path: {frame_path}")
        return video
    
    # FIXME
    def load_video(self, video_path,qid, max_frames_num, fps, force_sample=False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        
        print(video_path)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, total_frame_num, fps)]
        frame_time = [i / fps for i in frame_idx]
        print("load video")
        # print("qid",qid)
        # print("rag images",self.rag[qid])
        if len(frame_idx) > max_frames_num and qid in self.rag:
            rag_frame_idxs = [int(vr.get_avg_fps() * _) for _ in self.rag[qid] if int(vr.get_avg_fps() * _) < total_frame_num]
            frame_idx = sorted(rag_frame_idxs[:max_frames_num])
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            print(True)
        elif len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        print(frame_idx)
        if len(frame_idx)==0:
            print("0 frame")
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        elif len(frame_idx)<max_frames_num:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx_uni = uniform_sampled_frames.tolist()
            import random
            frame_idx_uni=random.sample(frame_idx_uni, max_frames_num-len(frame_idx))
            frame_idx=frame_idx+frame_idx_uni
            frame_time_uni = [i / vr.get_avg_fps() for i in frame_idx_uni]
            frame_time=frame_time+frame_time_uni
        
        frame_idx=sorted(frame_idx)
        frame_time=sorted(frame_time)
        print(frame_idx)
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()

        return spare_frames, frame_time, video_time

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
                print("phase1 ")
                qid=find_qid(self.rag_file ,visuals,contexts)
                video, frame_time, video_time = self.load_video(visuals[0],qid,self.max_frames_num, self.fps, force_sample=self.force_sample)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda()
                if self.torch_dtype == "bfloat16":
                    video = video.bfloat16()
                else:
                    video = video.half()
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
            # if self.task_dict[task][split][doc_id]["duration"] != "short":
            # # if doc_id != 112:
            #     # import pdb;pdb.set_trace()
            #     res.append("A")
            #     pbar.update(1)
            #     continue
            # encode, pad, and truncate contexts for this batch
            # import pdb;pdb.set_trace()
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            # visuals = [visuals]
            # visuals = self.flatten(visuals)
      
            videos = []
            try:
                # for visual in visuals:
                if len(visuals) == 1:
                    if self.video_decode_backend == "decord":
                        qid=find_qid(self.rag_file ,visuals,contexts)
                        video, frame_time, video_time = self.load_video(visuals[0],qid,self.max_frames_num, self.fps, force_sample=self.force_sample)
                    elif self.video_decode_backend == "pyav":
                        print("error, pyav not decord")
                        raise NotImplementedError("TODO: Implement pyav video loading for LLaVAVid")     # FIXME
                        video, frame_time, video_time = read_video_pyav(visuals[0], self.max_frames_num, self.fps, force_sample=self.force_sample)
                    elif self.video_decode_backend == "image":
                        print("error, image not decord")
                        video = self.load_image(visuals[0])
                else:
                    print("error, visuals not 1")
                    raise NotImplementedError("TODO: len(visuals) != 1")     # FIXME
                    if task == "seedbench":
                        video = visuals
                        frame_time = "1.00s"
                        video_time = 1
                    elif "mvbench" in task:
                        # video = visuals
                        # Reference: https://github.com/jayleicn/TVQA/blob/dfb0e5fe4582efca574dfddfeafd1008db3b33ef/data/README.md?plain=1#L50C34-L50C60
                        fps = 3
                        video_time = len(visuals) / fps
                        sampled_indices = np.linspace(0, len(visuals) - 1, self.max_frames_num, dtype=int)
                        frame_idx = sampled_indices.tolist()
                        frame_time = [i / fps for i in frame_idx]
                        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
                        video = [visuals[i] for i in frame_idx]

                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda()
                if self.torch_dtype == "bfloat16":
                    video = video.bfloat16()
                else:
                    video = video.half()
                videos.append(video)
            except Exception as e:
                # import pdb;pdb.set_trace()
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {visuals} can not load, check the source")
                video_path = "\n".join(visuals)
                res.append(f"Video {video_path} can not load, check the source")
                pbar.update(1)
                continue

            qs = contexts
            # import pdb;pdb.set_trace()
            if self.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                qs = f"{time_instruciton}\n{qs}"
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            #fixme
            # prompt=prompt.replace("<image>", "")
            # print(prompt)
            # exit()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if "llama_3" in self.conv_template:
                pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]

            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            cur_prompt = qs

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # import pdb;pdb.set_trace()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=videos,
                    attention_mask=attention_masks,
                    modalities="video",
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
            eval_logger.debug(f"Question: {cur_prompt}")
            eval_logger.debug(f"Answer: {outputs}")
            # import pdb;pdb.set_trace()
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
