# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
import numpy as np
from typing import Dict, Optional, Sequence, List
from glob import glob
from tqdm import tqdm

import torch

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
# from openscen
from llava.dataset import augmentation as t
from llava.dataset.voxelizer import Voxelizer

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    hidden_size: Optional[int] = field(default=768)
    qformer_num_query_token: Optional[int] = field(default=32)
    qformer_vision_width: Optional[int] = field(default=768)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    pc_feature_folder: Optional[str] = field(default=None)
    image_feature_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    def maybe_zero_3(param, ignore_status=False, name=None):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if not ignore_status:
                    logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
        return param
    def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
        to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
        to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
        return to_return

    # if getattr(trainer.args, "tune_mm_mlp_adapter", False):
    # Only save Adapter
    # keys_to_match = ['mm_projector']
    keys_to_match = ['mm_projector_3d', 'Qformer_pc', 'query_tokens_pc', 'mm_projector_2d', 'Qformer_2d', 'query_tokens_2d', 'position_embedding_projector', 'pointnet2', 'semanticmap_tokenizer']
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(['embed_tokens', 'embed_in'])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    if hasattr(trainer.model.model, 'pointnet2'):
        pointnet2_state_dict = trainer.model.model.pointnet2.state_dict()
        pointnet2_state_dict = {'model.pointnet2.' + key: pointnet2_state_dict[key] for key in pointnet2_state_dict if 'running' in key}
        weight_to_save.update(pointnet2_state_dict)
    trainer.model.config.save_pretrained(output_dir)  # save file 'config.json'

    current_folder = output_dir.split('/')[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith('checkpoint-'):
            mm_adapter_folder = os.path.join(parent_folder, "mm_adapter")
            os.makedirs(mm_adapter_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_adapter_folder, f'{current_folder}.bin'))
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f'mm_adapter.bin'))
        # return
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_scanqa(sources):
    '''make conversations and id'''
    for i, source in enumerate(sources):
        source['conversations'] = [
            {'from': 'human', 'value': source['question'] + '\n<image>'},
            {'from': 'gpt', 'value': source['answers'][0]}
        ]
        source['id'] = source['scene_id']
    return sources


def preprocess_image_token(sentence: str, data_args: DataArguments) -> str:
    if DEFAULT_IMAGE_TOKEN in sentence:
        sentence = sentence.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        sentence = DEFAULT_IMAGE_TOKEN + '\n' + sentence
        sentence = sentence.strip()
        if "mmtag" in conversation_lib.default_conversation.version:
            sentence = sentence.replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
    replace_token = DEFAULT_IMAGE_TOKEN
    if data_args.mm_use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    sentence = sentence.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sentence


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # if 'ScanQA' in data_args.data_path:
    #     for source in sources:
    #         source = preprocess_image_token(source, data_args)
    # else:
    #     for source in sources:
    #         for sentence in source:
    #             sentence['value'] = preprocess_image_token(sentence['value'], data_args)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # if conversation_lib.default_conversation.version == "ScanQA":
    #     return preprocess_scanqa(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


# how to rewrite 
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    # Augmentation arguments for 3D data
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer # LLM
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.aug = False
        self.eval_all = True
        self.input_color = False
        # temporary set it to a constant value given in the openscene
        voxel_size = 0.05
        data_aug_color_trans_ratio=0.1
        data_aug_color_jitter_std=0.05
        data_aug_hue_max=0.5
        data_aug_saturation_max=0.2

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)
        # always use data augmentation
        prevoxel_transform_train = [
            t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
        self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
        input_transforms = [
            t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
            t.ChromaticAutoContrast(),
            t.ChromaticTranslation(data_aug_color_trans_ratio),
            t.ChromaticJitter(data_aug_color_jitter_std),
            t.HueSaturationTranslation(
                data_aug_hue_max, data_aug_saturation_max),
        ]
        self.input_transforms = t.Compose(input_transforms)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        if 'image' in sources[0] or 'scene_id' in sources[0]:
            # point_file = self.list_data_dict[i]['image']
            # point_folder = self.data_args.image_folder
            # # self.point_file_path = glob(f"{point_folder}/{point_file}")
            # self.point_file_path = os.path.join(point_folder, point_file)
            # # load ".pth" form dataset file
            # locs_in, feats_in, labels_in = torch.load(self.point_file_path)

            # # copy from Point3DLoader (openscene)
            # labels_in[labels_in == -100] = 255
            # labels_in = labels_in.astype(np.uint8)
            # # no color in the input point cloud, e.g nuscenes
            # if np.isscalar(feats_in) and feats_in == 0:
            #     feats_in = np.zeros_like(locs_in)
            # feats_in = (feats_in + 1.) * 127.5
            # locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
            # locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(
            #     locs, feats_in, labels_in)
            # if self.eval_all:
            #     labels = labels_in
            # if self.aug:
            #     locs, feats, labels = self.input_transforms(locs, feats, labels)
            # coords = torch.from_numpy(locs).int()
            # coords = torch.cat(
            #     (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
            # if self.input_color:
            #     feats = torch.from_numpy(feats).float() / 127.5 - 1.
            # else:
            #     feats = torch.ones(coords.shape[0], 3)
            # labels = torch.from_numpy(labels).long()
            # self.point = (coords, feats, labels)

            if 'ScanQA' in self.data_args.data_path:
                sources = preprocess_scanqa(sources)
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            # if 'ScanQA' in self.data_args.data_path:
            #     sources = preprocess_multimodal(copy.deepcopy([e["question"] for e in sources]), self.data_args)
            # else:
            #     sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]) or ('scene_id' in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # 
        # data_dict['image'] = self.point
        if 'openscene' in self.data_args.pc_feature_folder:
            pc_feature_file = self.list_data_dict[i]['id'] + '/feature.pt'
            pc_pos_file = self.list_data_dict[i]['id'] + '/pos.pt'
        elif 'TaPA' in self.data_args.pc_feature_folder:
            pc_feature_file = self.list_data_dict[i]['image']  # TaPA's semantic map '.npy' file
            pc_pos_file = None
        else:
            pc_feature_file = self.list_data_dict[i]['id'] + '.pth'
            pc_pos_file = None

        if os.path.exists(os.path.join(self.data_args.pc_feature_folder, pc_feature_file)):
            if 'TaPA' in self.data_args.pc_feature_folder:
                pc_feature = np.load(os.path.join(self.data_args.pc_feature_folder, pc_feature_file))
                pc_feature = torch.tensor(pc_feature)
            else:
                pc_feature = torch.load(os.path.join(self.data_args.pc_feature_folder, pc_feature_file))
        else:
            pc_feature = None

        if pc_pos_file and os.path.exists(os.path.join(self.data_args.pc_feature_folder, pc_pos_file)):
            pc_pos, _, _ = torch.load(os.path.join(self.data_args.pc_feature_folder, pc_pos_file))
            pc_pos = torch.tensor(pc_pos)
        else:
            pc_pos = None
        data_dict['image'] = pc_feature
        data_dict['pc_pos'] = pc_pos
        data_dict['question_embedding'] = torch.rand(768)
        indices_key = 'similarity_sampled_indices_openscene-openseg-ensemble_ViT-L/14@336px'
        best_image_key = 'best_image_openscene-openseg-ensemble_ViT-L/14@336px_ViT-bigG-14'
        if indices_key in self.list_data_dict[i]:
            data_dict['indices'] = np.array(self.list_data_dict[i][indices_key])
            data_dict['indices'] = None
        else:
            data_dict['indices'] = None
        if best_image_key in self.list_data_dict[i]:
            best_image_list = np.array(self.list_data_dict[i][best_image_key])
            best_image_path = [os.path.join(self.data_args.image_feature_folder, self.list_data_dict[i]['id'], best_image_index + '.pt') for best_image_index in best_image_list]
            best_image = torch.cat([torch.load(path) for path in best_image_path])
            data_dict['best_image'] = best_image
            data_dict['best_image'] = None
        else:
            data_dict['best_image'] = None
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, question_embedding, indices, best_image = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "question_embedding", "indices", "best_image"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        question_embedding = torch.stack(question_embedding)
        # indices = np.stack(indices)
        if indices[0] is not None:
            indices = np.stack(indices)
        else:
            indices = None
        if best_image[0] is not None:
            best_image = torch.stack(best_image)
        else:
            best_image = None
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            question_embedding=question_embedding,
            indices=indices,
            best_image=best_image,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            pc_pos = [instance['pc_pos'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            #     batch['images'] = torch.stack(images)
            # else:
            #     batch['images'] = images

            # coords, feats, labels = list(zip(*images))  # ori code

            # for i, coord in enumerate(coords):  # ori code
            #     coord[:, 0] *= i  # ori code

            # batch['images'] = torch.cat(coords), torch.cat(feats), torch.cat(labels)  # ori code
            if images[0] is not None:
                batch['images'] = torch.cat(images)  # load pc_feature directly
            else:
                batch['images'] = images  # single_modal
            if pc_pos[0] is not None:
                batch['pc_pos'] = torch.cat(pc_pos)  # load pc_pos directly
            else:
                batch['pc_pos'] = pc_pos  # single_modal

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                # ignore_mismatched_sizes=True,  # added by yinhang
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if 'openscene' in data_args.pc_feature_folder.lower() or 'openseg' in data_args.pc_feature_folder.lower():
        model_args.qformer_vision_width = 768
    elif 'pla' in data_args.pc_feature_folder.lower():
        model_args.qformer_vision_width = 16
    if '3d-llm' in data_args.pc_feature_folder.lower():
        model_args.qformer_vision_width = 1408

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.float16, device=training_args.device)

        # data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            # for p in model.get_model().mm_projector_3d.parameters():
            #     p.requires_grad = True
            # for p in model.get_model().mm_projector_2d.parameters():
            #     p.requires_grad = True
            for p in [model.get_model().query_tokens_pc, model.get_model().query_tokens_2d]:
                p.requires_grad = True
            for module in [model.get_model().Qformer_pc, model.get_model().Qformer_2d, model.get_model().mm_projector_3d, model.get_model().mm_projector_2d, model.get_model().semanticmap_tokenizer]:
                for p in module.parameters():
                    p.requires_grad = True
            if hasattr(model.get_model(), 'pointnet2'):
                for p in model.get_model().pointnet2.parameters():
                    p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            # for p in model.get_model().mm_projector_3d.parameters():
            #     p.requires_grad = False
            # for p in model.get_model().mm_projector_2d.parameters():
            #     p.requires_grad = False
            for p in [model.get_model().query_tokens_pc, model.get_model().query_tokens_2d]:
                p.requires_grad = False
            for module in [model.get_model().Qformer_pc, model.get_model().Qformer_2d, model.get_model().mm_projector_3d, model.get_model().mm_projector_2d, model.get_model().semanticmap_tokenizer]:
                for p in module.parameters():
                    p.requires_grad = False
            if hasattr(model.get_model(), 'pointnet2'):
                for p in model.get_model().pointnet2.parameters():
                    p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector_3d.to(dtype=compute_dtype, device=training_args.device)
            model.get_model().mm_projector_2d.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    trainer.save_fn = safe_save_model_for_hf_trainer  # added by yinhang
    # model.get_model().get_vision_tower().to(dtype=torch.float32)  # added by yinhang
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # trainer.save_state()  # save file 'trainer_state.json'

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

        # 1. 3D-LLM embedding.shape
        # 2. random.choice换成pooling
        # 3. llama的token数量降低
        # 4. 做出定量和定性（回答正确的例子）的结果（是否需要回到2D上做）