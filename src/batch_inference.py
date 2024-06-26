# import numpy as np
# from pydantic import BaseModel
# import torch

# import copy
# from typing import Dict, List, Any

import ray
import torch
import time

import pandas as pd

from routellm.routers.causal_llm.configs import RouterModelConfig
from routellm.routers.causal_llm.llm_utils import (
    load_prompt_format,
    to_openai_api_messages,
)
from routellm.routers.causal_llm.model import CausalLLMClassifier


def batch_llm_inference(eval_data_df: pd.DataFrame):
    """TODO"""
    s = time.time()

    # Load configs
    model_config = RouterModelConfig(
        model_id="meta-llama/Meta-Llama-3-8B",
        model_type="causal",
        flash_attention_2=True,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
    )
    prompt_format = load_prompt_format(model_config.model_id)

    fn_constructor_kwargs = {
        "config": model_config,
        "ckpt_local_path": "routellm/causal_llm_gpt4_augmented",
        "score_threshold": 4,
        "prompt_format": prompt_format,
        "prompt_field": "messages",
        "additional_fields": [],
        "use_last_turn": False,
    }

    eval_ds = ray.data.from_pandas(eval_data_df, override_num_blocks=8)
    eval_ds = eval_ds.map(
        CausalLLMClassifier,
        fn_constructor_kwargs=fn_constructor_kwargs,
        num_gpus=1,  # each worker on a single gpu
        concurrency=torch.cuda.device_count(),
    )
    result_df = eval_ds.to_pandas()

    print(f"Done batch inference in {time.time() - s} seconds.")

    # result_df.to_csv(f"{output_fname}.csv", index=False)
    return result_df


def single_example_debug(input):
    """TODO"""
    # Load configs
    model_config = RouterModelConfig(
        model_id="meta-llama/Meta-Llama-3-8B",
        model_type="causal",
        flash_attention_2=True,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
    )
    prompt_format = load_prompt_format(model_config.model_id)

    # Load model
    model = CausalLLMClassifier(
        config=model_config,
        ckpt_local_path="routellm/causal_llm_gpt4_augmented",
        score_threshold=4,
        prompt_format=prompt_format,
        prompt_field="messages",
        additional_fields=[],
        use_last_turn=False,
    )

    model_output = model(input)
    print("predicted score:", model_output["score_pred"])
