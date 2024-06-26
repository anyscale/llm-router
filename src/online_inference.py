import pandas as pd
import json
import ray
from typing import Dict, Any, List
import copy
import openai
import time
import ray
from .utils import prepare_llm_queries, prepare_llm_judge_queries, parse_judge_responses


@ray.remote(num_cpus=0)
def get_llm_response(
    base_url: str,
    api_key: str,
    llm: str,
    temperature: float,
    max_tokens: int,
    pidx: int,
    messages: List[Dict[str, str]],
    max_retries=1,
    retry_interval=60,
) -> Dict[int, str]:
    """
    Use OpenAI's API to request completions from a specified LLM and manages request retries upon failures.
    """
    retry_count = 0
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    while retry_count <= max_retries:
        try:
            response = client.chat.completions.create(
                model=llm,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (pidx, response.choices[0].message.content)
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return (pidx, "")


def generate_batch_responses(
    base_url: str,
    api_key: str,
    llm: str,
    queries: Dict[int, Any],
    max_concurrent_queries: int,
    temperature: float,
    max_tokens: int,
    verbose: bool = False,
) -> Dict[int, str]:
    """
    This function manages online batch inference of queries using a specified LLM, tracking progress and handling responses.
    """
    print(f"Starting batch inference on {len(queries)} queries...")
    queue = copy.copy(queries)
    in_progress, responses = [], []

    start_time = time.time()
    while queue or in_progress:
        if len(in_progress) < max_concurrent_queries and queue:
            pidx, messages = queue.popitem()
            in_progress.append(
                get_llm_response.remote(
                    base_url, api_key, llm, temperature, max_tokens, pidx, messages
                )
            )
        ready, in_progress = ray.wait(in_progress, timeout=0.5)
        if verbose:
            print(
                f"# queries un-processed: {len(queue)}, in-progress: {len(in_progress)}, ready: {len(ready)}"
            )
        if ready:
            responses.extend(ray.get(ready))

    print(f"Done in {time.time() - start_time:.2f}sec.")
    return dict(responses)


def generate_mixtral_responses(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "https://api.endpoints.anyscale.com/v1",
    response_column: str = "mixtral_response",
) -> pd.DataFrame:
    """
    Generate Mixtral responses with Anyscale's public endpoint
    """
    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    mixtral_responses = generate_batch_responses(
        api_base,
        api_key,
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        llm_queries,
        max_concurrent_queries=25,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Add Mixtral responses as a column to the dataset
    dataset_df[response_column] = dataset_df.index.map(mixtral_responses)
    return dataset_df


def generate_llm_judge_labels(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    judge_llm: str = "gpt-4",
    answer_key: str = "mixtral_response",
    reference_key: str = "gpt4_response",
    label_key: str = "mixtral_score",
) -> pd.DataFrame:
    """
    Generate LLM-as-a-judge labels with OpenAI's API
    """
    with open("assets/judge_template.json") as f:
        judge_template = json.load(f)

    # Preprocess LLM-judge queries
    judge_queries = prepare_llm_judge_queries(
        dataset_df, judge_template, answer_key, reference_key
    )

    # Generate GPT-4 as a judge labels with OpenAI API
    judge_responses = generate_batch_responses(
        api_base,
        api_key,
        judge_llm,
        judge_queries,
        max_concurrent_queries=10,
        temperature=0,
        max_tokens=256,
        verbose=True,
    )

    # Parse judge responses
    labels, explanations = parse_judge_responses(judge_responses)

    # Add judge score as a label column
    dataset_df[label_key] = dataset_df.index.map(labels)

    return dataset_df
