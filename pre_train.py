import pandas as pd
from functools import reduce
import re

from src.utils import (
    prepare_ft_messages_multirouter,
    balance_dataset,
)

def extract_number(s):
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None

def process_dataframes(json_files_with_scores):
    dataframes = []
    weight_mapping = {}

    for json_file, score in json_files_with_scores:
        df = pd.read_json(json_file)

        precision = df.iloc[0][6]
        params = extract_number(df.iloc[0][7])

        weight = params * 2 if precision == 'fp16' else params
        weight_mapping[score] = weight

        df['source'] = df['source'].apply(lambda x: str(x) if isinstance(x, list) else x)

        dataframes.append(df)

    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=['prompt', 'source', 'gpt4_response'], how='outer'),
        dataframes
    )
    
    model_columns = [score for _, score in json_files_with_scores]
    model_mapping = {model: idx for idx, model in enumerate(model_columns)}

    df_merged["messages"] = prepare_ft_messages_multirouter(df_merged, model_columns, model_mapping)

    cost_effectiveness = df_merged[model_columns].div(pd.Series(weight_mapping))

    df_merged["routing_label"] = cost_effectiveness.idxmax(axis=1).map(model_mapping)

    balanced_train_df = balance_dataset(df_merged, key="routing_label")

    print(f"Train size: {len(balanced_train_df)}")

    output_file = "train_data_sample.jsonl"
    n_sample = 10000
    max_samples = min(n_sample, len(balanced_train_df))
    subsampled_df = balanced_train_df.sample(n=max_samples, random_state=42)
    subsampled_df.to_json(output_file, orient="records", lines=True)

if __name__ == "__main__":
    
    # Add models of choice along with their score column name...
    json_files_with_scores = [
        ("intermediate_llama3_2_3b.json", "llama3_2_3b_score"),
        ("intermediate_nemotron_70b.json", "nemotron_70b_score"),
    ]
    process_dataframes(json_files_with_scores)
