import fire
import os
import subprocess
import random
import string


def generate_model_tag(model_id: str) -> str:
    """
    Constructs a finetuned model ID based on the Anyscale endpoints convention.
    """
    username = os.environ.get("ANYSCALE_USERNAME")
    if username:
        username = username.strip().replace(" ", "")[:5]
        if len(username) < 5:
            padding_char = username[-1] if username else "a"
            username += padding_char * (5 - len(username))
    else:
        username = "".join(random.choices(string.ascii_lowercase, k=5))
    suffix = "".join(random.choices(string.ascii_lowercase, k=5))
    return f"{model_id}:{username}:{suffix}"


def main(ft_config_path):
    """
    Submit a finetuning job with a configuration file.
    """

    entrypoint = f"llmforge dev finetune {ft_config_path}"

    result = subprocess.run(entrypoint, check=True, shell=True)
    assert result.returncode == 0, "Finetuning failed."


if __name__ == "__main__":
    fire.Fire(main)
