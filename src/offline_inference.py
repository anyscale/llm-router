from routellm.routers.causal_llm.configs import RouterModelConfig
from routellm.routers.causal_llm.llm_utils import load_prompt_format
from routellm.routers.causal_llm.model import CausalLLMClassifier


def single_example_inference(input):
    """
    Perform inference on a single example using a finetuned Causal LLM model.
    """
    # Load configs
    model_config = RouterModelConfig(
        model_id="meta-llama/Meta-Llama-3-8B",
        model_type="causal",
        flash_attention_2=False,
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

    # Inference
    model_output = model(input)
    return model_output
