from llm.llm_configurations import LLM, MODEL_REGISTRY
from llm.llm_configurations import LLM_COST_STUB_RESPONSE_IS_PROMPT, LLM_YIELD_RESULT_WITHOUT_API,  LLM_COST_STUB_RESPONSE_IS_LIKE_DAFNYBENCH
from llm.llm_open_ai import OpenAI_LLM
from llm.llm_amazon_bedrock import AmazonBedrock_LLM

def create_llm(name: str, model: str, **kwargs) -> LLM:
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}")

    info = MODEL_REGISTRY[model]

    if info.provider == "openai":
        return OpenAI_LLM(name, info, **kwargs)

    if info.provider == "bedrock":
        return AmazonBedrock_LLM(name, info, **kwargs)
    
    if info.provider == "debug":
        if info.model_id == "cost_stub_almost_real":
            return LLM_COST_STUB_RESPONSE_IS_PROMPT(name, info)
        elif info.model_id == "cost_stub_response_dafnybench":
            return LLM_COST_STUB_RESPONSE_IS_LIKE_DAFNYBENCH(name, info)
        elif info.model_id == "without_api":
            return LLM_YIELD_RESULT_WITHOUT_API(name, info)
        else:
            raise RuntimeError("Invalid Options for debug provider")
    raise RuntimeError("No valid option provided")
        