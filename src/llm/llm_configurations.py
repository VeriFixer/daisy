import json

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelInfo:
    provider: str          # "openai" | "bedrock"
    model_id: str          # provider-specific model id
    max_context: int
    cost_1M_in : float
    cost_1M_out : float


MODEL_REGISTRY = {
    # Claude (Bedrock)
    # Source: https://www.anthropic.com/claude/opus (Source 1.1)
   "claude-opus-4.5": ModelInfo(
        provider="bedrock",
        model_id="us.anthropic.claude-opus-4-5-20251101-v1:0",
        max_context=200_000, # Max context can reach 200K standard, with tiered pricing beyond
        cost_1M_in=5.00,
        cost_1M_out=25.00
    ),
    # Source: https://www.cursor-ide.com/blog/claude-sonnet-4-5-pricing (Source 1.2)
    "claude-sonnet-4.5": ModelInfo(
        provider="bedrock",
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_context=200_000, # Max context can reach 200K standard, with tiered pricing beyond
        cost_1M_in=3.00,
        cost_1M_out=15.00
    ),

    "claude-haiku-4.5": ModelInfo(
        provider="bedrock",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        max_context=200_000, # Max context can reach 200K standard, with tiered pricing beyond
        cost_1M_in=1.00,
        cost_1M_out=5.00
    ),
 
 
    # --- Open-source on Bedrock (On-Demand Pricing) ---
    #https://aws.amazon.com/bedrock/pricing/
    "deepseek-r1": ModelInfo(
        provider="bedrock",
        model_id="us.deepseek.r1-v1:0",
        max_context=64_000, # DeepSeek API documentation suggests 64K context
        cost_1M_in=1.35, # $0.00135 per 1K tokens
        cost_1M_out=5.40  # $0.0054 per 1K tokens
    ),
    "qwen3-coder-480b": ModelInfo(
        provider="bedrock",
        model_id="qwen.qwen3-coder-480b-a35b-v1:0",
        max_context=262_000, # Context window of 262K tokens
        cost_1M_in=0.45,     # Low-end quote from Source 3.2, rounded from $0.22/M to be conservative
        cost_1M_out=1.8
    ),
    # Source: OpenRouter API comparison, often reflecting Bedrock rates: https://openrouter.ai/compare/amazon/nova-premier-v1/qwen/qwen3-coder (Source 3.2, lower quote)
    "qwen3-coder-30b": ModelInfo(
        provider="bedrock",
        model_id="qwen.qwen3-coder-30b-a3b-v1:0",
        max_context=128_000,
        cost_1M_in=0.15,     # Highly competitive pricing, derived from 1K token rates (Source 3.2, lower quote)
        cost_1M_out=0.60
    ),
    # Source: https://www.nops.io/blog/amazon-bedrock-pricing/ (Source 5.1)
    "llama-3.3-70b": ModelInfo(
        provider="bedrock",
        model_id="meta.llama3-3-70b-instruct-v1:0",
        max_context=128_000,
        cost_1M_in=0.72,  # Bedrock pricing is $0.002 per 1K tokens
        cost_1M_out=0.72   # Bedrock pricing is $0.002 per 1K tokens
    ),

    # OpenAI example
    "gpt-5.2": ModelInfo(
        provider="openai",
        model_id="gpt-5.2",
        max_context=400_000,
        cost_1M_in=1.75,
        cost_1M_out=14.00
    ),
    # Source: https://platform.openai.com/docs/models/gpt-5-mini (Source 7.3)
    "gpt-5-mini": ModelInfo(
        provider="openai",
        model_id="gpt-5-mini",
        max_context=400_000,
        cost_1M_in=0.25,
        cost_1M_out=2.00
    ), 
    "gpt-4.1": ModelInfo(
        provider="openai",
        model_id="gpt-4.1-2025-04-14",
        max_context=128_000,
        cost_1M_in=2.00,
        cost_1M_out=8.00
    ),

    # Stubs 
    "cost_stub_response_dafnybench": ModelInfo(
        provider="debug",
        model_id="cost_stub_response_dafnybench",
        max_context=128_000,
        cost_1M_in=0,
        cost_1M_out=0

    ),

    "cost_stub_almost_real": ModelInfo(
        provider="debug",
        model_id="cost_stub_almost_real",
        max_context=128_000,
        cost_1M_in=0,
        cost_1M_out=0
    ),

    # Debug Interactive
    "without_api": ModelInfo(
        provider="debug",
        model_id="without_api",
        max_context=128_000,
        cost_1M_in=0,
        cost_1M_out=0
    ),
}

class LLM:
    def __init__(self, name: str, model : ModelInfo):
        self.name = name 
        self.model = model 

        self.system_prompt = ""

        # Handled directly by This parent Class
        self.total_chars_prompted : int = 0
        self.total_chars_response : int = 0
        self.prompt_number : int = 0


        # Variable to be handled by the implementations
        self.reasoning_tokens_output : int = 0

        self.chat_history = []
    def __str__(self):
        return f"LLM({self.name})"
    
    def get_name(self):
        return self.name
    
    def set_system_prompt(self, message : str):
        self.system_prompt = message

    def reset_chat_history(self):
        self.chat_history: list[str] = []

    def get_chat_history(self):
        return self.chat_history

    def _get_response(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement the _generate_response method for API interaction.")
 
    def get_response(self, prompt: str) -> str:
        """
        Public method that handles logging and calls the subclass's implementation.
        """
        prompt_len = len(prompt)
        self.total_chars_prompted += prompt_len
        self.prompt_number += 1
        
        response = self._get_response(prompt)
        
        response_len = len(response)
        self.total_chars_response += response_len
        
        return response
    def get_my_cost_statisitcs(self):
        self.get_cost_statistics(self.model)

    def get_cost_statistics(self, model : ModelInfo):
        mi = model

        how_many_chars_per_token = 3
        num_tokens_input = self.total_chars_prompted / how_many_chars_per_token
        num_tokens_output = self.total_chars_response / how_many_chars_per_token
        
        cost_input = (num_tokens_input / 1_000_000) * mi.cost_1M_in
        cost_output = (num_tokens_output / 1_000_000) * mi.cost_1M_out

        cost_reasoning = (self.reasoning_tokens_output / 1_000_000) * mi.cost_1M_out
     
        total_cost = cost_input + cost_output + cost_reasoning

        print(f"Expected Prices Model name: {self.name} Model id: {model.model_id}")
        print(f"{'Statistic':<40}{'Value':<20}")
        print("=" * 40)
        print(f"{'Total Prompts ':<40}{self.prompt_number:<20}")
        print(f"{'Total Chars Prompted ':<40}{self.total_chars_prompted:<20}")
        print(f"{'Total Chars Response ':<40}{self.total_chars_response:<20}")
        print(f"{'Total Tokens Input ':<40}{num_tokens_input:<20.2f}")
        print(f"{'Total Tokens Output ':<40}{num_tokens_output:<20.2f}")
        print(f"{'Total Tokens Output Reason':<40}{self.reasoning_tokens_output:<20.2f}")
        print(f"{'Cost Input ($) ':<40}{cost_input:<20.6f}")
        print(f"{'Cost Output ($) ':<40}{cost_output:<20.6f}")
        print(f"{'Cost Output Reason($) ':<40}{cost_reasoning:<20.6f}")
        print(f"{'Total Cost ($) ':<40}{total_cost:<20.6f}")
        print("=" * 40)

    def reset_all_measurement(self):
        self.total_chars_prompted = 0
        self.total_chars_response = 0
        self.prompt_number = 0


# Stub for LLM to account number of bytes etc

# Rule of tumb for LLM 1 token ~= 3 chars in English
class LLM_EMPTY_RESPONSE_STUB(LLM):  # 'extends' should be 'LLM_STUB(LLM)'
    def _get_response(self, prompt : str) -> str:  # Fix indentation
        return "" # Return exacly unchanged prompt is a good upper bound in total size
class LLM_COST_STUB_RESPONSE_IS_LIKE_DAFNYBENCH(LLM):  # 'extends' should be 'LLM_STUB(LLM)'
    def _get_response(self, prompt:str):  # Fix indentation
        self.chat_history.append(prompt) # orinal prompt
        needle = " === TASK === "
        start_task_pos = prompt.find(needle)
        start_task_pos += len(needle)

        task_text = prompt[start_task_pos:]
        needle = "CODE:"

        avg_added_assertion = "assert 123452==123452 && 4 == 4;"*2

        start_code = task_text.find(needle)
        start_code += len(needle)
        end_code = task_text[start_code:].find("OUTPUT:")
        response = task_text[start_code:start_code + end_code] # Reponse size eqals to the code
        response += avg_added_assertion
        self.chat_history.append(response)
        return response # Return exacly unchanged prompt is a good upper bound in total size
    

class LLM_COST_STUB_RESPONSE_IS_PROMPT(LLM):  # 'extends' should be 'LLM_STUB(LLM)'
    def _get_response(self, prompt:str):  # Fix indentation
        self.chat_history.append(prompt) # orinal prompt
        if("JSON array of line numbers ONLY" in prompt): #fix position prompt
          response = json.dumps([10,11])
        else:
          response = json.dumps([[f"assert 123452==123452 && {i} == {i};" for i in range(10)]*2])
        self.chat_history.append(response)
        return response # Return exacly unchanged prompt is a good upper bound in total size
    
# This allows to test setup through CHAT
class LLM_YIELD_RESULT_WITHOUT_API(LLM):  # 'extends' should be 'LLM_STUB(LLM)'
    def _get_response(self, prompt:str): 
        self.chat_history.append(prompt)
        print("The prompt is Prompt\n")
        print(f"System Prompt: {self.system_prompt}\n") 
        print(f"Main Prompt: {prompt}\n") 

        response = input("Enter your response (write #END# to end):\n")  

        # Read multiple lines until an empty line is entered
        lines = [response]
        while True:
            line = input()
            if line == "#END#":
                break
            lines.append(line)

        response = "\n".join(lines)  # Combine lines into a single response
        self.chat_history.append(response)
        return response  # Return exactly what the user provided
