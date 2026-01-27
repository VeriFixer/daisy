import json
import time
import llm.llm_configurations as llm_configurations
import requests
import os
import boto3
# You will no longer need the 'requests' library

class AmazonBedrock_LLM(llm_configurations.LLM):
    def __init__(
        self,
        name: str,
        model: llm_configurations.ModelInfo,
        # The api_key is now primarily for mock mode, as boto3 pulls from env
        verbose: bool = False,
    ):
        super().__init__(name, model)
        self.model = model
        self.verbose = verbose
        self.chat_history = []
        self.system_prompt = "" # Assuming this is set elsewhere

        api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key:
            api_key = "NO_KEY"
        else:
            print("BEDROCK KEY PROVIDED")

        region = os.getenv("AWS_DEFAULT_REGION")
        if not region:
            api_region = "NO_REGION"
            print("NO BEDROCK API_DEFAULT_REGION provided, us-east-2 tends to be the better one")

        if api_key == "NO_KEY":
            print("NO BEDROCK API key provided — running in mock mode")
            self.client = None # Use None to signify mock mode
            return
            
        # Boto3 will automatically pick up the API key from the environment
        # variable AWS_BEARER_TOKEN_BEDROCK
        self.client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=region
        )


    def _get_response(self, prompt: str) -> str:
        # ... (timing, chat_history update, _trim_context) ...
        if self.client is None:
            return "Mock Reply"

        # 1. Format messages for the standard Converse API
        # The Converse API requires content to be a list of dictionaries (e.g., [{'text': '...'}]
        bedrock_messages = [
            {"role": msg["role"], "content": [{"text": msg["content"]}]}
            for msg in self.chat_history
        ]

        prompt_message = {"role": "user", "content" : [{"text" : prompt}]}
        
        bedrock_messages.append(prompt_message)
        # 2. Call the standardized Converse API
        response = self.client.converse(
          modelId=self.model.model_id,
          messages=bedrock_messages,
          system = [{"text": self.system_prompt}] 
        )
     

        # 3. Extract the response
        reply = response['output']['message']['content'][0]['text']

        # ... (chat_history append, return reply)
        self.chat_history.append({"role": "assistant", "content": reply})
        return reply