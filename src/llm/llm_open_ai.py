import llm.llm_configurations as llm_configurations

from openai import OpenAI
import openai
import time

# Enforcing max data rate of 500 per minute
class OpenAI_LLM(llm_configurations.LLM):
    def __init__(self, name, model="gpt-4o-mini", max_context_size=128000, openaiKey = "NO_KEY", verbose=0):
        super().__init__(name)
        self.model = model
        self.chat_history = []  # Store conversation history
        self.max_context_size = max_context_size  
        self.openaiKey = openaiKey
        self.verbose = verbose

        if(self.openaiKey == "NO_KEY"): # -1 used to mock openAI to test around machinery
          self.openai_client = None
          print("NO OPEN AI API key provided running in mock mode")
        else:
         print("API KEY provided")
         try:
            self.openai_client = OpenAI(
                api_key = openaiKey
            )  
            models = self.openai_client.models.list()
            if(verbose):
              print("Avaiable models")
              for model in models:
                print(model)
                
            print("API key is valid!")
         except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            exit()
         except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            exit()
         except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            exit()

    def get_response(self, prompt):
        # to not exceed the 500 messages per second
        time.sleep(0.12)
        self.chat_history.append({"role": "user", "content": prompt})  # Add user input
        self._trim_context()

        if(self.openaiKey == "NO_KEY"):
            reply = "Mock Reply"
        else:
            my_messages = [{"role":"developer","content":self.system_prompt} ] + self.chat_history
            #print(my_messages)
            response = self.openai_client.chat.completions.create(model=self.model,
               messages= my_messages)
            reply = response.choices[0].message.content 
        self.chat_history.append({"role": "assistant", "content": reply})  # Store AI response
        return reply

    def _trim_context(self):
        """Ensures the chat history fits within max_context_size tokens."""
        estimated_bytes = sum(len(m["content"])  for m in self.chat_history)  # Approximate token count
        if estimated_bytes > self.max_context_size:
            # Keep first message and trim middle, retaining latest exchanges
            first_message = self.chat_history[0]
            trimmed_messages = self.chat_history[1:]
            while estimated_bytes > self.max_context_size and len(trimmed_messages) > 1:
                trimmed_messages.pop(0)  # Remove oldest messages first
                estimated_bytes = sum(len(m["content"])  for m in [first_message] + trimmed_messages)
            self.chat_history = [first_message] + trimmed_messages  # Reconstruct conversation
