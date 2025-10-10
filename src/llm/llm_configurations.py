import json

class LLM:
    def __init__(self, name):
        self.name = name 
        self.chat_history = []
        self.system_prompt = ""
    def get_response(self, prompt):
        raise NotImplementedError("Subclasses must implement the prompt method")

    def get_name(self):
        return self.name
    
    def set_system_prompt(self, message):
        self.system_prompt = message

    def reset_chat_history(self):
        self.chat_history = []

    def get_chat_history(self):
        return self.chat_history



# Stub for LLM to account number of bytes etc

# Rule of tumb for LLM 1 token ~= 4 chars in English
# Will high ball number of chars in prompt equal to the ones on the response
class LLM_EMPTY_RESPONSE_STUB(LLM):  # 'extends' should be 'LLM_STUB(LLM)'
    def __init__(self,name):
        super().__init__(name)  


    def get_response(self, prompt):  # Fix indentation
        return "" # Return exacly unchanged prompt is a good upper bound in total size


class LLM_COST_STUB_RESPONSE_IS_PROMPT(LLM):  # 'extends' should be 'LLM_STUB(LLM)'
    def __init__(self,name):
        super().__init__(name)  
        self.total_chars_prompted = 0
        self.total_chars_response = 0
        self.prompt_number = 0
    def reset_all_measurement(self):
        self.total_chars_prompted = 0
        self.total_chars_response = 0
        self.prompt_number = 0

    def get_response(self, prompt):  # Fix indentation
        chat_history_size = sum([len(s) for s in self.chat_history])
        self.total_chars_prompted += len(prompt) + chat_history_size + len(self.system_prompt)
        self.prompt_number += 1
        self.chat_history.append(prompt) # orinal prompt
        if("The task is on finding the fix position" in prompt): #fix position prompt
          response = json.dumps([123,243])
        else:
          #response = json.dumps([[f"assert 123452==123452;"]])
          response = json.dumps([[f"assert 123452==123452 && {i} == {i};" for i in range(10)]*2])
        
        self.total_chars_response += len(response)
        self.chat_history.append(response)
        return response # Return exacly unchanged prompt is a good upper bound in total size
    
    def get_cost_statistics(self, model_name, price_per_1M_tokens_input, price_per_1M_tokens_output, how_many_chars_per_token):
        num_tokens_input = self.total_chars_prompted / how_many_chars_per_token
        num_tokens_output = self.total_chars_response / how_many_chars_per_token
        
        cost_input = (num_tokens_input / 1_000_000) * price_per_1M_tokens_input
        cost_output = (num_tokens_output / 1_000_000) * price_per_1M_tokens_output
        total_cost = cost_input + cost_output

        print(f"Expected Price Modes {model_name}")
        print(f"{'Statistic':<20}{'Value':<20}")
        print("=" * 40)
        print(f"{'Total Prompts ':<20}{self.prompt_number:<20}")
        print(f"{'Total Chars Prompted ':<20}{self.total_chars_prompted:<20}")
        print(f"{'Total Chars Response ':<20}{self.total_chars_response:<20}")
        print(f"{'Total Tokens Input ':<20}{num_tokens_input:<20.2f}")
        print(f"{'Total Tokens Output ':<20}{num_tokens_output:<20.2f}")
        print(f"{'Cost Input ($) ':<20}{cost_input:<20.6f}")
        print(f"{'Cost Output ($) ':<20}{cost_output:<20.6f}")
        print(f"{'Total Cost ($) ':<20}{total_cost:<20.6f}")
        print("=" * 40)

# This allows to test setup through CHAT
class LLM_YIELD_RESULT_WITHOUT_API(LLM):  # 'extends' should be 'LLM_STUB(LLM)'
    def __init__(self,name):
        super().__init__(name)  

    def get_response(self, prompt): 
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
