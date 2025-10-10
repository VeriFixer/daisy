import os
from llm.llm_open_ai import *
import unittest

class TestOpenAILLM(unittest.TestCase):
    def test_open_ai_api_setup(self):
        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        # Sets model gpt4-mini and calls the model list function
        if self.openai_api_key is None:
          raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

        self.llm = OpenAI_LLM("gpt-4o-mini", model="gpt-4o-mini", max_context_size=128000, openaiKey=self.openai_api_key, verbose=1)
        self.llm.set_system_prompt("Dummy System Prompt")    

    def test_call_basic_prompt(self):
        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        # Sets model gpt4-mini and calls the model list function
        if self.openai_api_key is None:
          raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

        self.llm = OpenAI_LLM("gpt-4o-mini", model="gpt-4o-mini", max_context_size=128000, openaiKey=self.openai_api_key, verbose=1)
        self.llm.set_system_prompt("You are a Dafny expert") 
        # Uncomment to try a real prompt (obviously it cost as it uses API)
        #res = self.llm.get_response("Create a method to comput the maximum of a sequence.Return only the method code and specifications")
        #print(res)

    def test_trim_chat_history(self):
        self.llm = OpenAI_LLM("trim_dummy", model="trim_dummy", max_context_size=128000, openaiKey="NO_KEY", verbose=1)
        self.llm.set_system_prompt("Dummy System Prompt") 

        # Checks trim for chat history
        self.llm.get_response(60000 * "=")
        self.assertEqual(self.llm.chat_history[0]["content"][0], "=")
        self.assertEqual(self.llm.chat_history[1]["role"], "assistant")

        self.llm.get_response(60000 * "*")
        self.assertEqual(self.llm.chat_history[0]["content"][0], "=")
        self.assertEqual(self.llm.chat_history[1]["role"], "assistant")
        self.assertEqual(self.llm.chat_history[2]["content"][0], "*")

        self.llm.get_response(60000  * "-")
        self.assertEqual(self.llm.chat_history[0]["content"][0], "=")
        self.assertEqual(self.llm.chat_history[1]["role"], "assistant")
        self.assertEqual(self.llm.chat_history[2]["content"][0], "-")

        self.llm.get_response(60000 * "+")
        self.assertEqual(self.llm.chat_history[0]["content"][0], "=")
        self.assertEqual(self.llm.chat_history[2]["content"][0], "+")
        self.assertEqual(self.llm.chat_history[1]["role"], "assistant")
    
    #def test_dafny_reply_extraction(self):
    #   self.assertEqual(extract_dafny_code("direct dafny code"), "direct dafny code")
    #   self.assertEqual(extract_dafny_code("```dafny non direct dafny code ```"), "non direct dafny code")
    #   self.assertEqual(extract_dafny_code("```dafny non direct dafny code \n multiline ```"), "non direct dafny code \n multiline")

if __name__ == "__main__":
    unittest.main()
