import os
import unittest

import utils.global_variables as gl
from llm.llm_configurations import MODEL_REGISTRY, ModelInfo, LLM
from llm.llm_create import create_llm
from llm.llm_open_ai import OpenAI_LLM
from llm.llm_amazon_bedrock import AmazonBedrock_LLM

class TestAllModels(unittest.TestCase):

    def _run_basic_probe(self, llm : LLM):
        llm.set_system_prompt("You are a test harness")
        res = llm.get_response("Reply with only the word OKOK.")
        print(f"From {llm} got : {res}")
        self.assertIn("OKOK", res)

    # -------------------------
    # DEBUG / MOCK MODELS
    # -------------------------
    def test_debug_and_mock_models(self):
        for model_name, info in MODEL_REGISTRY.items():
            if info.provider != "debug":
                continue

            with self.subTest(model=model_name):
                llm = create_llm("debug_test", model_name)
                llm.set_system_prompt("You are a test harness")

                # Debug models have unpredictable responses at least we test the creation
                # res = llm.get_response("Reply with only the word OK.")
                # Debug models should never hit real APIs
    # -------------------------
    # OPENAI MODELS (REAL)
    # -------------------------
    def test_openai_models_real(self):
        if not gl.RUN_TEST_THAT_COST_MONEY:
            self.skipTest("Money-spending tests disabled")

        model_name = "gpt-5-mini" # works
        #model_name = "gpt-5.2" # works
        #model_name = "gpt-4.1" # works

        print(model_name)
        llm = create_llm(
            "real_openai_test",
            model_name
        )
        self._run_basic_probe(llm)

    # -------------------------
    # BEDROCK MODELS (REAL)
    # -------------------------
    def test_bedrock_models_real(self):
        if not gl.RUN_TEST_THAT_COST_MONEY:
            self.skipTest("Money-spending tests disabled")

        model_name = "qwen3-coder-30b"  # works 
        #model_name = "qwen3-coder-480b" # works 
        #model_name = "llama-3.3-70b" # works
        #model_name = "deepseek-r1" # works
        #model_name = "claude-opus-4.5" # works
        #model_name = "claude-sonnet-4.5" # works

        print(model_name)
        llm = create_llm(
            "real_amazon_bedrock_test",
            model_name
        )
        self._run_basic_probe(llm)

class TestModelRegistry(unittest.TestCase):

    def test_registry_entries_are_well_formed(self):
        self.assertGreater(len(MODEL_REGISTRY), 0)

        for name, info in MODEL_REGISTRY.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(info, ModelInfo)
            self.assertIn(info.provider, ("openai", "bedrock","debug"))
            self.assertIsInstance(info.model_id, str)
            self.assertGreater(info.max_context, 0)
            # cost fields exist and are sane
            self.assertIsInstance(info.cost_1M_in, float)
            self.assertIsInstance(info.cost_1M_out, float)
            self.assertGreaterEqual(info.cost_1M_in, 0.0)
            self.assertGreaterEqual(info.cost_1M_out, 0.0)


class TestLLMFactory(unittest.TestCase):

    def test_factory_creates_correct_provider(self):
        for model, info in MODEL_REGISTRY.items():
            llm = create_llm(
                name="test",
                model=model
            )

            if info.provider == "openai":
                self.assertIsInstance(llm, OpenAI_LLM)
            elif info.provider == "bedrock":
                self.assertIsInstance(llm, AmazonBedrock_LLM)
            elif info.provider != "debug":
                self.fail(f"Unknown provider {info.provider}")