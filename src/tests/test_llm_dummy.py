from dafny.dafny_runner import *
from llm.llm_configurations import LLM_COST_STUB_RESPONSE_IS_PROMPT
import llm.llm_pipeline as llm_pipeline

import unittest
import tempfile
import os
import shutil
import json
from pathlib import Path
import utils.global_variables as gl
import utils.assertion_method_classes as assert_lib
from datasets.dafny_dataset_generator import dafny_file_dataset_generator

import llm.llm_configurations as llm

class LLM_DOUBLE_REPONSE(llm.LLM):  # 'extends' should be 'LLM_STUB(LLM)'
    def __init__(self,name:str):
        super().__init__(name)  

    def get_response(self, prompt:str):  # Fix indentation
        self.chat_history.append(prompt) # orinal prompt
        if("Only give at most two lines in the answer" in prompt): #fix position prompt
          response = json.dumps([10,11])
        else:
          response = json.dumps([[f"assert 123452==123452 && {i} == {i};" for i in range(10)] for _ in range(2)])
        
        return response # Return exacly unchanged prompt is a good upper bound in total size



class TestLLM(unittest.TestCase):
    def setUp(self):
        self.llm_double = LLM_DOUBLE_REPONSE("test_stub_double")
        self.llm_example = LLM_DOUBLE_REPONSE("test_stub_example")

        self.dafny_exec = gl.DAFNY_EXEC

        if not os.path.exists(self.dafny_exec):
            self.fail(f"DAFNY_EXEC not found: {self.dafny_exec}")
       
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)

        # Change these lines do the bellow to be able to debug temporary files
        #self.dataset_temp_dir = tempfile.TemporaryDirectory()
        #self.dataset_temp_path = Path(self.outputdir.name)

        self.dataset_temp_dir = gl.TEMP_FOLDER / "dat_temp"
        self.dataset_temp_path = self.dataset_temp_dir

        self.llm_temp_dir = gl.TEMP_FOLDER / "llm_temp"
        self.llm_temp_path = self.llm_temp_dir

        self.data_path_assertive: Path = gl.UNIT_TESTS_DIR / "files/dataset_creator_assertion_group_test/assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        self.dataset_temp_dir.mkdir(parents=True, exist_ok=True)
        self.llm_temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate dataset artifacts into the output_path using the dafny wrapper
        dafny_file_dataset_generator(
            gl.DAFNY_EXEC,
            self.data_path_assertive,
            self.dataset_temp_path,
            self.tmp_path,
            1,
        )

        # The dataset generator is expected to create a directory named after the program
        self.expected_dir = (
            self.dataset_temp_path
            / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        )

        self.global_options = llm_pipeline.GlobalOptions(
            dafny_exec= gl.DAFNY_EXEC,
            temp_dir= gl.TEMP_FOLDER,
            llm_results_dir=self.llm_temp_path,
            assertion_dataset_path=self.dataset_temp_path
        )

        self.run_options = llm_pipeline.RunOptions(
            number_assertions_to_test=-1, # if -1 test all assetions
            number_rounds=1, # number of indepedent rounds in each assertion
            number_retries_chain=1, #number of retries it tries to ask fix for previous wrong andwer

            add_error_message = True,
            remove_empty_lines=True,
            change_assertion_per_text="", #Assertion with oracle to retrieve position
            base_prompt=gl.BASE_PROMPT,
            localization_base_prompt=gl.LOCALIZATION_BASE_PROMPT,
            examples_to_augment_prompt_type = llm_pipeline.ExampleStrategies.DYNAMIC, # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options    
                                                    # DYNAMIC represents my code/error message embeddings
            examples_weight_of_error_message = 0.5, # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
            number_examples_to_add=3,

            # Only used in LLM_EXAMPLE (used DYNAMIC 3 examples 0,5 underneath)
            
            # Fields ignore for all except LLM_EXAMPLE
            examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.DYNAMIC, # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options    
                                                    # DYNAMIC represents my code/error message embeddings
            examples_weight_of_error_message_pos = 0.5, # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
            number_examples_to_add_pos=3,

            limit_example_length_bytes = 1200, # Option Not implemented
            verifier_output_filter_warnings = True, # If 1 only errors are passed in the verifier ouput string
            system_prompt=gl.SYSTEM_PROMPT,
            skip_original_verification= True,

            localization = llm_pipeline.LocStrategies.LLM_EXAMPLE,
            # Options related with if will prompt LLM or gathered from saved results
            skip_verification= True,
            only_verify = True,
            only_get_location = True,
            only_get_assert_candidate = True,
        )


    def _require_localization_artifact(self):
        # path that test_run_llm_localization_no_example creates
        result_dir = self.llm_temp_path / "test_stub_double__nAssertions_ALL_nRounds_1_nRetries_1_addError_True_ExType_ExampleStrategies.NONE_loc_LocStrategies.LLM"
        expected_program_folder = result_dir / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        localization_path = expected_program_folder / "method_start_0_as_start_67_end_127" / "localization"
        localization_file = localization_path / "localization_raw_response.txt"

        if not localization_file.exists():
            self.skipTest(f"Skipping because localization artifact not found: {localization_file}")

    def _require_assertions_artifact(self):
        result_dir = self.llm_temp_path / "test_stub_double__nAssertions_ALL_nRounds_1_nRetries_1_addError_True_ExType_ExampleStrategies.NONE_loc_LocStrategies.LLM"
        expected_program_folder = result_dir / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        assertions_list_path = expected_program_folder / "method_start_0_as_start_67_end_127" / "assertions_list"
        assertions_list_file = assertions_list_path / "assertions_parsed.json"

        if not assertions_list_file.exists():
            self.skipTest(f"Skipping because assertions artifact not found: {assertions_list_file}")

    def tearDown(self):
        self.tempdir.cleanup()
        #self.outputdir.cleanup()

    def test_run_llm_localization_no_example(self):  
        self.run_options.skip_verification=True
        self.run_options.only_get_assert_candidate = False
        self.run_options.only_get_location = True
        self.run_options.only_verify = False

        self.run_options.localization = llm_pipeline.LocStrategies.LLM
        self.run_options.examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.NONE
        self.run_options.examples_to_augment_prompt_type  = llm_pipeline.ExampleStrategies.NONE

        (self.assertion_groups, self.model_dir, self.assertion_dataset_info_df) = llm_pipeline.setup_llm_evaluate_all(self.llm_double, self.global_options, self.run_options)
        self.test_group : assert_lib.assertionGroup
        for group in self.assertion_groups:
            if("HasAddends" in str(group)):
                self.test_group = group
        self.assertIsNotNone(self.test_group, "test group should be not none")

        llm_pipeline.process_group(self.test_group, self.model_dir,
                                   self.assertion_dataset_info_df, self.llm_double, 
                                   self.assertion_groups, self.run_options, self.global_options)
        
        result_dir = self.llm_temp_path / "test_stub_double__nAssertions_ALL_nRounds_1_nRetries_1_addError_True_ExType_ExampleStrategies.NONE_loc_LocStrategies.LLM"
        self.assertTrue(result_dir.exists() and result_dir.is_dir(),
                        f"Expected program folder not found: {result_dir}")
        expected_program_folder = result_dir / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        self.assertTrue(expected_program_folder.exists() and expected_program_folder.is_dir(),
                        f"Expected program folder not found: {expected_program_folder}")

        localization_path = expected_program_folder / "method_start_0_as_start_67_end_127" / "localization"
        self.assertTrue(localization_path.exists() and localization_path.is_dir(),
                        f"Expected localization directory not found: {localization_path}")

        localization_file = localization_path / "localization_raw_response.txt"
        self.assertTrue(localization_file.exists() and localization_file.is_file(),
                        f"Expected localization_raw_response.txt not found at: {localization_file}")

        content = localization_file.read_text(encoding="utf-8").strip()
        self.assertEqual(content, "[10, 11]", f"Unexpected localization content: {content}")

    def test_llm_inference_no_example(self):
        self._require_localization_artifact()
        self.run_options.skip_verification=True
        self.run_options.only_get_assert_candidate = True
        self.run_options.only_get_location = False
        self.run_options.only_verify = False

        self.run_options.localization = llm_pipeline.LocStrategies.LLM
        self.run_options.examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.NONE
        self.run_options.examples_to_augment_prompt_type  = llm_pipeline.ExampleStrategies.NONE

        (self.assertion_groups, self.model_dir, self.assertion_dataset_info_df) = llm_pipeline.setup_llm_evaluate_all(self.llm_double, self.global_options, self.run_options)
        self.test_group : assert_lib.assertionGroup
        for group in self.assertion_groups:
            if("HasAddends" in str(group)):
                self.test_group = group
        self.assertIsNotNone(self.test_group, "test group should be not none")

        llm_pipeline.process_group(self.test_group, self.model_dir,
                                   self.assertion_dataset_info_df, self.llm_double, 
                                   self.assertion_groups, self.run_options, self.global_options)
        
        result_dir = self.llm_temp_path / "test_stub_double__nAssertions_ALL_nRounds_1_nRetries_1_addError_True_ExType_ExampleStrategies.NONE_loc_LocStrategies.LLM"
        self.assertTrue(result_dir.exists() and result_dir.is_dir(),
                        f"Expected program folder not found: {result_dir}")
        expected_program_folder = result_dir / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        self.assertTrue(expected_program_folder.exists() and expected_program_folder.is_dir(),
                        f"Expected program folder not found: {expected_program_folder}")

        assertions_list_path = expected_program_folder / "method_start_0_as_start_67_end_127" / "assertions_list"
        self.assertTrue(assertions_list_path.exists() and assertions_list_path.is_dir(),
                        f"Expected assertions_list directory not found: {assertions_list_path}")

        assertions_list_file = assertions_list_path / "assertions_parsed.json"
        self.assertTrue(assertions_list_file.exists() and assertions_list_file.is_file(),
                        f"Expected assertions_parserd.json not found at: {assertions_list_file}")

        content = assertions_list_file.read_text(encoding="utf-8").strip()
        self.assertTrue("assert 123452==" in content, f"Unexpected localization content: {content}")
  
    def test_run_verification_on_changed_assertions(self):
        self._require_localization_artifact()
        self._require_assertions_artifact()
        self.run_options.skip_verification=False
        self.run_options.only_get_assert_candidate = False
        self.run_options.only_get_location = False
        self.run_options.only_verify = True

        self.run_options.localization = llm_pipeline.LocStrategies.LLM
        self.run_options.examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.NONE
        self.run_options.examples_to_augment_prompt_type  = llm_pipeline.ExampleStrategies.NONE

        (self.assertion_groups, self.model_dir, self.assertion_dataset_info_df) = llm_pipeline.setup_llm_evaluate_all(self.llm_double, self.global_options, self.run_options)
        self.test_group : assert_lib.assertionGroup
        for group in self.assertion_groups:
            if("HasAddends" in str(group)):
                self.test_group = group
        self.assertIsNotNone(self.test_group, "test group should be not none")

        llm_pipeline.process_group(self.test_group, self.model_dir,
                                   self.assertion_dataset_info_df, self.llm_double, 
                                   self.assertion_groups, self.run_options, self.global_options)
        
        result_dir = self.llm_temp_path / "test_stub_double__nAssertions_ALL_nRounds_1_nRetries_1_addError_True_ExType_ExampleStrategies.NONE_loc_LocStrategies.LLM"
        self.assertTrue(result_dir.exists() and result_dir.is_dir(),
                        f"Expected program folder not found: {result_dir}")
        expected_program_folder = result_dir / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        self.assertTrue(expected_program_folder.exists() and expected_program_folder.is_dir(),
                        f"Expected program folder not found: {expected_program_folder}")

        verification_path = expected_program_folder / "method_start_0_as_start_67_end_127" / "verification" / "Assertion_id_3"
        self.assertTrue(verification_path.exists() and verification_path.is_dir(),
                        f"Expected assertions_list directory not found: {verification_path}")

        verification_file = verification_path / "program_with_new_assertions.txt"
        self.assertTrue(verification_file.exists() and verification_file.is_file(),
                        f"Expected assertions_parserd.json not found at: {verification_file}")
        
        content = verification_file.read_text(encoding="utf-8").strip()
        self.assertTrue(content.count("assert 123452==123452 && 3 == 3") == 2 , f"Expectd two equal assertions tt be inserted on this file")

        # Check if first match appears on line 11 and next on line 13
        lines = content.splitlines()
        target = "assert 123452==123452 && 3 == 3"

        occurrences = [i for i, line in enumerate(lines) if target in line]

        self.assertEqual(
            occurrences,
            [11, 13],
            f"Expected assertion occurrences on lines 11 and 13, but got: {occurrences}"
        )

    def test_run_llm_localization_example(self):
        self.run_options.skip_verification=True
        self.run_options.only_get_assert_candidate = False
        self.run_options.only_get_location = True
        self.run_options.only_verify = False

        self.run_options.examples_to_augment_prompt_type  = llm_pipeline.ExampleStrategies.DYNAMIC
        self.run_options.examples_weight_of_error_message = 0.5 
        self.run_options.number_examples_to_add=3

        self.run_options.localization = llm_pipeline.LocStrategies.LLM_EXAMPLE
        self.run_options.examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.DYNAMIC
        self.run_options.examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.DYNAMIC
        self.run_options.examples_weight_of_error_message_pos = 0.5
        self.run_options.number_examples_to_add_pos=3

        (self.assertion_groups, self.model_dir, self.assertion_dataset_info_df) = llm_pipeline.setup_llm_evaluate_all(self.llm_example, self.global_options, self.run_options)
        self.test_group : assert_lib.assertionGroup
        for group in self.assertion_groups:
            if("HasAddends" in str(group)):
                self.test_group = group
        self.assertIsNotNone(self.test_group, "test group should be not none")

        llm_pipeline.process_group(self.test_group, self.model_dir,
                                   self.assertion_dataset_info_df, self.llm_example, 
                                   self.assertion_groups, self.run_options, self.global_options)
        
        result_dir = self.llm_temp_path / "test_stub_example__nAssertions_ALL_nRounds_1_nRetries_1_addError_True_addExamp_3_alpha_0.5_ExType_ExampleStrategies.DYNAMIC_loc_LocStrategies.LLM_EXAMPLE"
        self.assertTrue(result_dir.exists() and result_dir.is_dir(),
                        f"Expected program folder not found: {result_dir}")
        expected_program_folder = result_dir / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        self.assertTrue(expected_program_folder.exists() and expected_program_folder.is_dir(),
                        f"Expected program folder not found: {expected_program_folder}")

        localization_path = expected_program_folder / "method_start_0_as_start_67_end_127" / "localization"
        self.assertTrue(localization_path.exists() and localization_path.is_dir(),
                        f"Expected localization directory not found: {localization_path}")

        localization_file = localization_path / "localization_prompt.txt"
        self.assertTrue(localization_file.exists() and localization_file.is_file(),
                        f"Expected localization_raw_response.txt not found at: {localization_file}")

        content = localization_file.read_text(encoding="utf-8").strip()
        self.assertTrue("Example 3 Error" in content, f"Contet of file {localization_file} expected Example errors in the prompt")


    def test_llm_inference_example(self):
        self.run_options.skip_verification=True
        self.run_options.only_get_assert_candidate = True
        self.run_options.only_get_location = False
        self.run_options.only_verify = False

        self.run_options.examples_to_augment_prompt_type  = llm_pipeline.ExampleStrategies.DYNAMIC
        self.run_options.examples_weight_of_error_message = 0.5 
        self.run_options.number_examples_to_add=3

        self.run_options.localization = llm_pipeline.LocStrategies.LLM_EXAMPLE
        self.run_options.examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.DYNAMIC
        self.run_options.examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.DYNAMIC
        self.run_options.examples_weight_of_error_message_pos = 0.5
        self.run_options.number_examples_to_add_pos=3

        (self.assertion_groups, self.model_dir, self.assertion_dataset_info_df) = llm_pipeline.setup_llm_evaluate_all(self.llm_example, self.global_options, self.run_options)
        self.test_group : assert_lib.assertionGroup
        for group in self.assertion_groups:
            if("HasAddends" in str(group)):
                self.test_group = group
        self.assertIsNotNone(self.test_group, "test group should be not none")

        llm_pipeline.process_group(self.test_group, self.model_dir,
                                   self.assertion_dataset_info_df, self.llm_example, 
                                   self.assertion_groups, self.run_options, self.global_options)
        result_dir = self.llm_temp_path / "test_stub_example__nAssertions_ALL_nRounds_1_nRetries_1_addError_True_addExamp_3_alpha_0.5_ExType_ExampleStrategies.DYNAMIC_loc_LocStrategies.LLM_EXAMPLE"
        self.assertTrue(result_dir.exists() and result_dir.is_dir(),
                        f"Expected program folder not found: {result_dir}")
        expected_program_folder = result_dir / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        self.assertTrue(expected_program_folder.exists() and expected_program_folder.is_dir(),
                        f"Expected program folder not found: {expected_program_folder}")

        assertions_list_path = expected_program_folder / "method_start_0_as_start_67_end_127" / "assertions_list"
        self.assertTrue(assertions_list_path.exists() and assertions_list_path.is_dir(),
                        f"Expected assertions_list directory not found: {assertions_list_path}")

        assertions_list_file = assertions_list_path / "assertions_prompt.txt"
        self.assertTrue(assertions_list_file.exists() and assertions_list_file.is_file(),
                        f"Expected assertions_parserd.json not found at: {assertions_list_file}")

        content = assertions_list_file.read_text(encoding="utf-8").strip()
        self.assertTrue("Example 3 Error" in content, f"Prompt should have examples there")
          





if __name__ == "__main__":
    unittest.main()  