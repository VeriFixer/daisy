import utils.global_variables as gl
import llm.llm_configurations as llm_configurations
import llm.llm_pipeline as llm_pipeline
import llm.llm_open_ai as llm_open_ai
from llm.llm_create import create_llm

base_prompt = gl.BASE_PROMPT

localization_base_prompt = gl.LOCALIZATION_BASE_PROMPT

system_prompt = gl.SYSTEM_PROMPT
# Test point for the test dataset and for the llm_results_dir_Test
global_options = llm_pipeline.GlobalOptions(
    dafny_exec= gl.DAFNY_EXEC,
    temp_dir= gl.TEMP_FOLDER,
    llm_results_dir=gl.LLM_RESULTS_DIR_TEST,
    assertion_dataset_path=gl.DAFNY_ASSERTION_DATASET
)

llm_cost_stub = create_llm("cost","cost_stub_almost_real")
llm_without_api = create_llm("without_api","without_api")

run_options = llm_pipeline.RunOptions(
    number_assertions_to_test=-1, # if -1 test all assetions
    number_rounds=1, # number of indepedent rounds in each assertion
    number_retries_chain=1, #number of retries it tries to ask fix for previous wrong andwer
    add_error_message = True,
    remove_empty_lines=True,
    change_assertion_per_text="", #Assertion with oracle to retrieve position
    base_prompt=base_prompt,
    localization_base_prompt=localization_base_prompt,
    examples_to_augment_prompt_type = llm_pipeline.ExampleStrategies.NONE, # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options    
                                               # DYNAMIC represents my code/error message embeddings
    examples_weight_of_error_message = 0.5, # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
    number_examples_to_add=3,
    # Fields ignore for all except LLM_EXAMPLE
    examples_to_augment_prompt_type_pos= llm_pipeline.ExampleStrategies.NONE, # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options    
                                               # DYNAMIC represents my code/error message embeddings
    examples_weight_of_error_message_pos = 0.5, # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
    number_examples_to_add_pos=3,

    limit_example_length_bytes = 1200, # Option Not implemented
    verifier_output_filter_warnings = True, # If 1 only errors are passed in the verifier ouput string
    system_prompt=system_prompt,
    skip_original_verification= True,
    localization = llm_pipeline.LocStrategies.ORACLE,
    # Options related with if will prompt LLM or gathered from saved results
    skip_verification= True,
    only_verify = False,
    only_get_location = False,
    only_get_assert_candidate = False,
)
#### COST ESTIMATION OF RUNNING WHAT IS ASKED ####

print("GET Cost Statistics of: Running all experiment refering to NONE on example retrieval using different position inferer techniques")
run_options.examples_to_augment_prompt_type=llm_pipeline.ExampleStrategies.NONE # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3

run_options.skip_verification=True
# Cost estimation llm to get estimated costs of running one full evaluation
#llm_pipeline.evaluate_all(llm_cost_stub, global_options, run_options)
#llm_cost_stub.get_cost_statistics(llm_configurations.MODEL_REGISTRY["claude-haiku-4.5"])
# Ability to see all prompts as they are send to the LLMs

run_options.localization = llm_pipeline.LocStrategies.HYBRID
run_options.examples_to_augment_prompt_type_pos=llm_pipeline.ExampleStrategies.DYNAMIC # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message_pos = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add_pos=3
run_options.skip_verification=False

llm_pipeline.evaluate_all(llm_without_api, global_options, run_options)
#exit()

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################

def perform_prompt_localizaiton_and_inferece(llm : llm_configurations.LLM, run_options : llm_pipeline.RunOptions, global_options : llm_pipeline.GlobalOptions):
    print(f"Testing name {llm.name} model {llm.model}")
    print("\nBeggin Prompt Localization\n")
    run_options.skip_verification=True
    run_options.only_get_assert_candidate = False
    run_options.only_get_location = True
    run_options.only_verify = False
    llm_pipeline.evaluate_all(llm, global_options, run_options)
    print("\nEnd Prompt Localization\n")

    llm.get_my_cost_statisitcs()
    print("\nBeggin Prompt Assertion Candidate\n")
    run_options.skip_verification= True
    run_options.only_get_assert_candidate = True
    run_options.only_get_location = False
    run_options.only_verify = False
    llm_pipeline.evaluate_all(llm, global_options, run_options)
    print("\nEND Prompt Assertion Candidate\n")
    llm.get_my_cost_statisitcs()

def perform_verification(llm : llm_configurations.LLM, run_options : llm_pipeline.RunOptions, global_options : llm_pipeline.GlobalOptions):
    print(f"\nBeggin Verification {llm.name} model {llm.model}\n")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification= False
    run_options.only_get_assert_candidate = False
    run_options.only_get_location = False
    run_options.only_verify = True
    llm_pipeline.evaluate_all(llm, global_options, run_options, parallel_run=True)

