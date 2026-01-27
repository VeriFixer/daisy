import utils.global_variables as gl
import llm.llm_configurations as llm_configurations
import llm.llm_pipeline as llm_pipeline
import llm.llm_open_ai as llm_open_ai
from llm.llm_create import create_llm

base_prompt = gl.BASE_PROMPT

localization_base_prompt = gl.LOCALIZATION_BASE_PROMPT

system_prompt = gl.SYSTEM_PROMPT

global_options = llm_pipeline.GlobalOptions(
    dafny_exec= gl.DAFNY_EXEC,
    temp_dir= gl.TEMP_FOLDER,
    #llm_results_dir=gl.LLM_RESULTS_DIR_TEST,
    llm_results_dir=gl.BASE_PATH / "results/dafny_llm_results_testing_different_models_very_fast",
    assertion_dataset_path=gl.DAFNY_ASSERTION_DATASET_TEST

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
run_options.skip_original_verification=True

print("GET Cost Statistics of: Running all experiment refering to ORACLE position and all retrieval example methods")
# Cost estimation llm to get estimated costs of running one full evaluation
llm_pipeline.evaluate_all(llm_cost_stub, global_options, run_options)
llm_cost_stub.get_cost_statistics(llm_configurations.MODEL_REGISTRY["gpt-5.2"])
llm_cost_stub.get_my_cost_statisitcs()
# Ability to see all prompts as they are send to the LLMs


run_options.skip_verification= False
#llm_pipeline.evaluate_all(llm_without_api, global_options, run_options)


run_options.skip_original_verification = True

########################################################################################################
########################################################################################################
########################################################################################################
def perform_prompt_localizaiton_and_inferece(llm : llm_configurations.LLM, run_options : llm_pipeline.RunOptions, global_options : llm_pipeline.GlobalOptions):
    print(f"Testing name {llm.name} model {llm.model}")
    print("Beggin Prompt Localization")
    run_options.skip_verification=True
    run_options.only_get_assert_candidate = False
    run_options.only_get_location = True
    run_options.only_verify = False
    llm_pipeline.evaluate_all(llm, global_options, run_options)
    print("End Prompt Localization")

    llm.get_my_cost_statisitcs()
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification= True
    run_options.only_get_assert_candidate = True
    run_options.only_get_location = False
    run_options.only_verify = False
    llm_pipeline.evaluate_all(llm, global_options, run_options)
    print("END Prompt Assertion Candidate")
    llm.get_my_cost_statisitcs()

def perform_verification(llm : llm_configurations.LLM, run_options : llm_pipeline.RunOptions, global_options : llm_pipeline.GlobalOptions):
    print(f"Beggin Verification {llm.name} model {llm.model}")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification= False
    run_options.only_get_assert_candidate = False
    run_options.only_get_location = False
    run_options.only_verify = True
    llm_pipeline.evaluate_all(llm, global_options, run_options, parallel_run=True)

name = "claude-sonnet-4.5"
llmsonnet = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llmsonnet, run_options, global_options)

name = "claude-haiku-4.5"
llmhaiku = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llmhaiku, run_options, global_options)

name = "claude-opus-4.5"
llmopus = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llmopus, run_options, global_options)

name = "deepseek-r1"
llmdeep = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llmdeep, run_options, global_options)

name ="qwen3-coder-480b" 
llmqwen = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llmqwen, run_options, global_options)

name ="qwen3-coder-30b" 
llmqwen30 = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llmqwen30, run_options, global_options)

name ="llama-3.3-70b" 
llmllama = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llmllama, run_options, global_options)

name ="gpt-4.1" 
llm41 = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llm41, run_options, global_options)

name ="gpt-5.2" 
llm52 = create_llm(name, name)
#perform_prompt_localizaiton_and_inferece(llm52, run_options, global_options)


perform_verification(llmsonnet, run_options, global_options)
perform_verification(llmhaiku, run_options, global_options)
perform_verification(llmopus, run_options, global_options)
perform_verification(llmdeep, run_options, global_options)
perform_verification(llmqwen, run_options, global_options)
perform_verification(llmqwen30, run_options, global_options)
perform_verification(llmllama, run_options, global_options)
perform_verification(llm41, run_options, global_options)
perform_verification(llm52, run_options, global_options)