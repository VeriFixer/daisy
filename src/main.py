import utils.global_variables as gl
import llm.llm_configurations as llm_configurations
import llm.llm_pipeline as llm_pipeline
import llm.llm_open_ai as llm_open_ai

base_prompt = gl.BASE_PROMPT

localization_base_prompt = gl.LOCALIZATION_BASE_PROMPT

system_prompt = gl.SYSTEM_PROMPT
# Test point for the test dataset and for the llm_results_dir_Test
global_options = llm_pipeline.GlobalOptions(
    dafny_exec= gl.DAFNY_EXEC,
    temp_dir= gl.TEMP_FOLDER,
    llm_results_dir=gl.LLM_RESULTS_DIR,
    assertion_dataset_path=gl.DAFNY_ASSERTION_DATASET
)

llm_cost_stub = llm_configurations.LLM_COST_STUB_RESPONSE_IS_PROMPT("cost")
llm_without_api = llm_configurations.LLM_YIELD_RESULT_WITHOUT_API("without_api")


run_options = llm_pipeline.RunOptions(
    number_assertions_to_test=-1, # if -1 test all assetions
    number_rounds=1, # number of indepedent rounds in each assertion
    number_retries_chain=1, #number of retries it tries to ask fix for previous wrong andwer
    add_error_message=1,
    remove_empty_lines=True,
    change_assertion_per_text="", #Assertion with oracle to retrieve position
    base_prompt=base_prompt,
    localization_base_prompt=localization_base_prompt,
    examples_to_augment_prompt_type="DYNAMIC", # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options    
                                               # DYNAMIC represents my code/error message embeddings
    examples_weight_of_error_message = 0.5, # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
    number_examples_to_add=3,

    # Only used in LLM_EXAMPLE (used DYNAMIC 3 examples 0,5 underneath)
    
    # Fields ignore for all except LLM_EXAMPLE
    examples_to_augment_prompt_type_pos="RANDOM", # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options    
                                               # DYNAMIC represents my code/error message embeddings
    examples_weight_of_error_message_pos = 0.5, # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
    number_examples_to_add_pos=3,



    limit_example_length_bytes = 1200, # Option Not implemented
    verifier_output_filter_warnings = 1, # If 1 only errors are passed in the verifier ouput string
    system_prompt=system_prompt,
    skip_original_verification=1,


    localization = "LLM_EXAMPLE",
    # Options related with if will prompt LLM or gathered from saved results
    skip_verification=1,
    only_verify = 0,
    only_get_location = 0,
    only_get_assert_candidate = 0,
)
#### COST ESTIMATION OF RUNNING WHAT IS ASKED ####
run_options.skip_original_verification=1

print("GET Cost Statistics of: Running all experiment refering to ORACLE position and all retrieval example methods")
# Cost estimation llm to get estimated costs of running one full evaluation
llm_pipeline.evaluate_all(llm_cost_stub, global_options, run_options)
llm_cost_stub.get_cost_statistics("gpt_4.1",2,8,3)
# Ability to see all prompts as they are send to the LLMs
llm_pipeline.evaluate_all(llm_without_api, global_options, run_options)
run_options.skip_original_verification=1


########################################################################################################
########################################################################################################
########################################################################################################

exit()
# COMMENT exit() AND comment llm_without_api evaluate_all on upper lines to run full replicating paper 
# pipeline. Estimated cost around 76 dolars

########################################################################################################
########################################################################################################
########################################################################################################

import os
openai_api_key=os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")
llm_openai = llm_open_ai.OpenAI_LLM("gpt_4.1", model="gpt-4.1-2025-04-14", max_context_size=128000, openaiKey=openai_api_key, verbose=0)
llm_to_use = llm_openai

print("BEST OVERALL")
print("\nA\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


print("\nB\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "LLM_EXAMPLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


print("\nC\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "LAUREL_BETTER"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")

print("BEST POSITION")

print("\nD\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "LAUREL"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


print("\nE\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "LAUREL_BETTER"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


print("\nF\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "LLM"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


print("\nG\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "LLM_EXAMPLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")

print("\nH\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")

print("Best Example Retrieval")

print("\nI\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")

print("\nJ\n")
run_options.examples_to_augment_prompt_type="RANDOM" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")

print("\nL\n")
run_options.examples_to_augment_prompt_type="EMBEDDED" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")

print("\nM\n")
run_options.examples_to_augment_prompt_type="TFIDF" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")

print("\nN\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.25 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


print("\nO\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.50 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


print("\nP\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.75 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


print("\nQ\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 1 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 1
if(part == 1):
    print("Beggin Prompt Localization")
    run_options.skip_verification=1
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 1
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("End Prompt Localization")
part = 2
if(part == 2):
    print("Beggin Prompt Assertion Candidate")
    run_options.skip_verification=1 
    run_options.only_get_assert_candidate = 1
    run_options.only_get_location = 0
    run_options.only_verify = 0
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options)
    print("END Prompt Assertion Candidate")


a = """
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 1 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part =  3

""" 

print("VERIFICATION")
print("\nA\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")



print("\nB\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "LLM_EXAMPLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")


print("\nC\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "LAUREL_BETTER"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")


print("\nD\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "LAUREL"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")


print("\nE\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "LAUREL_BETTER"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")

print("\nF\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "LLM"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")


print("\nG\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "LLM_EXAMPLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")

print("\nH\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")

print("\nI\n")
run_options.examples_to_augment_prompt_type="NONE" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=0
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")



print("\nJ\n")
run_options.examples_to_augment_prompt_type="RANDOM" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")



print("\nL\n")
run_options.examples_to_augment_prompt_type="EMBEDDED" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")



print("\nM\n")
run_options.examples_to_augment_prompt_type="TFIDF" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.5 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")



print("\nN\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.25 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")




print("\nO\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.50 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")




print("\nP\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 0.75 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")




print("\nQ\n")
run_options.examples_to_augment_prompt_type="DYNAMIC" # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options     
run_options.examples_weight_of_error_message = 1 # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
run_options.number_examples_to_add=3
run_options.localization= "ORACLE"
part = 3
if(part == 3):
    print("Beggin Verification")
    ### Only verificaiton is safe to run in parallel !
    #### Three Pass run Only verify 
    run_options.skip_verification=0
    run_options.only_get_assert_candidate = 0
    run_options.only_get_location = 0
    run_options.only_verify = 1
    llm_pipeline.evaluate_all(llm_to_use, global_options, run_options, parallel_run=1)
    print("End Verification")



