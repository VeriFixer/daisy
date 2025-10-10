def get_file_content(file,readType="r"):
    try :
      with open(file, readType) as f:
        file = f.read()
    except:
        print(f"Could not open the file {file}")
        exit()
    return file

if __name__ == "__main__":
    import sys 
    import time
    import tempfile
    import dafny.dafny_runner as dafny_runner
    from llm.extract_error_blocks import extract_error_blocks
    import utils.global_variables as global_variables
    argv = sys.argv
    if(len(argv) not in [5,7]):
        print("usage should be:")
        print("python single_file_run.py <path_to_code> <path_to_error> <path_to_method_code> <position_or_infer>")
        print("python single_file_run.py <path_to_code> <path_to_error> <path_to_method_code> infer <byte_method_start> <byte_method_end>")
        exit()


    file_code = argv[1]
    file_error = argv[2]
    file_method = argv[3]
    pos_or_infer = argv[4]

    if(pos_or_infer == "infer"):
        if(len(argv) != 7):
          print("python single_file_run.py <path_to_code> <path_to_error> <path_to_method_code> infer <byte_method_start> <byte_method_end>")
          exit()


    code_binary = get_file_content(file_code, "rb")
    code = get_file_content(file_code)
    error = get_file_content(file_error)
    method = get_file_content(file_method)

    if(not pos_or_infer in ["pos", "infer"]):
        print("position_or_infer should have values equal to pos or infer,examples: ")
        print("python single_file_run.py <path_to_code> <path_to_error> <path_to_method_code> <position_or_infer>")
        print("python single_file_run.py <path_to_code> <path_to_error> <path_to_method_code> pos")
        print("python single_file_run.py <path_to_code> <path_to_error> <path_to_method_code> infer")
        exit()

    import llm.llm_open_ai as llm_open_ai
    import llm.llm_pipeline as llm_pipeline
    import llm.llm_configurations as llm_configurations
    import utils.global_variables as gl

    import os
    from pprint import pprint

    run_options = llm_pipeline.RunOptions(
        number_assertions_to_test=-1, # if -1 test all assetions
        number_rounds=1, # number of indepedent rounds in each assertion
        number_retries_chain=1, #number of retries it tries to ask fix for previous wrong andwer
        add_error_message=1,
        remove_empty_lines=True,
        change_assertion_per_text="", #Assertion with oracle to retrieve position
        base_prompt=  gl.BASE_PROMPT,
        localization_base_prompt=gl.LOCALIZATION_BASE_PROMPT,
        examples_to_augment_prompt_type="DYNAMIC", # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options    
                                                # DYNAMIC represents my code/error message embeddings
        examples_weight_of_error_message = 0.5, # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
        number_examples_to_add=3,

        # Only used in LLM_EXAMPLE (used DYNAMIC 3 examples 0,5 underneath)
        # Fields ignore for all except LLM_EXAMPLE
        examples_to_augment_prompt_type_pos="DYNAMIC", # "RANDOM", "DYNAMIC", "NONE", "EMBEDDED" "TFIDF" all options    
                                                # DYNAMIC represents my code/error message embeddings
        examples_weight_of_error_message_pos = 0.5, # Wieght of the erro message in case of DYNAMIC in relantion with code                                 
        number_examples_to_add_pos=3,
        limit_example_length_bytes = 1200, # Option Not implemented
        verifier_output_filter_warnings = 1, # If 1 only errors are passed in the verifier ouput string
        system_prompt=gl.SYSTEM_PROMPT,
        skip_original_verification=1,

        localization = "LLM_EXAMPLE",
        # Options related with if will prompt LLM or gathered from saved results
        skip_verification=1,
        only_verify = 0,
        only_get_location = 0,
        only_get_assert_candidate = 0,
    )
 
    #run_llm_get_localization(method_missing_assertions, original_error, llm, 
    ##  run_options, file.file_path.parent.name , assertion_group_name)

    # Not used these fields are used to prevent examples from equal files (not the case in lauching for single file)
    program_name = ""
    assertion_group_name = ""
    if(pos_or_infer  == "pos"):
      
      #openai_api_key=os.getenv("OPENAI_API_KEY")
      #if openai_api_key is None:
      #  raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")
      #llm = llm_open_ai.OpenAI_LLM("gpt_4.1", model="gpt-4.1-2025-04-14", max_context_size=128000, openaiKey=openai_api_key, verbose=0)
      # temp untill everything connected
      print("[20,30]")
      exit()
      res = llm_pipeline.run_llm_get_localization(method, error, llm, run_options, program_name , assertion_group_name ) 
      localization_prompt, localization_raw_response, method_missing_assertions = res 
      print(localization_raw_response)
      exit()
    
    if(pos_or_infer == "infer"):
      from llm.llm_pipeline import zip_with_empty
      # Connected now i have to find the assertion that fizes it
      #print("""["assert 1==1;"]""")
      #exit()
      #prompt = llm_pipeline.get_base_prompt(gl.SYSTEM_PROMPT, method_missing_assertions, error, run_options, program_name, assertion_group_name)
      #res= llm_pipeline.run_llm_get_assertions(prompt, llm)
      #raw_response, response_assertions, chat_history = res

      response_assertions = [["assert data[data.Length -n1 +n] == s1[n1-1-n];"]]

      assertions_ziped = zip_with_empty(response_assertions)
      #print(f"ziped assertions: {assertions_ziped}")

      method_start_byte = int(argv[5])
      method_end_byte = int(argv[6])
      

      for assertion_opt in assertions_ziped:
        method_fixed = method[:]
        assertions_list = []
        for assertion in assertion_opt:
              assertions_list.append(assertion)
              method_fixed = method_fixed.replace(gl.ASSERTION_PLACEHOLDER , assertion, 1)

        #print(method_fixed)

        method_fixed_binary = method_fixed.encode("utf-8")
        code_binary_new = code_binary[:method_start_byte] +  method_fixed_binary + code_binary[method_end_byte+1:]
        code_new = code_binary_new.decode("utf-8")

        #print(code_new)


        status, stdout_msg, stderr = dafny_runner.run_dafny_from_text(gl.DAFNY_EXEC, code_new, gl.TEMP_FOLDER)
        if(run_options.verifier_output_filter_warnings):
            filter_error = extract_error_blocks(stdout_msg)
        
        import json
        if(status == "VERIFIED"):
           print(json.dumps(assertions_list))
    
  
