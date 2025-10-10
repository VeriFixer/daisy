import utils.global_variables as gl
import utils.assertion_method_classes as assertion_lib
import dafny.dafny_runner as dafny_runner     
import llm.retrieve_examples as ret
from llm.parse_raw_response import parse_raw_response
from analysis.get_dataframe_from_results import retrieve_information_from_dataset_mod
from llm.extract_error_blocks import  extract_error_blocks

import os
import json
import random
import re
import pandas as pd 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_localization_prompt(localization_prompt, method_missing_assertions, original_error, run_options, prog_name, group_name):
    if(run_options.localization == "LLM_EXAMPLE"):
      localization_prompt = add_examples_if_required(localization_prompt, method_missing_assertions,  original_error, run_options, prog_name, group_name, localization_prompt = 1)
   
    err = f"The error got from the verifier is:\n {original_error}\n"
    numbered_lines = "\n".join(
        f"{line_id}: {line}" for line_id, line in enumerate(method_missing_assertions.splitlines())
    )
    return localization_prompt + err + numbered_lines

num_threads = os.cpu_count()  # Number of logical cores
def evaluate_all(
    llm, global_options, run_options, parallel_run=0
):
    llm.set_system_prompt(run_options.system_prompt)
    assertion_dataset = assertion_lib.Dataset(global_options.assertion_dataset_path) 
    assertion_groups = assertion_dataset.get_all_assertion_groups()
    print(f"LEN assertion groups {len(assertion_groups)}")
    model_name = llm.get_name() + "_" + run_options.get_one_line_main_options()
    model_dir = global_options.llm_results_dir / model_name
    os.makedirs(model_dir, exist_ok=True)

    # Read dataframe 
    assertion_datset_info =  retrieve_information_from_dataset_mod(global_options.assertion_dataset_path)
    assertion_datset_info_df = pd.DataFrame(assertion_datset_info)



    sumi = 0
    def process_group(assertion_group):
        file = assertion_lib.get_file_from_assertion_group(assertion_group)
        parent_folder_name = os.path.basename(file.file_parent_folder_path)
        file_dir = model_dir / parent_folder_name
        os.makedirs(file_dir, exist_ok=True)

        assertion_group_id_str = assertion_lib.get_assertion_group_string_id(assertion_group)
        results_dir = file_dir / assertion_group_id_str
        os.makedirs(results_dir, exist_ok=True)

        # Only process group if it is not of the type "w/o-2 one w/o-1"
        match = assertion_datset_info_df[
        (assertion_datset_info_df["prog"] == parent_folder_name) &
        (assertion_datset_info_df["group"] == assertion_group_id_str)
         ]

        # If a matching row exists and benchmark is "w/o-2 one w/o-1", skip processing
        #if not match.empty and (match["benchmark"].iloc[0] in ["w/o-2 one w/o-1"]):


        if not match.empty and (match["benchmark"].iloc[0] in ["w/o-2 one w/o-1"]):
            #print(f"Skipping {parent_folder_name}/{assertion_group_id_str} due to benchmark 'w/o-2 one w/o-1'")
            return "SKIP"
    
        results = run_llm_fixes(
            llm, assertion_group, assertion_groups, results_dir, run_options, global_options
        )
        return results
    
    if(parallel_run == 0): 
      total = run_options.number_assertions_to_test if run_options.number_assertions_to_test != -1 else len(assertion_groups)
      for number_tested, assertion_group in enumerate(
        tqdm(assertion_groups, desc="Processing assertion groups", total=total), start=0
      ):
        if run_options.number_assertions_to_test != -1 and number_tested >= run_options.number_assertions_to_test:
            return None  # Skip this group
        process_group(assertion_group)
    else:
        PHYSICAL_CORES = os.cpu_count() // 2  # Logical cores usually = 2x physical
        SAFE_THREADS = max(1, PHYSICAL_CORES) 
        random.shuffle(assertion_groups)
        l = len(assertion_groups)
        with ThreadPoolExecutor(max_workers=SAFE_THREADS) as executor:
            futures = {executor.submit(process_group, ag): ag for ag in assertion_groups}
            for future in tqdm(as_completed(futures), total=l, desc="Verifying"):
                future.result()

class RunOptions:
    def __init__(self, number_assertions_to_test, number_rounds, number_retries_chain, add_error_message, skip_verification, remove_empty_lines, 
                 change_assertion_per_text, base_prompt, localization_base_prompt, examples_to_augment_prompt_type, number_examples_to_add, limit_example_length_bytes, 
                 verifier_output_filter_warnings, system_prompt, localization, only_verify, only_get_location, only_get_assert_candidate, skip_original_verification,
                 examples_weight_of_error_message, examples_to_augment_prompt_type_pos, examples_weight_of_error_message_pos, number_examples_to_add_pos
):
        self.number_assertions_to_test = number_assertions_to_test
        self.number_rounds = number_rounds
        self.number_retries_chain = number_retries_chain
        self.add_error_message = add_error_message
        self.skip_verification = skip_verification
        self.remove_empty_lines = remove_empty_lines
        self.change_assertion_per_text = change_assertion_per_text
        self.base_prompt = base_prompt
        self.localization_base_prompt = localization_base_prompt
        

        self.examples_to_augment_prompt_type= examples_to_augment_prompt_type 
        self.examples_weight_of_error_message = examples_weight_of_error_message
        self.number_examples_to_add =  number_examples_to_add
        self.limit_example_length_bytes = limit_example_length_bytes
        self.verifier_output_filter_warnings = verifier_output_filter_warnings
        self.system_prompt = system_prompt

        self.only_verify = only_verify 
        self.only_get_location = only_get_location 
        self.only_get_assert_candidate = only_get_assert_candidate

        # VALS LLM/ ORACLE/ LAUREL
        self.localization = localization  
        self.llm_localization_text =  gl.ASSERTION_PLACEHOLDER
        self.oracle_localization_text =  gl.ASSERTION_PLACEHOLDER

        self.skip_original_verification = skip_original_verification # Masures get wrong as does not introduce error in prompt but it is ok

        self.examples_to_augment_prompt_type_pos = examples_to_augment_prompt_type_pos
        self.examples_weight_of_error_message_pos = examples_weight_of_error_message_pos
        self.number_examples_to_add_pos=number_examples_to_add_pos

    def get_one_line_main_options(self):
        s = "_"
        s += f'nAssertions_{"ALL" if self.number_assertions_to_test == -1 else self.number_assertions_to_test}_'
        s += f"nRounds_{self.number_rounds}_"
        s += f"nRetries_{self.number_retries_chain}_"
        s += f"addError_{self.add_error_message}_"
        s += f"addExamp_{self.number_examples_to_add}_"
        if(self.examples_to_augment_prompt_type=="DYNAMIC"):
             s += f"alpha_{self.examples_weight_of_error_message}_"
        s += f"ExType_{self.examples_to_augment_prompt_type}_"
        s += f"loc_{self.localization}" 
        return s

class GlobalOptions:
    def __init__(self, dafny_exec, temp_dir, llm_results_dir, assertion_dataset_path):
        self.dafny_exec = dafny_exec 
        self.temp_dir = temp_dir
        self.llm_results_dir = llm_results_dir
        self.assertion_dataset_path = assertion_dataset_path

def build_system_prompt(run_options, assertion_group, all_assertion_groups):
    prompt = run_options.base_prompt
    if run_options.add_error_message:
        prompt += "The verifier error message is located inside the #### START ERROR MESSAGE #### block.\n"
    if run_options.localization == "ORACLE" or run_options.localization == "LAUREL" or  run_options.localization == "LAUREL_BETTER":
        prompt += f"Notice that the assertion locations are expected where the text'{run_options.oracle_localization_text}' is.\n"
    return prompt

def prepare_method_for_verification_default_options(assertion_group, oracle=1, oracle_text=gl.ASSERTION_PLACEHOLDER):
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    text_to_substitute = ""
    if(oracle):
        text_to_substitute =  oracle_text
    remove_empty_lines = 1
    method_with_assertions = method.get_method_with_assertion_group_changed(assertion_group,remove_empty_lines, text_to_substitute)
    _, file_text = file.substitute_method_with_text(method, method_with_assertions)
    return file, method, method_with_assertions


def prepare_method_for_verification(assertion_group, run_options):
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    text_to_substitute = ""
    if(run_options.localization == "ORACLE"):
        text_to_substitute = run_options.oracle_localization_text
    method_with_assertions = method.get_method_with_assertion_group_changed(assertion_group, run_options.remove_empty_lines, text_to_substitute)
    _, file_text = file.substitute_method_with_text(method, method_with_assertions)
    return file, method, method_with_assertions

def verify_initial_state(file, method, method_text, run_options, global_options):
    _, full_file_text = file.substitute_method_with_text(method, method_text)
    status, stdout_msg, _ = dafny_runner.run_dafny_from_text(global_options.dafny_exec, full_file_text, global_options.temp_dir)
    return stdout_msg, full_file_text

def add_error_message_if_required(base_prompt, error_text, run_options):
    if run_options.add_error_message:
        base_prompt += "\n #### START ERROR MESSAGE #### \n"
        if run_options.verifier_output_filter_warnings:
            error_text = extract_error_blocks(error_text)
        base_prompt += error_text + "\n #### END ERROR MESSAGE #### \n"
    return base_prompt


def add_examples_if_required(base_prompt, method_missing_assertions,  original_error, run_options, prog_name, group_name, localization_prompt = 0):
    if(localization_prompt):
      nex = run_options.number_examples_to_add_pos
      alpha = run_options.examples_weight_of_error_message_pos
      type = run_options.examples_to_augment_prompt_type_pos
    else:
      nex = run_options.number_examples_to_add
      alpha = run_options.examples_weight_of_error_message
      type = run_options.examples_to_augment_prompt_type
    
    
    augmentation_localization = (localization_prompt == 1) and ((type != "NONE") and (nex != 0))
    augmentation_examples = (localization_prompt == 0) and ((type != "NONE") and (nex != 0))

    if((not augmentation_localization) and  (not augmentation_examples)):
        return base_prompt
    
    error_txt_filter = extract_error_blocks(original_error)

    #else augment examples
    if(augmentation_localization or (augmentation_examples and type != "NONE")):  
        if(type == "RANDOM"):
            typeEx = "embedding"
        elif (type == "TFIDF"):
            typeEx = "tfidf"
        elif (type == "EMBEDDED"):
            typeEx = "embedding"
        elif (type == "DYNAMIC"):
            typeEx = "error_code"
        else:
             raise ValueError(f"run_options.examples_to_augment_prompt_type == {type}  NOT supported")
    
        entries, model, device, tfidf_vectorizer, tfidf_matrix = ret.generate_example_model()
        results = ret.retrieve_by_error_and_code(
            error_txt_filter , 
            method_missing_assertions, 
            entries, 
            top_k=-1, 
            method=typeEx,
            α=alpha,  
            prog_original=prog_name, 
            group_original=group_name,
            model=model,
            device=device, 
            diferent_methods=1,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix
        )
        if(run_options.examples_to_augment_prompt_type == "RANDOM"):
            # randomize options
            random.shuffle(results)
    else:
        raise ValueError(f"run_options.examples_to_augment_prompt_type == {run_options.examples_to_augment_prompt_type}  NOT supported")
    
   
        
    idx = 0
    if(augmentation_localization == 1):
        example_prompt = "Consider these examples: \n The assertion placeholder is already inserted in the oracle fix position \n"
    else:
      example_prompt = "Consider these examples: \n"
    tot = 0
    for rank, r in enumerate(results, 1):
            tot += 1
            if(tot > nex):
                 break
            filtered_error_message = extract_error_blocks(r['error_message'])
            example_prompt += f"Augmenting Prog {prog_name} Group {group_name}:\n"
            example_prompt += f"Prog {r["prog"]} Group {r["group"]}:\n"
            example_prompt += f"Example {rank} Error:\n{filtered_error_message}\n"
            if(augmentation_localization == 1):
              numbered_lines = "\n".join(
                 f"{line_id}: {line}" for line_id, line in enumerate(r['code_snippet'].splitlines()))     
              example_prompt += numbered_lines
              example_prompt += f"\n Oracle Fix position {r["oracle_pos"]} for Assertions:\n{r["assertions"]}\n"
            else:
                example_prompt += f"Code:\n{r['code_snippet']}\n"
                example_prompt += f"Assertions:\n{r["assertions"]}\n"
    return base_prompt + example_prompt



  


def run_llm_get_localization(method_missing_assertions, original_error, llm, run_options, prog_name, group_name):
        llm_localization_substitute_text = run_options.llm_localization_text
        localization_base_prompt = run_options.localization_base_prompt

        localization_prompt = get_localization_prompt(localization_base_prompt, method_missing_assertions, original_error, run_options, prog_name, group_name)
        llm.reset_chat_history()
        localization_raw_response = llm.get_response(localization_prompt)
        try:
            localization_array = parse_raw_response(localization_raw_response)
        except:
            print(f"failed parsing localizaiton raw reponse: {localization_raw_response}")
            localization_array =  [0]
        lines = method_missing_assertions.split("\n")
        lines_added_pos = []

        for i, line in enumerate(lines):
           lines_added_pos.append(line.rstrip())  # Remove trailing newline (will add later)
           for loc in localization_array: # if model answer 17,17 i shouls insert two missing lines and not one
             if i == loc:
               lines_added_pos.append(llm_localization_substitute_text)

        method_missing_assertions = "\n".join(lines_added_pos)
        return localization_prompt, localization_raw_response, method_missing_assertions

def save_llm_localization_information(dir, localization_prompt, localization_raw_response, method_missing_assertions):
    files = {
        "localization_prompt.txt" : localization_prompt,
        "localization_raw_response.txt" : localization_raw_response,
        "localization_method_missin_assertions.txt" : method_missing_assertions 
    }
    for filename, content in files.items():
        with open(os.path.join(dir, filename), "w") as f:
            f.write(content)

def get_llm_localization_information(localization_dir):
    with open(os.path.join(localization_dir, "localization_prompt.txt"), "r") as f:
        localization_prompt = f.read()

    with open(os.path.join(localization_dir, "localization_raw_response.txt"), "r") as f:
        localization_raw_response = f.read()

    with open(os.path.join(localization_dir, "localization_method_missin_assertions.txt"), "r") as f:
        method_missing_assertions = f.read()

    return localization_prompt, localization_raw_response, method_missing_assertions

# response_assertions is a list of assertions
def save_llm_assertion_list_information(dir, prompt, raw_response, response_assertions, chat_history):
    files = {
        "assertions_prompt.txt": prompt,
        "assertions_raw_response.txt": raw_response,
        "assertions_chat_history.txt" : str(chat_history),
        "assertions_parsed.json": response_assertions  # join list into string
    }

    for filename, content in files.items():
        with open(os.path.join(dir, filename), "w") as f:
            if filename.endswith(".json"):
              json.dump(content, f)   
            else:
              f.write(content)

def get_llm_assertion_list_information(dir):
    with open(os.path.join(dir, "assertions_prompt.txt"), "r") as f:
        prompt = f.read()
    with open(os.path.join(dir, "assertions_raw_response.txt"), "r") as f:
        raw_response = f.read()
    with open(os.path.join(dir, "assertions_parsed.json"), "r") as f:
        text = f.read()
        response_assertions =  json.loads(text)
    with open(os.path.join(dir, "assertions_chat_history.txt"), "r") as f:
        chat_history = f.read()
    return prompt, raw_response, response_assertions, chat_history

def get_base_prompt(system_prompt, method_missing_assertions, original_error, run_options, prog_name, group_name):
        base_prompt = system_prompt
        base_prompt = add_examples_if_required(base_prompt, method_missing_assertions,  original_error, run_options, prog_name, group_name)
        base_prompt += "\n You must correct this: \n" + method_missing_assertions
        base_prompt = add_error_message_if_required(base_prompt, original_error, run_options)
        return base_prompt

def run_llm_get_assertions(prompt, llm):
    llm.reset_chat_history()
    # For now ignore the retries at all
    raw_response = llm.get_response(prompt)
    try:
        response_assertions = parse_raw_response(raw_response)
    except:
        response_assertions = [["FAILED_RECEIVING_JSON_BAD_FORMATTED_JSON"]]
    chat_history = llm.get_chat_history() 
    return raw_response, response_assertions, chat_history

def save_verification_information(dir, status, verif_stdout, verif_stderr, verif_filter_warnings, verif_file, assertion, program_with_localization, program_with_new_assertions):
    files = {
        "status.txt": status,
        "verif_stdout.txt": verif_stdout,
        "verif_stdout_filter_warnings.txt": verif_filter_warnings,
        "verif_stderr.txt" : verif_stderr,
        "verif_file.txt" : verif_file,
        "verif_assertion.txt" : assertion,
        "program_with_localization.txt" : program_with_localization,
        "program_with_new_assertions.txt" : program_with_new_assertions
    }

    for filename, content in files.items():
        with open(os.path.join(dir, filename), "w") as f:
            f.write(content)

def get_verification_information(dir):
    with open(os.path.join(dir, "status.txt"), "r") as f:
        status = f.read()

    with open(os.path.join(dir, "verif_stdout.txt"), "r") as f:
        verif_stdout = f.read()

    with open(os.path.join(dir, "verif_stderr.txt"), "r") as f:
        verif_stderr = f.read()

    return status, verif_stdout, verif_stderr

def zip_with_empty(assertions):
    # 1) standard zip up to the shortest list
    min_len = min(map(len, assertions))
    zipped = list(zip(*(lst[:min_len] for lst in assertions)))

    # 2) leftover tuples for each element in each list
    leftovers = []
    n = len(assertions)
    if(n!= 1):
      for idx, lst in enumerate(assertions):
        for x in lst:
            # build a tuple with x in position idx, "" elsewhere
            t = tuple(x if i == idx else "" for i in range(n))
            leftovers.append(t)

    return zipped + leftovers
# Example:
#assertions = [[1, 2], [3, 4], [5,6]]
#print(zip_with_empty(assertions))
# [
#   (1, 3, 5  ), (2, 4, 6  ),     # zipped up to length 1 (min length)
#   (1, "", ""), (2, "", ""), # leftovers from first list
#   ("", 3, ""), ("", 4, ""), # leftovers from second list
#   ("", "", 5), ("", "",6 )  # leftovers from third list
# ]

def run_verification_on_assertions(dir, assertions, run_options, global_options, method_text_with_location, file, method):
    # Early exit if something was verified
    assertions_ziped = zip_with_empty(assertions)
    for idx, assertion_opt in enumerate(assertions_ziped):
        method_fixed = method_text_with_location
        verif_dir = os.path.join(dir, f"Assertion_id_{idx}")
        status_file = os.path.join(verif_dir, "status.txt")
        if os.path.exists(status_file):
            with open(os.path.join(verif_dir, "status.txt"), "r") as f:
                this_status = f.read()
                if this_status == "VERIFIED":
                    return True
            continue

    # For the rest perform verification
    for idx, assertion_opt in enumerate(assertions_ziped):
        method_fixed = method_text_with_location
        verif_dir = os.path.join(dir, f"Assertion_id_{idx}")
        status_file = os.path.join(verif_dir, "status.txt")
        if os.path.exists(status_file):
            continue   
        else:
            print(f"Running {verif_dir}")

        os.makedirs(verif_dir, exist_ok=True)
        
        try:
          for assertion in assertion_opt:
              method_fixed = method_fixed.replace(run_options.oracle_localization_text, assertion, 1)
        except:
            #Bad parsing in status
            status = "BAD_PARSING_ASSERTIONS"
            save_verification_information(verif_dir, status, status, status, status, status, str(assertion_opt), status, status),
            continue

        _, fixed_file = file.substitute_method_with_text(method, method_fixed)
        _, file_only_with_location_file = file.substitute_method_with_text(method, method_text_with_location)

        status, stdout_msg, stderr = dafny_runner.run_dafny_from_text(global_options.dafny_exec, fixed_file, global_options.temp_dir)
        if(run_options.verifier_output_filter_warnings):
            filter_error = extract_error_blocks(stdout_msg)
        
        save_verification_information(verif_dir, status, stdout_msg, stderr, filter_error, fixed_file, str(assertion_opt), file_only_with_location_file, fixed_file),

        if status == "VERIFIED":
            return True
    return False

def save_original_error(dir, original_error):
    files = {
        "original_error.txt": original_error
    }

    for filename, content in files.items():
        with open(os.path.join(dir, filename), "w") as f:
            f.write(content)

def get_original_error(dir, assertion_group):
    try:
        # see in results folder, if not there it still needs to be copied (expect branch)
        with open(os.path.join(dir, "original_error.txt"), "r") as f:
            original_error = f.read()
    except:
          file = assertion_lib.get_file_from_assertion_group(assertion_group)
          assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
          folder = file.file_path.parent
          original_error_file = folder / assertion_group_name / "verifier_output.txt"

          with open(original_error_file, "r", encoding="utf-8") as f:
              original_error = f.read()

          save_original_error(dir, original_error)
    return original_error

_MODES = {
    "ORACLE": {
        "prompt":      "No prompt Oracle Used",
        "placeholder": "method_with_assertion_placeholder.dfy",
        "position":    "oracle_fix_position.txt",
        "replace":     "",
    },
    "LAUREL": {
        "prompt":      "No prompt LAUREL Used",
        "placeholder": "laurel_method_with_placeholder_on_position.dfy",
        "position":    "laurel_assertion_position.txt",
        "replace":     "<assertion> Insert assertion here </assertion>",
    },
    "LAUREL_BETTER": {
        "prompt":      "No prompt LAUREL_BETTER Used",
        "placeholder": "laurel_BETTERmethod_with_placeholder_on_position.dfy",
        "position":    "laurel_BETTERassertion_position.txt",
        "replace":     "<assertion> Insert assertion here </assertion>",
    },
    "LLM": None  # handled separately
}

def do_localization(localization_dir, assertion_group, run_options):
    mode = run_options.localization
    if(mode == "LLM"): # There is computed
        raise ValueError("ERROR this should only be used for when localization is loaded from dataset")
    else: # There are loaded ORACLE/LAUREL/LAUREL_BETTER
        mode_inf = _MODES[mode]
        assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
        file = assertion_lib.get_file_from_assertion_group(assertion_group)
        file_folder = file.file_path.parent
        base_path = file_folder / assertion_group_name
        positions = base_path / mode_inf["position"]
        with open(positions , "r", encoding="utf-8") as f:
            localization_raw_response =  f.read( )
            
        placeholder = base_path / mode_inf["placeholder"]
        with open(placeholder , "r", encoding="utf-8") as f:
            method_missing_assertions =  f.read( )
        if(mode_inf["replace"] != ""):
            method_missing_assertions  = method_missing_assertions .replace(mode_inf["replace"], run_options.llm_localization_text)
        localization_prompt =  mode_inf["prompt"]
    return  localization_prompt, localization_raw_response, method_missing_assertions


def run_llm_fixes(llm, assertion_group, all_assertion_groups, results_dir, run_options, global_options):
    assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
    system_prompt = build_system_prompt(run_options, assertion_group, all_assertion_groups)
    file, method, method_missing_assertions = prepare_method_for_verification(assertion_group, run_options)

    original_error = get_original_error(results_dir, assertion_group)
   
    if(run_options.verifier_output_filter_warnings):
        original_error = extract_error_blocks(original_error)

    only_run_some = run_options.only_get_location or run_options.only_get_assert_candidate or run_options.only_verify
    only_run_num = run_options.only_get_location + run_options.only_get_assert_candidate + run_options.only_verify
    if(only_run_num > 1):
        raise ValueError("CONFIGURATION ERROR run_options only parameter can only have one only active")

    
    localization_dir = results_dir / "localization"
    if(only_run_some and not run_options.only_get_location): # If we are running the other fases we can load the localizaiton file directly
        localization_prompt, localization_raw_response, method_missing_assertions  = get_llm_localization_information(localization_dir) 
    else:
      if(not os.path.exists(localization_dir)): #localization already run no need to rerun   
          
        if(run_options.localization == "ORACLE" or run_options.localization == "LAUREL" or run_options.localization == "LAUREL_BETTER"): # if change_assertion_per_text_oracle
            localization_prompt, localization_raw_response, method_missing_assertions = do_localization(localization_dir, assertion_group, run_options)
        elif(run_options.localization == "LLM_EXAMPLE" or run_options.localization == "LLM"):
            localization_prompt, localization_raw_response, method_missing_assertions = run_llm_get_localization(method_missing_assertions, original_error, llm,                                                                                                     run_options, file.file_path.parent.name , assertion_group_name)
        else:
            raise ValueError("Provided localization option is not supported")
        os.makedirs(localization_dir, exist_ok=True)
        save_llm_localization_information(localization_dir, localization_prompt, localization_raw_response, method_missing_assertions)
      else:
          localization_prompt, localization_raw_response, method_missing_assertions  = get_llm_localization_information(localization_dir) 
          #print("Skipped Loc")
    if(run_options.only_get_location):
          return True 
    
    assertions_list_dir = results_dir / "assertions_list"
    if(only_run_some and not run_options.only_get_assert_candidate):
        prompt, raw_response, response_assertions, chat_history = get_llm_assertion_list_information(assertions_list_dir)
    else: # Only perform LLM prompting (lets kill retry)
      if(not os.path.exists(assertions_list_dir )): #localization already run no need to rerun 
        program_name = file.file_path.parent.name
        prompt = get_base_prompt(system_prompt, method_missing_assertions, original_error, run_options, program_name, assertion_group_name)
        raw_response, response_assertions, chat_history = run_llm_get_assertions(prompt, llm)
        os.makedirs(assertions_list_dir, exist_ok=True)
        save_llm_assertion_list_information(assertions_list_dir, prompt, raw_response, response_assertions, chat_history)
      else:
         #print("Skipped Assert")
         prompt, raw_response, response_assertions, chat_history = get_llm_assertion_list_information(assertions_list_dir)
    if(run_options.only_get_assert_candidate):
        return True

    verification_list_dir = results_dir / "verification"
    if(not(only_run_some and not run_options.only_verify)):
      if(not(run_options.skip_verification)):
        return run_verification_on_assertions(verification_list_dir, response_assertions, run_options, global_options, method_missing_assertions, file, method) 

    return False