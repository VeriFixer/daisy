import utils.global_variables as gl 
import utils.assertion_method_classes as assertion_lib
import dafny.dafny_runner as dafny_runner     
import llm.retrieve_examples as ret
import llm.llm_configurations as con
from llm.parse_raw_response import parse_raw_response
from llm.extract_error_blocks import  extract_error_blocks
from utils.run_parallel_or_seq import run_parallel_or_seq

import ast 

from analysis.get_dataframe_from_results import retrieve_information_from_dataset_mod

from pathlib import Path
import os
import json
import random
import pandas as pd 

from enum import Enum

class LocStrategies(Enum):
    ORACLE = "ORACLE"
    LLM = "LLM"
    LLM_EXAMPLE = "LLM_EXAMPLE"
    LAUREL = "LAUREL"
    LAUREL_BETTER = "LAUREL_BETTER"
    HYBRID = "HYBRID"

class ExampleStrategies(Enum):
    NONE = "NONE"
    RANDOM = "RANDOM"
    TFIDF = "TFIDF"
    EMBEDDED = "EMBEDDED"
    DYNAMIC = "DYNAMIC"

class RunOptions:
    def __init__(self, number_assertions_to_test : int, number_rounds: int, number_retries_chain : int, 
                 add_error_message: bool, skip_verification: bool, remove_empty_lines: bool, 
                 change_assertion_per_text: str, base_prompt: str, localization_base_prompt: str,
                 examples_to_augment_prompt_type: ExampleStrategies, number_examples_to_add : int, limit_example_length_bytes: int, 
                 verifier_output_filter_warnings: bool, system_prompt: str, localization: LocStrategies, 
                 only_verify: bool, only_get_location: bool, only_get_assert_candidate: bool, skip_original_verification: bool,
                 examples_weight_of_error_message:float, examples_to_augment_prompt_type_pos: ExampleStrategies,
                 examples_weight_of_error_message_pos: float,number_examples_to_add_pos : int
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
        if(self.examples_to_augment_prompt_type != ExampleStrategies.NONE or
           self.examples_to_augment_prompt_type_pos != ExampleStrategies.NONE):
            s += f"addExamp_{self.number_examples_to_add}_"
        if(self.examples_to_augment_prompt_type == ExampleStrategies.DYNAMIC or 
           self.examples_to_augment_prompt_type_pos == ExampleStrategies.DYNAMIC):
             s += f"alpha_{self.examples_weight_of_error_message}_"

        s += f"ExType_{self.examples_to_augment_prompt_type}_"
        s += f"loc_{self.localization}" 
        return s

class GlobalOptions:
    def __init__(self, dafny_exec : Path , temp_dir: Path, llm_results_dir: Path, assertion_dataset_path: Path):
        self.dafny_exec = dafny_exec 
        self.temp_dir = temp_dir
        self.llm_results_dir = llm_results_dir
        self.assertion_dataset_path = assertion_dataset_path

def get_localization_prompt(localization_prompt :str, method_missing_assertions: str, original_error: str, run_options: RunOptions, prog_name: str, group_name: str) -> str:
    if(run_options.localization == LocStrategies.LLM_EXAMPLE or run_options.localization == LocStrategies.HYBRID):
      localization_prompt = add_examples_if_required(localization_prompt, method_missing_assertions,  original_error, run_options, prog_name, group_name, localization_prompt = True)
   
    err = f"\n=== TASK === \n Verifier error:\n {original_error}\n Program (numbered):\n"
    numbered_lines = "\n".join(
        f"{line_id}: {line}" for line_id, line in enumerate(method_missing_assertions.splitlines())
    )
    return localization_prompt + err + numbered_lines + "\n OUTPUT: JSON array of line numbers ONLY, e.g. [2,5] (NO OTHER TEXT OR EXPLANATION)"

num_threads = os.cpu_count()  # Number of logical cores

import utils.dataset_class as dat
def process_group(assertion_group : assertion_lib.assertionGroup, model_dir : Path, 
                   assertion_datset_info_df : pd.DataFrame,  llm : con.LLM, assertion_groups : list[assertion_lib.assertionGroup],
                   run_options : RunOptions, global_options : GlobalOptions):
        file = assertion_lib.get_file_from_assertion_group(assertion_group)

        parent_folder_name = os.path.basename(file.file_path.parent.name)
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

def setup_llm_evaluate_all(llm : con.LLM, global_options : GlobalOptions, run_options : RunOptions):
    llm.set_system_prompt(run_options.system_prompt)
    assertion_dataset = dat.Dataset.from_dataset_assertion_groups(global_options.assertion_dataset_path) 
    assertion_groups = assertion_dataset.get_all_assertion_groups()
    print(f"LEN assertion groups {len(assertion_groups)}")
    model_name = llm.get_name() + "_" + run_options.get_one_line_main_options()
    model_dir = global_options.llm_results_dir / model_name
    os.makedirs(model_dir, exist_ok=True)
    assertion_datset_info =  retrieve_information_from_dataset_mod(global_options.assertion_dataset_path)
    assertion_dataset_info_df : pd.DataFrame = pd.DataFrame(assertion_datset_info)

    return (assertion_groups, model_dir, assertion_dataset_info_df)

def evaluate_all(
    llm : con.LLM, global_options : GlobalOptions, run_options : RunOptions, parallel_run : bool = False
):

    (assertion_groups, model_dir, assertion_dataset_info_df) = setup_llm_evaluate_all(llm, global_options, run_options)
   
    llm_name = model_dir.name.split("__")[0]

    return run_parallel_or_seq(
        assertion_groups,
        process_group,
        f"Eval Groups: {llm_name}",
        model_dir,
        assertion_dataset_info_df, 
        llm, 
        assertion_groups, 
        run_options, 
        global_options,
        parallel=parallel_run,
        requires_semaphore=False
    )


def build_system_prompt(run_options : RunOptions, assertion_group : assertion_lib.assertionGroup,
                        all_assertion_groups : list[assertion_lib.assertionGroup]):
    prompt = run_options.base_prompt
    if run_options.localization in {LocStrategies.ORACLE,LocStrategies.LAUREL,LocStrategies.LAUREL_BETTER}:
        prompt += f"Notice that the assertion locations are expected where the text'{run_options.oracle_localization_text}' is.\n"
    return prompt

def prepare_method_for_verification_default_options(assertion_group: assertion_lib.assertionGroup, oracle : bool = True,
                                                    oracle_text: str=gl.ASSERTION_PLACEHOLDER):
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    text_to_substitute = ""
    if(oracle):
        text_to_substitute =  oracle_text
    remove_empty_lines = True
    method_with_assertions = method.get_method_with_assertion_group_changed(assertion_group,remove_empty_lines, text_to_substitute)
    _, _text = file.substitute_method_with_text(method, method_with_assertions)
    return file, method, method_with_assertions


def prepare_method_for_verification(assertion_group : assertion_lib.assertionGroup, run_options : RunOptions):
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    text_to_substitute = ""
    if(run_options.localization == LocStrategies.ORACLE):
        text_to_substitute = run_options.oracle_localization_text
    method_with_assertions = method.get_method_with_assertion_group_changed(assertion_group, run_options.remove_empty_lines, text_to_substitute)
    _, _ = file.substitute_method_with_text(method, method_with_assertions)
    return file, method, method_with_assertions

def verify_initial_state(file: assertion_lib.FileInfo, method: assertion_lib.MethodInfo, method_text : str,
                         run_options : RunOptions, global_options : GlobalOptions):
    _, full_file_text = file.substitute_method_with_text(method, method_text)
    _, stdout_msg, _ = dafny_runner.run_dafny_from_text(global_options.dafny_exec, full_file_text, global_options.temp_dir)
    return stdout_msg, full_file_text

def add_error_message_if_required(base_prompt : str, error_text : str, run_options : RunOptions):
    if run_options.add_error_message:
        base_prompt += "\nERROR:\n"
        if run_options.verifier_output_filter_warnings:
            error_text = extract_error_blocks(error_text)
        base_prompt += error_text + "\n"
    return base_prompt


def add_examples_if_required(base_prompt : str, method_missing_assertions: str,  original_error : str, run_options: RunOptions, prog_name: str, group_name: str, localization_prompt: bool = False):
    if(localization_prompt):
      nex = run_options.number_examples_to_add_pos
      alpha = run_options.examples_weight_of_error_message_pos
      type = run_options.examples_to_augment_prompt_type_pos
    else:
      nex = run_options.number_examples_to_add
      alpha = run_options.examples_weight_of_error_message
      type = run_options.examples_to_augment_prompt_type
    
    
    augmentation_localization = (localization_prompt) and ((type != ExampleStrategies.NONE) and (nex != 0))
    augmentation_examples = (not localization_prompt) and ((type != ExampleStrategies.NONE) and (nex != 0))

    if((not augmentation_localization) and  (not augmentation_examples)):
        return base_prompt
    
    error_txt_filter = extract_error_blocks(original_error)

    #else augment examples
    if(augmentation_localization or (augmentation_examples and type != ExampleStrategies.NONE)):  
        if(type == ExampleStrategies.RANDOM):
            typeEx = "embedding"
        elif (type == ExampleStrategies.TFIDF):
            typeEx = "tfidf"
        elif (type == ExampleStrategies.EMBEDDED):
            typeEx = "embedding"
        elif (type == ExampleStrategies.DYNAMIC):
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
        if(run_options.examples_to_augment_prompt_type == ExampleStrategies.RANDOM):
            # randomize options
            random.shuffle(results)
    else:
        raise ValueError(f"run_options.examples_to_augment_prompt_type == {run_options.examples_to_augment_prompt_type}  NOT supported")
    
    if(augmentation_localization == 1):
        example_prompt = "Consider these examples: \n"
    else:
      example_prompt = "Consider these examples: \n"
    tot = 0
    for rank, r in enumerate(results, 1):
            tot += 1
            if(tot > nex):
                 break
            filtered_error_message = extract_error_blocks(r['error_message'])
            example_prompt += "=== EXAMPLE ===\n"
            example_prompt += f"Error:\n{filtered_error_message}\n"
            if(augmentation_localization == 1):
              numbered_lines = "\n".join(    
                 f"{line_id}: {line}" for line_id, line in enumerate(r['method_without_assertion_group'].splitlines()))     
              example_prompt += f"\nCODE:\n{numbered_lines}\n"
              example_prompt += f"OUTPUT:\n{r['oracle_pos']}\n"
            else:
                example_prompt += f"CODE:\n{r['code_snippet']}\n"
                assertion_list = ast.literal_eval(r["assertions"])
                example_prompt += f"OUTPUT (as this is oracle only one option is shown the one that fixes the problem): \n {[[x] for x in assertion_list]}\n"
            
            example_prompt += "=== END ===\n"
    return base_prompt + example_prompt



  
def get_method_with_assertions_placeholder(method_missing_assertions : str, localization_array : list[int], llm_localization_substitute_text : str) -> str:
        lines = method_missing_assertions.split("\n")
        lines_added_pos: list[str] = []

        for i, line in enumerate(lines):
           lines_added_pos.append(line.rstrip())  # Remove trailing newline (will add later)
           for loc in localization_array: # if model answer 17,17 i shouls insert two missing lines and not one
             if i == loc:
               lines_added_pos.append(llm_localization_substitute_text)

        method_missing_assertions = "\n".join(lines_added_pos)
        return method_missing_assertions


def run_llm_get_localization(method_missing_assertions: str, original_error:str, llm: con.LLM , 
                             run_options: RunOptions, prog_name: str, group_name: str):
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
        method_missing_assertions = get_method_with_assertions_placeholder(method_missing_assertions, localization_array, llm_localization_substitute_text)
        return localization_prompt, localization_raw_response, method_missing_assertions

def save_llm_localization_information(dir : Path, localization_prompt: str, localization_raw_response:str, method_missing_assertions: str):
    files = {
        "localization_prompt.txt" : localization_prompt,
        "localization_raw_response.txt" : localization_raw_response,
        "localization_method_missin_assertions.txt" : method_missing_assertions 
    }
    for filename, content in files.items():
        with open(os.path.join(dir, filename), "w") as f:
            f.write(content)

def get_llm_localization_information(localization_dir : Path):
    with open(os.path.join(localization_dir, "localization_prompt.txt"), "r") as f:
        localization_prompt = f.read()

    with open(os.path.join(localization_dir, "localization_raw_response.txt"), "r") as f:
        localization_raw_response = f.read()

    with open(os.path.join(localization_dir, "localization_method_missin_assertions.txt"), "r") as f:
        method_missing_assertions = f.read()

    return localization_prompt, localization_raw_response, method_missing_assertions

# response_assertions is a list of assertions
def save_llm_assertion_list_information(dir : Path, prompt: str, raw_response: str, response_assertions: list[str], chat_history: list[str]):
    files = {
        "assertions_prompt.txt": prompt,
        "assertions_raw_response.txt": raw_response,
        "assertions_chat_history.json" : chat_history,
        "assertions_parsed.json": response_assertions  # join list into string
    }

    for filename, content in files.items():
        with open(os.path.join(dir, filename), "w") as f:
            if(type(content) == str):
              f.write(content)
            elif(type(content) == list):
              json.dump(content, f)   

from typing import cast
def get_llm_assertion_list_information(dir : Path) -> tuple[str,str,list[str],list[str]]:
    with open(os.path.join(dir, "assertions_prompt.txt"), "r") as f:
        prompt = f.read()
    with open(os.path.join(dir, "assertions_raw_response.txt"), "r") as f:
        raw_response = f.read()
    with open(os.path.join(dir, "assertions_parsed.json"), "r") as f:
        text = f.read()
        response_assertions = cast(list[str],json.loads(text))
    with open(os.path.join(dir, "assertions_chat_history.json"), "r") as f:
        text = f.read()
        chat_history = cast(list[str],json.loads(text))
    return prompt, raw_response, response_assertions, chat_history

def get_base_prompt(system_prompt: str, method_missing_assertions: str, original_error: str, run_options: RunOptions, prog_name : str, group_name: str):
        base_prompt = system_prompt
        base_prompt = add_examples_if_required(base_prompt, method_missing_assertions,  original_error, run_options, prog_name, group_name)
        base_prompt += "\n === TASK === \n"
        base_prompt = add_error_message_if_required(base_prompt, original_error, run_options)
        base_prompt += "\nCODE:\n" + method_missing_assertions
        base_prompt += "\nOUTPUT:\n" + "Enter your response as a JSON array of arrays (containing the assertions to fix the program) ONLY, no extra text. (NO OTHER TEXT OR EXPLANATION)"

        return base_prompt

def run_llm_get_assertions(prompt : str, llm: con.LLM):
    llm.reset_chat_history()
    # For now ignore the retries at all
    raw_response = llm.get_response(prompt)
    try:
        response_assertions = parse_raw_response(raw_response)
    except:
        response_assertions = [["FAILED_RECEIVING_JSON_BAD_FORMATTED_JSON"]]
    chat_history = llm.get_chat_history() 
    return raw_response, response_assertions, chat_history

def save_verification_information(dir : Path, status : dafny_runner.Status , verif_stdout : str, verif_stderr : str,
                                 verif_filter_warnings : str, verif_file : str, assertion : str,
                                 program_with_localization : str, program_with_new_assertions : str):
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
            f.write(str(content))

def get_verification_information(dir : Path):
    with open(os.path.join(dir, "status.txt"), "r") as f:
        status = f.read()

    with open(os.path.join(dir, "verif_stdout.txt"), "r") as f:
        verif_stdout = f.read()

    with open(os.path.join(dir, "verif_stderr.txt"), "r") as f:
        verif_stderr = f.read()

    return status, verif_stdout, verif_stderr

def zip_with_empty_indexed(assertions: list[list[str]]) -> tuple[list[list[str]], list[list[int]]]:
    """
    Returns two lists: 
    1. Values (zipped then individual leftovers padded with "")
    2. Indices (original index of each value, or -1 for padding)
    """
    n = len(assertions)
    if not assertions:
        return [], []

    # 1) Standard zip up to the shortest list
    min_len = min(map(len, assertions))
    
    # Values
    zipped_vals = [list(row) for row in zip(*(lst[:min_len] for lst in assertions))]
    # Indices (all will be the current row index 'i')
    zipped_inds = [[i] * n for i in range(min_len)]

    # 2) Leftover logic
    leftover_vals: list[list[str]] = []
    leftover_inds: list[list[int]] = []

    if n != 1:
        for list_idx, lst in enumerate(assertions):
            for item_idx, val in enumerate(lst):
                # Build the row for values: val at list_idx, "" elsewhere
                v_row = [val if i == list_idx else "" for i in range(n)]
                # Build the row for indices: item_idx at list_idx, -1 elsewhere
                i_row = [item_idx if i == list_idx else -1 for i in range(n)]
                
                leftover_vals.append(v_row)
                leftover_inds.append(i_row)

    return zipped_vals + leftover_vals, zipped_inds + leftover_inds

# Example:
# assertions = [[1, 2], [3, 4], [5, 6]]
# vals, inds = zip_with_empty_indexed(assertions)

# vals = [
#   [1, 3, 5], [2, 4, 6],      # zipped up to min length (2)
#   [1, "", ""], [2, "", ""],  # leftovers from first list
#   ["", 3, ""], ["", 4, ""],  # leftovers from second list
#   ["", "", 5], ("", "", 6]   # leftovers from third list
# ]

# inds = [
#   [0, 0, 0], [1, 1, 1],       # original positions for zipped elements
#   [0, -1, -1], [1, -1, -1],   # positions for first list leftovers
#   [-1, 0, -1], [-1, 1, -1],   # positions for second list leftovers
#   [-1, -1, 0], [-1, -1, 1]    # positions for third list leftovers
# ]

def get_verif_dir(dir : Path, assertion_opt : list[int]):
    assertion_identifier = "_".join([str(x) for x in assertion_opt])
    verif_dir = os.path.join(dir, f"Assertion_id_{assertion_identifier}")
    return verif_dir


def check_if_any_assertion_was_verified(dir : Path, assertions_ziped : list[list[int]]):
    for assertion_pos_opt in assertions_ziped:
        res = check_if_assertion_was_verified(dir, assertion_pos_opt)
        if(res == True):
            return True 
    return False

def check_if_assertion_was_verified(dir : Path, assertion_pos_opt : list[int]):
    verif_dir = get_verif_dir(dir, assertion_pos_opt)
    status_file = os.path.join(verif_dir, "status.txt")
    if os.path.exists(status_file):
        with open(os.path.join(verif_dir, "status.txt"), "r") as f:
            this_status = f.read()
            if this_status == "Status.VERIFIED":
                return True
    return False


def run_verification_on_assertions(dir : Path, assertions : list[list[str]], run_options : RunOptions, global_options : GlobalOptions, 
                                   method_text_with_location : str, file : assertion_lib.FileInfo, method : assertion_lib.MethodInfo):

    # Assertion gatherer 
    assertions_ziped, assertions_pos_ziped = zip_with_empty_indexed(assertions)

    if(check_if_any_assertion_was_verified(dir, assertions_pos_ziped)):
        return True
    #for assertion_to_test_with_pos in zip(assertions_ziped, assertions_pos_ziped):
    #    status = run_verification_on_assertion(assertion_to_test_with_pos, dir,  run_options, global_options, 
    #                               method_text_with_location, file, method)
    #    if status ==  StatusRunVerification.VERIFIED:
    #        return True
    # Now this is ready for run_parallel_or_seq
    method_id = dir.parent.name.split("_")[2]
    prog_id = dir.parent.parent.name[:10]
    llm_name = dir.parent.parent.parent.name.split("__")[0]
    
    assertions_to_test_with_pos = list(zip(assertions_ziped, assertions_pos_ziped))
    res = run_parallel_or_seq(assertions_to_test_with_pos, 
                                                           run_verification_on_assertion, 
                                                           f"Verify {llm_name}:prog:{prog_id}meth:{method_id}",  
                                                           dir, 
                                                           run_options, 
                                                           global_options, 
                                                           method_text_with_location, 
                                                           file, 
                                                           method, 
                                                           parallel=True,
                                                           requires_semaphore=True,
                                                           disable_progress_bar=True
                                                           )
    for el in res:
        if(el == StatusRunVerification.VERIFIED):
            return True
    
    return False

class StatusRunVerification(Enum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    ALREADY_RUN = "ALREADY_RUN"
    BAD_PARSING = "BAD_PARSING"

def run_verification_on_assertion(assertion_to_test_with_pos : tuple[list[str], list[int]], dir : Path, run_options : RunOptions, global_options : GlobalOptions, 
                                   method_text_with_location : str, file : assertion_lib.FileInfo, method : assertion_lib.MethodInfo):
        
        assertion_opt, assertion_pos_opt = assertion_to_test_with_pos

        if(check_if_assertion_was_verified(dir, assertion_pos_opt)):
            return StatusRunVerification.VERIFIED
        
        method_fixed = method_text_with_location
        verif_dir = get_verif_dir(dir, assertion_pos_opt)
        status_file = os.path.join(verif_dir, "status.txt")
        if os.path.exists(status_file):
            return StatusRunVerification.ALREADY_RUN  
        else:
            #print(f"Running {verif_dir}")
            pass

        os.makedirs(verif_dir, exist_ok=True)
        
        try:
          for assertion in assertion_opt:
              method_fixed = method_fixed.replace(run_options.oracle_localization_text, assertion, 1)
        except:
            #Bad parsing in status
            status = dafny_runner.Status.ERROR
            s : str = "BAD_PARSING_ERROR"
            save_verification_information(Path(verif_dir), status, s, s, s, s, str(assertion_opt), s, s)
            return StatusRunVerification.BAD_PARSING

        _, fixed_file = file.substitute_method_with_text(method, method_fixed)
        _, file_only_with_location_file = file.substitute_method_with_text(method, method_text_with_location)

        status, stdout_msg, stderr = dafny_runner.run_dafny_from_text(global_options.dafny_exec, fixed_file, global_options.temp_dir)
        filter_error: str = ""
        if(run_options.verifier_output_filter_warnings):
            filter_error = extract_error_blocks(stdout_msg)
        
        save_verification_information(Path(verif_dir), status, stdout_msg, stderr, filter_error, fixed_file, str(assertion_opt), file_only_with_location_file, fixed_file)
        if status == dafny_runner.Status.VERIFIED:
            return StatusRunVerification.VERIFIED
        else:
            return StatusRunVerification.NOT_VERIFIED

def save_original_error(dir : Path, original_error : str):
    files = {
        "original_error.txt": original_error
    }

    for filename, content in files.items():
        with open(os.path.join(dir, filename), "w") as f:
            f.write(content)

def get_original_error(dir : Path, assertion_group : assertion_lib.assertionGroup):
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

_MODES: dict[LocStrategies, dict[str,str]] = {
    LocStrategies.ORACLE: {
        "prompt":      "No prompt Oracle Used",
        "placeholder": "method_with_assertion_placeholder.dfy",
        "position":    "oracle_fix_position.txt",
        "replace":     "",
    },
    LocStrategies.LAUREL: {
        "prompt":      "No prompt LAUREL Used",
        "placeholder": "laurel_LAURELmethod_with_placeholder_on_position.dfy",
        "position":    "laurel_LAURELassertion_position.txt",
        "replace":     "<assertion> Insert assertion here </assertion>",
    },
    LocStrategies.LAUREL_BETTER: {
        "prompt":      "No prompt LAUREL_BETTER Used",
        "placeholder": "laurel_LAUREL_BETTERmethod_with_placeholder_on_position.dfy",
        "position":    "laurel_LAUREL_BETTERassertion_position.txt",
        "replace":     "<assertion> Insert assertion here </assertion>",
    },
    LocStrategies.HYBRID: {
        "prompt":      "No prompt HYBRID Used",
        "placeholder": "laurel_LAUREL_BETTERmethod_with_placeholder_on_position.dfy",
        "position":    "laurel_LAUREL_BETTERassertion_position.txt",
        "replace":     "<assertion> Insert assertion here </assertion>",
    },
    LocStrategies.LLM: {"None" : "None"}  # handled separately
}

def do_localization(localization_dir: Path, assertion_group : assertion_lib.assertionGroup, run_options : RunOptions):
    mode = run_options.localization
    if(mode in {LocStrategies.LLM,LocStrategies.LLM_EXAMPLE} ): # There is computed
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


def run_llm_fixes(llm : con.LLM, assertion_group : assertion_lib.assertionGroup, all_assertion_groups: list[assertion_lib.assertionGroup],
                 results_dir: Path, run_options: RunOptions, global_options: GlobalOptions):
    
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
          
        if(run_options.localization == LocStrategies.ORACLE or 
           run_options.localization == LocStrategies.LAUREL or 
           run_options.localization == LocStrategies.LAUREL_BETTER): # if change_assertion_per_text_oracle
                localization_prompt, localization_raw_response, method_missing_assertions = do_localization(localization_dir, assertion_group, run_options)
        elif(run_options.localization == LocStrategies.LLM_EXAMPLE or 
             run_options.localization == LocStrategies.LLM):
                localization_prompt, localization_raw_response, method_missing_assertions = run_llm_get_localization(method_missing_assertions, original_error, llm,                                                                                                     run_options, file.file_path.parent.name , assertion_group_name)
        elif(run_options.localization == LocStrategies.HYBRID):
            _, localization_raw_response_static, _ = do_localization(localization_dir, assertion_group, run_options)
            localization_prompt, localization_raw_response_llm, _ = run_llm_get_localization(method_missing_assertions, original_error, llm,                                                                                                     run_options, file.file_path.parent.name , assertion_group_name)
            loc_raw_static_list = ast.literal_eval(localization_raw_response_static)
            loc_raw_llm_list = ast.literal_eval(localization_raw_response_llm)
            # use always laurel better and remove llm predictions if the same as laurel better
            loc_list = loc_raw_static_list + [x for x in loc_raw_llm_list if x not in loc_raw_static_list]
            localization_raw_response = str(loc_list)
            method_missing_assertions = get_method_with_assertions_placeholder(method_missing_assertions, 
                                                                               loc_list, run_options.llm_localization_text)
  
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