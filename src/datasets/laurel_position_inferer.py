import sys
import os
import utils.global_variables as gl
import utils.assertion_method_classes as assertion_lib
import dafny.dafny_runner as dafny_runner
import llm.llm_pipeline as llm_pipe

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the laurel directory to sys.path (to call modified wrapper to call C# script)
sys.path.append(os.path.abspath(gl.PATH_TO_LAUREL))
from placeholder_wrapper import call_placeholder_finder

def check_laurel_errors(assertion_group, type=""):
    str_id = assertion_lib.get_assertion_group_string_id(assertion_group)
    assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    file_folder = file.file_path.parent
    base_path = file_folder / assertion_group_name 

    prefix = "laurel_" + type + "error.txt"

    laurel_error_file = base_path / prefix
    
    error = ""
    with open(laurel_error_file , "r", encoding="utf-8") as f:
        error = f.read()

    if("Error in call_placeholder_finder" in error or "Unexpected error" in error):
        print(str_id)
        print(error)
        print("###########################################")
        return 1
    return 0



def run_laurel_fix_position_finder_and_create_laurel_files(assertion_group, type=""):
    str_id = assertion_lib.get_assertion_group_string_id(assertion_group)
    
    assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    file_folder = file.file_path.parent
    base_path = file_folder / assertion_group_name
    
    # program without assertion group
    program_without_assertion_group_file = base_path / "program_without_assertion_group.dfy"
    method_with_missing_assertions = ""
    # Error message
    error_message = ""
    error_message_file = base_path / "verifier_output.txt"
    with open(error_message_file, "r", encoding="utf-8") as f:
        error_message = f.read()
    # Method name (the method name on the program source corresponds to the last elemnt)
    # In the ast we have: _module._default.gaussian instead of gaussian
    method_name = method.method_name.split(".")[-1]

    error_message = llm_pipe.extract_error_blocks(error_message)
    
    if(type == ""):
         use_laurel_better=False
    if(type == "BETTER"):
         use_laurel_better=True

    method_with_placeholder, error = call_placeholder_finder(
      error_message,
      program_without_assertion_group_file,
      method_name,
      use_laurel_better
    )

    # Get laurel fix positions
    laurel_infered_position_missing_assertions = []
    laurel_assertion_placeholder = "<assertion> Insert assertion here </assertion>"
    method_with_assertions_no_position_hint = []
    added_lines = 0
    for id, line in  enumerate(method_with_placeholder.splitlines(keepends=True)):
         if(laurel_assertion_placeholder in line):
           added_lines += 1
           laurel_infered_position_missing_assertions.append(id-added_lines)
         else:
           method_with_assertions_no_position_hint.append(line)


    # Save Laurel Restuls :
    prefix = "laurel_" + type

    placeholder_file = prefix + "method_with_placeholder_on_position.dfy"
    laurel_assertion_placeholder_file = base_path / placeholder_file 

    pos_file = prefix + "assertion_position.txt"
    laurel_assertion_pos_file = base_path / pos_file

    error_file = prefix + "error.txt"
    laurel_error_file = base_path / error_file
    
    with open(laurel_error_file , "w", encoding="utf-8") as f:
        f.write(error + "\n Inputed error message \n " + error_message)

    with open(laurel_assertion_placeholder_file , "w", encoding="utf-8") as f:
        f.write(method_with_placeholder )

    with open(laurel_assertion_pos_file , "w", encoding="utf-8") as f:
        f.write(str(laurel_infered_position_missing_assertions))

    return method_with_placeholder, laurel_infered_position_missing_assertions
    

def expand_dataset_with_laurel_fix_position_results(dataset_dir, parallel, type=""):
    """
    Get all possible positions for the fix that makes everything working.
    
    Parameters:
    - DATASET_DIR: Path to the dataset directory.
    - parallel: If 1, runs in parallel using ThreadPoolExecutor. If 0, runs sequentially.
    """
    assertion_dataset = assertion_lib.Dataset(dataset_dir)
    assertion_groups = assertion_dataset.get_all_assertion_groups()
    l = len(assertion_groups)

    if parallel == 0:
        for assertion_group in tqdm(assertion_groups, desc="Creatin Laurel fix positions", total=l):
            run_laurel_fix_position_finder_and_create_laurel_files(assertion_group, type)
    else:
        PHYSICAL_CORES = os.cpu_count() // 2  # Logical cores usually = 2x physical
        SAFE_THREADS = max(1, PHYSICAL_CORES - 1)  # Prevent over-subscription
        with ThreadPoolExecutor(max_workers=SAFE_THREADS) as executor:
            futures = {executor.submit(run_laurel_fix_position_finder_and_create_laurel_files, ag, type): ag for ag in assertion_groups}
            for future in tqdm(as_completed(futures), total=l, desc="Computing syntatic valid positions"):
                future.result()

def check_all_errors(dataset_dir, type=""):
    assertion_dataset = assertion_lib.Dataset(dataset_dir)
    assertion_groups = assertion_dataset.get_all_assertion_groups()
    l = len(assertion_groups)

    err_n = 0
    for assertion_group in assertion_groups:
        err_n += check_laurel_errors(assertion_group, type)
    # 77 errors
    # 419 dao bem ...
    print(err_n)
    print(len(assertion_groups))
       

# Add regular Laurel position inference
def laurel_position_inferer():
    print("Expand dataset computing Laurel fix position placeholder")
    expand_dataset_with_laurel_fix_position_results(gl.DAFNY_ASSERTION_DATASET, 1, "")
    print("Expand dataset computing Laurel Better fix position placeholder")
    expand_dataset_with_laurel_fix_position_results(gl.DAFNY_ASSERTION_DATASET, 1, "BETTER")
