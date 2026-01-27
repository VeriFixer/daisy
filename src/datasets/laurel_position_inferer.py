import sys
import os
import utils.global_variables as gl
import utils.assertion_method_classes as assertion_lib
import dafny.dafny_runner as dafny_runner
import llm.llm_pipeline as llm_pipe

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# Add the laurel directory to sys.path (to call modified wrapper to call C# script)
sys.path.append(os.path.abspath(gl.PATH_TO_LAUREL))
from placeholder_wrapper import call_placeholder_finder

class LaurelS(Enum):
    LAUREL = "LAUREL"
    LAUREL_BETTER = "LAUREL_BETTER"


def check_laurel_errors(assertion_group: assertion_lib.assertionGroup, type : LaurelS):
    str_id = assertion_lib.get_assertion_group_string_id(assertion_group)
    assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    file_folder = file.file_path.parent
    base_path = file_folder / assertion_group_name 

    prefix :str = "laurel_" + type.value + "error.txt"

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


from typing import cast

def run_laurel_fix_position_finder_and_create_laurel_files(assertion_group: assertion_lib.assertionGroup, type : LaurelS):
    assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    file_folder = file.file_path.parent
    base_path = file_folder / assertion_group_name
    
    # program without assertion group
    program_without_assertion_group_file = base_path / "program_without_assertion_group.dfy"
    # Error message
    error_message = ""
    error_message_file = base_path / "verifier_output.txt"
    with open(error_message_file, "r", encoding="utf-8") as f:
        error_message = f.read()
    # Method name (the method name on the program source corresponds to the last elemnt)
    # In the ast we have: _module._default.gaussian instead of gaussian
    method_name = method.method_name.split(".")[-1]

    error_message = llm_pipe.extract_error_blocks(error_message)
    
    if(type == LaurelS.LAUREL):
         use_laurel_better=False
    else:
         use_laurel_better=True

    method_with_placeholder : str
    error : str

    method_with_placeholder, error = cast(tuple[str,str], call_placeholder_finder(
      error_message,
      program_without_assertion_group_file,
      method_name,
      use_laurel_better
    ))

    # Get laurel fix positions
    laurel_infered_position_missing_assertions: list[int] = []
    laurel_assertion_placeholder = "<assertion> Insert assertion here </assertion>"
    method_with_assertions_no_position_hint: list[str] = []
    added_lines = 0
    for id, line in  enumerate(method_with_placeholder.splitlines(keepends=True)):
         if(laurel_assertion_placeholder in line):
           added_lines += 1
           laurel_infered_position_missing_assertions.append(id-added_lines)
         else:
           method_with_assertions_no_position_hint.append(line)


    # Save Laurel Restuls :
    prefix = "laurel_" + type.value

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
    
from pathlib import Path
import utils.dataset_class as dat
from utils.run_parallel_or_seq import run_parallel_or_seq

def expand_dataset_with_laurel_fix_position_results(dataset_dir : Path, parallel: bool, laurelType : LaurelS):
    """
    Get all possible positions for the fix that makes everything working.
    
    Parameters:
    - DATASET_DIR: Path to the dataset directory.
    - parallel: If 1, runs in parallel using ThreadPoolExecutor. If 0, runs sequentially.
    """
    assertion_dataset = dat.Dataset.from_dataset_assertion_groups(dataset_dir) 
    assertion_groups = assertion_dataset.get_all_assertion_groups()

    return run_parallel_or_seq(
        assertion_groups,
        run_laurel_fix_position_finder_and_create_laurel_files,
        "Creating Laurel Fix Pos",
        laurelType,
        parallel=parallel,
    )


def check_all_errors(dataset_dir : Path, type : LaurelS):
    assertion_dataset = dat.Dataset.from_dataset_all(dataset_dir)
    assertion_groups = assertion_dataset.get_all_assertion_groups()

    err_n = 0
    for assertion_group in assertion_groups:
        err_n += check_laurel_errors(assertion_group, type)
    print(err_n)
    print(len(assertion_groups))
       

# Add regular Laurel position inference
def laurel_position_inferer():
    print("Expand dataset computing Laurel fix position placeholder")
    expand_dataset_with_laurel_fix_position_results(gl.DAFNY_ASSERTION_DATASET, gl.GATHERER_DATASET_PARALLEL, LaurelS.LAUREL)
    print("Expand dataset computing Laurel Better fix position placeholder")
    expand_dataset_with_laurel_fix_position_results(gl.DAFNY_ASSERTION_DATASET, gl.GATHERER_DATASET_PARALLEL, LaurelS.LAUREL_BETTER)
