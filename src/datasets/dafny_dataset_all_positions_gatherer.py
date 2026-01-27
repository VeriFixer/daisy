from utils.dafny_read_assertions_xml import assertionGroup
import utils.global_variables as gl
import utils.assertion_method_classes as assertion_lib
import utils.dataset_class as dat

import llm.llm_pipeline as llm_pipe
import dafny.dafny_runner as dafny_runner

from utils.run_parallel_or_seq import run_parallel_or_seq
# This method returns also the position of the actual assertions that make this method work
def get_method_for_verification_and_oracle_positions(assertion_group: assertionGroup):
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    text_to_substitute = gl.ASSERTION_PLACEHOLDER
    remove_empty_lines = True


    method_with_assertions = method.get_method_with_assertion_group_changed(assertion_group, remove_empty_lines, text_to_substitute)

    oracle_position_missing_assertions: list[int]= []

    #assertion can be inserted in the same line:
    # IF  line0
    #     missinhere
    #     missinghere
    #     line1
    # modle should return 0 and 0 for both because
    # it receives
    # line0
    # line1 
    # Both asseertion should be between line0 and line1

    method_with_assertions_no_position_hint: list[str]= []
    added_lines = 0
    for id, line in  enumerate(method_with_assertions.splitlines(keepends=True)):
         if(text_to_substitute in line):
           added_lines += 1
           oracle_position_missing_assertions.append(id-added_lines)
         else:
           method_with_assertions_no_position_hint.append(line)

    method_str = "".join(method_with_assertions_no_position_hint)
    return file, method, method_str,  method_with_assertions, oracle_position_missing_assertions

def verify_initial_state(file : assertion_lib.FileInfo, method : assertion_lib.MethodInfo, method_text: str):
    _, full_file_text = file.substitute_method_with_text(method, method_text)
    #print(full_file_text)
    _, stdout_msg, _ = dafny_runner.run_dafny_from_text(gl.DAFNY_EXEC, full_file_text, gl.TEMP_FOLDER)
    return stdout_msg, full_file_text

import json
def get_orignal_error_of_assertion_group(assertion_group: assertionGroup):
        assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
        #print(assertion_group_name)

        file, method, method_missing_assertions, method_with_assertion_placeholder, oracle_position = get_method_for_verification_and_oracle_positions(assertion_group)
        # previous method missing assertions and is buggy this is correct
        file, method, method_missing_assertions = llm_pipe.prepare_method_for_verification_default_options(assertion_group, oracle=True, oracle_text=gl.ASSERTION_PLACEHOLDER)
        
        file_folder = file.file_path.parent
        base_path = file_folder / assertion_group_name
        assertions_list: list[str]= []

        # The assertion_group can be out of order of the assertion need to sort it first 
        # This is necessary to the assertion text to relate to the correct position 
        for a in sorted(assertion_group, key=lambda x: x.start_pos):
            assertions_list.append(a.segment_str)
        
        oracle_assertions_file = base_path / "oracle_assertions.json"
        original_error_file = base_path / "original_error.txt"
        oracle_position_file = base_path / "oracle_fix_position.txt"
        method_with_placeholder_file = base_path / "method_with_assertion_placeholder.dfy"


        if(method_with_placeholder_file.exists()): # Skip if already performed
            return 
        
        with open(oracle_assertions_file, "w", encoding="utf-8") as f:
            json.dump(assertions_list, f, indent=2, ensure_ascii=False)

        original_error, _= verify_initial_state(file, method, method_missing_assertions)

        with open(original_error_file, "w", encoding="utf-8") as f:
            f.write(original_error)

        with open(oracle_position_file, "w", encoding="utf-8") as f:
            f.write(str(oracle_position))

        with open(method_with_placeholder_file, "w", encoding="utf-8") as f:
            f.write(method_with_assertion_placeholder)


# ind: represents which assertion from the ilst is going to be moved
def get_all_method_with_assertions_at_ind_relocated( methodWithoutAssertionLines : list[str], assertions : list[str], 
                                                     assertionOracleLines : list[int], ind :int) -> list[list[str]]: 
    n_assert = len(assertions)
    n_lines_without_assert = len(methodWithoutAssertionLines)
    assert n_lines_without_assert > 0
    assert n_assert == len(assertionOracleLines)
    assert 0 <= ind < n_assert

    target_assertion = assertions[ind]
    methods : list[list[str]] = []
    for change_target_assertion_to_pos in range(n_lines_without_assert):
        method: list[str] = []
        for insert_after_x in range(n_lines_without_assert):
            method.append(methodWithoutAssertionLines[insert_after_x])
            if(change_target_assertion_to_pos == insert_after_x):
                method.append(target_assertion)
            for ind_assertion,(assertion, line) in enumerate(zip(assertions, assertionOracleLines)):
                if(ind == ind_assertion):
                    continue # Skip assertion if assertion is the one we are going to introduce
                if line == insert_after_x:
                    method.append(assertion)
        methods.append(method)

    return methods
    # Exemplo supondo fichiero original (e assumindo que sao todas essenciais)
    # line 0
    # assert 1
    # assert 2
    # line 3
    # assert 4

    # assertionOracleLines
    # [0,0,1]
    # If assertions are to be inserted after line 0, two, and after line 1 one 
    # they will obtain the correct example
    
    # methosWithoutAssertion
    # line 0
    # line 3

    # Algorith Inserts the assertions not at evaluation in the correct position (the ones that are not ind)
    # tests all possible position for placing the ind assertion (and returns all methods with assertion realocated)

# suppose it has 3 assertion on the assertion group returns a list content 3 list, where each list are the valid position for the
# relevant assertion that makes verification work, mantaining the others on the correct position
# [[ posassert1, posassert1], [posasssert2, posassert2 ], ..]
def get_all_valid_positions_to_fix_assertion_group_with_oracle_assertions(assertion_group : assertionGroup) -> list[list[int]]:
    assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)

    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    file_folder = file.file_path.parent
    base_path = file_folder / assertion_group_name
    
    # Early Return if files already exist
    all_lines_that_fix_file = base_path / "all_lines_that_fix_file.json"
    method_missing_assertions_file = base_path /  "method_without_assertion_group.dfy"
    oracle_assertions_file = base_path / "oracle_assertions.json"
    oracle_position_file = base_path / "oracle_fix_position.txt"

    if(all_lines_that_fix_file.exists()):
        return

    method_with_missing_assertions = ""
    with open(method_missing_assertions_file, "r", encoding="utf-8") as f:
       method_with_missing_assertions = f.read()

    assertions_list = []
    with open(oracle_assertions_file, "r", encoding="utf-8") as f:
        assertions_list = json.load(f)

    assertion_oracle_position_list = []
    with open(oracle_position_file,"r", encoding="utf-8") as f:
        assertion_oracle_position_list = json.load(f)
       
    methodWithoutAssertionLines = method_with_missing_assertions.splitlines()
    lines_that_fix_all_assertions: list[list[int]] = []

    for ind in range(len(assertions_list)):
        lines_that_fix_assertion : list[int] = []
        methods = get_all_method_with_assertions_at_ind_relocated(methodWithoutAssertionLines   , assertions_list,
                                                                               assertion_oracle_position_list, ind)
        for (pos,method_new_text_lines) in enumerate(methods):
            method_new_text = "\n".join(method_new_text_lines)
            _, full_file_text = file.substitute_method_with_text(method, method_new_text)
            status, _, _ = dafny_runner.run_dafny_from_text(gl.DAFNY_EXEC, full_file_text, gl.TEMP_FOLDER)
            if(status == dafny_runner.Status.VERIFIED):
                lines_that_fix_assertion.append(pos)
        lines_that_fix_all_assertions.append(lines_that_fix_assertion)

    with open(all_lines_that_fix_file, "w", encoding="utf-8") as f:
            json.dump(lines_that_fix_all_assertions, f, indent=2, ensure_ascii=False)

    return lines_that_fix_all_assertions 

def get_all_syntatix_valid_positions_of_assertions_on_this_group(assertion_group : assertionGroup):

    assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
    #if(assertion_group_name.count("start") <= 3): #to test one in particular
    #      return # Only perform for the ones that are big
    
    method = assertion_lib.get_method_from_assertion_group(assertion_group)

    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    file_folder = file.file_path.parent
    base_path = file_folder / assertion_group_name

    method_missing_assertions_file = base_path / "method_without_assertion_group.dfy"
    all_lines_that_are_syntatic_valid_file = base_path / "all_lines_that_are_syntatic_valid.json"

    # Early exit 
    if(all_lines_that_are_syntatic_valid_file.exists()):
        return 
    
    method_with_missing_assertions = ""
    
    with open(method_missing_assertions_file, "r", encoding="utf-8") as f:
       method_with_missing_assertions = f.read()

    assertion_on_test = "assert 1==1;"
    program_lines = method_with_missing_assertions.splitlines(keepends=True)
    max_lines = len(program_lines)
    lines_that_fix : list[int]= []

    for cur_line_fix_candidate in range(max_lines):
      new_program : list[str] = []
      for id, line in  enumerate(program_lines):
        new_program.append(line)
        if(id == cur_line_fix_candidate):
           new_program.append(assertion_on_test + "\n")
      new_method_str = "".join(new_program)

      _, full_file_text = file.substitute_method_with_text(method, new_method_str)
      _ , stdout_msg, _ = dafny_runner.run_dafny_from_text(gl.DAFNY_EXEC, full_file_text, gl.TEMP_FOLDER, option="resolve")
      if(not ("parse errors detected" in stdout_msg)):
           # lines that represent good places for assertions! 
           lines_that_fix.append(cur_line_fix_candidate)

    with open(all_lines_that_are_syntatic_valid_file , "w", encoding="utf-8") as f:
            json.dump(lines_that_fix, f, indent=2, ensure_ascii=False)



from pathlib import Path
def expand_assertion_groups_with_original_error_info(dataset_dir: Path, parallel: bool):
    """
    Enriches each assertion group with its original verification error and oracle fix position.
    
    Parameters:
    - DATASET_DIR: Path to the dataset directory.
    - parallel: If 1, runs in parallel using ThreadPoolExecutor. If 0, runs sequentially.
    """
    assertion_dataset = dat.Dataset.from_dataset_assertion_groups(dataset_dir)
    assertion_groups = assertion_dataset.get_all_assertion_groups()

    return run_parallel_or_seq(
        assertion_groups,
        get_orignal_error_of_assertion_group,
        "Get Original Error Assertion groups",
        parallel=parallel
    )

import random
def expand_assertion_groups_with_all_fix_positions(dataset_dir : Path, parallel : bool):
    """
    Get all possible positions for the fix that makes everything working.
    
    Parameters:
    - DATASET_DIR: Path to the dataset directory.
    - parallel: If 1, runs in parallel using ThreadPoolExecutor. If 0, runs sequentially.
    """
    assertion_dataset = dat.Dataset.from_dataset_assertion_groups(dataset_dir)
    assertion_groups = assertion_dataset.get_all_assertion_groups()
    random.shuffle(assertion_groups)
    return run_parallel_or_seq(
        assertion_groups,
        get_all_valid_positions_to_fix_assertion_group_with_oracle_assertions,
        "Get All Valid Positions Assertion groups",
        parallel=parallel
    )

def expand_assertion_groups_with_all_syntatic_valid_positions_for_assertions(dataset_dir : Path, parallel : bool):
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
  get_all_syntatix_valid_positions_of_assertions_on_this_group,
        "Get All Syntatic Positions Assertion groups",
        parallel=parallel
    )

def dafny_dataset_all_positions_gatherer():
    DATASET_DIR = gl.DAFNY_ASSERTION_DATASET
    print("Retrieving original error of the Verification and save")
    expand_assertion_groups_with_original_error_info(DATASET_DIR,  True)
    print("Compute All syntatic valid positions to insert assertions")
    expand_assertion_groups_with_all_syntatic_valid_positions_for_assertions(DATASET_DIR, True)
    print("Compute All valid positions for assertions for w/o 1 case")
    expand_assertion_groups_with_all_fix_positions(DATASET_DIR, True)
