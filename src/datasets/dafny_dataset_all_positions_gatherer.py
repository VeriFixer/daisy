import utils.global_variables as gl
import utils.assertion_method_classes as assertion_lib
import llm.llm_pipeline as llm_pipe
import dafny.dafny_runner as dafny_runner

from tqdm import tqdm

# This method returns also the position of the actual assertions that make this method work
def get_method_for_verification_and_oracle_positions(assertion_group):
    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    method = assertion_lib.get_method_from_assertion_group(assertion_group)
    text_to_substitute = gl.ASSERTION_PLACEHOLDER
    remove_empty_lines = 1


    method_with_assertions = method.get_method_with_assertion_group_changed(assertion_group, remove_empty_lines, text_to_substitute)

    oracle_position_missing_assertions = []

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

    method_with_assertions_no_position_hint = []
    added_lines = 0
    for id, line in  enumerate(method_with_assertions.splitlines(keepends=True)):
         if(text_to_substitute in line):
           added_lines += 1
           oracle_position_missing_assertions.append(id-added_lines)
         else:
           method_with_assertions_no_position_hint.append(line)

    method_str = "".join(method_with_assertions_no_position_hint)
    return file, method, method_str,  method_with_assertions, oracle_position_missing_assertions

def verify_initial_state(file, method, method_text):
    _, full_file_text = file.substitute_method_with_text(method, method_text)
    #print(full_file_text)
    status, stdout_msg, _ = dafny_runner.run_dafny_from_text(gl.DAFNY_EXEC, full_file_text, gl.TEMP_FOLDER)
    return stdout_msg, full_file_text

import json
def get_orignal_error_of_assertion_group(assertion_group):
        assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
        print(assertion_group_name)

        file, method, method_missing_assertions, method_with_assertion_placeholder, oracle_position = get_method_for_verification_and_oracle_positions(assertion_group)
        # previous method missing assertions and is buggy this is correct
        file, method, method_missing_assertions = llm_pipe.prepare_method_for_verification_default_options(assertion_group, oracle=1, oracle_text=gl.ASSERTION_PLACEHOLDER)
        
        file_folder = file.file_path.parent
        base_path = file_folder / assertion_group_name
        assertions_list = []

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

        original_error, file_missing_assertions = verify_initial_state(file, method, method_missing_assertions)

        with open(original_error_file, "w", encoding="utf-8") as f:
            f.write(original_error)

        with open(oracle_position_file, "w", encoding="utf-8") as f:
            f.write(str(oracle_position))

        with open(method_with_placeholder_file, "w", encoding="utf-8") as f:
            f.write(method_with_assertion_placeholder)


def get_all_valid_positions_to_fix_assertion_group_with_oracle_assertions(assertion_group):
    str_id = assertion_lib.get_assertion_group_string_id(assertion_group)
    
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
    all_lines_that_fix_file = base_path / "all_lines_that_fix_file.json"

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
       
    if(len(assertions_list) == 1):
     assertion_on_test = assertions_list[0] 
     #return;  # Temp as i only want to run the other one this was already expanded
     program_lines = method_with_missing_assertions.splitlines(keepends=True)
     max_lines = len(program_lines)
     try_location = 0
     lines_that_fix = []

     for cur_line_fix_candidate in range(max_lines):
        new_program = []
        for id, line in  enumerate(program_lines):
            new_program.append(line)
            if(id == cur_line_fix_candidate):
                new_program.append(assertion_on_test + "\n")
        new_method_str = "".join(new_program)

        _, full_file_text = file.substitute_method_with_text(method, new_method_str)
        status, stdout_msg, _ = dafny_runner.run_dafny_from_text(gl.DAFNY_EXEC, full_file_text, gl.TEMP_FOLDER)
        if(status == "VERIFIED"):
           lines_that_fix.append(cur_line_fix_candidate)
      
     with open(all_lines_that_fix_file, "w", encoding="utf-8") as f:
            json.dump(lines_that_fix, f, indent=2, ensure_ascii=False)
     print(f"Wrote fixing slots to {all_lines_that_fix_file}")
    
    # Note the line per assertion corresponds to the posiiton of one assertion with the other assertion fixed at the right oracle line
    # this will not correspond directly in the second assertion with a correct llm response (it will be needed to subtract one at the second elment)
    if(len(assertions_list) == 2):
        return #not used anymore 
        # This corresponds to a generatlization of the other
        program_lines = method_with_missing_assertions.splitlines(keepends=True)
        n = len(program_lines)
        slots = list(range(n + 1))  # slots 0..n

        lines_that_fix_all_assertions = []

        # For each assertion we want to test insertion for
        for test_idx, assertion_on_test in enumerate(assertions_list):
            # Gather the “other” oracle (there are only two)
            other_idx = 1 - test_idx
            other_assertion = assertions_list[other_idx]
            other_pos = assertion_oracle_position_list[other_idx]
            fixing_slots = []
            # Try every slot
            for slot in slots:
                new_program = []

                # Build new_program by walking each slot i,
                # inserting test-assertion if i==slot, then other oracle if its pos==i,
                # then the original line i (if any).
                for i in slots:
                    if i < n:
                        new_program.append(program_lines[i])

                    if i == slot:
                        new_program.append(
                            assertion_on_test + f"  // inserted at slot {slot}\n"
                        )
                    if i == other_pos:
                        new_program.append(
                            other_assertion + f"  // oracle at orig pos {other_pos}\n"
                        )

                new_method_str = "".join(new_program)

                # Substitute back into the file and run Dafny
                _, full_file_text = file.substitute_method_with_text(method, new_method_str)
                status, stdout_msg, _ = dafny_runner.run_dafny_from_text(
                    gl.DAFNY_EXEC,
                    full_file_text,
                    gl.TEMP_FOLDER
                )

                if status == "VERIFIED":
                    fixing_slots.append(slot)

            lines_that_fix_all_assertions.append(fixing_slots)

        # Dump results
        with open(all_lines_that_fix_file, "w", encoding="utf-8") as f:
            json.dump(lines_that_fix_all_assertions, f, indent=2, ensure_ascii=False)
        print(f"Wrote fixing slots to {all_lines_that_fix_file}")


def get_all_syntatix_valid_positions_of_assertions_on_this_group(assertion_group):

    assertion_group_name = assertion_lib.get_assertion_group_string_id(assertion_group)
    #if(assertion_group_name.count("start") <= 3): #to test one in particular
    #      return # Only perform for the ones that are big
    
    method = assertion_lib.get_method_from_assertion_group(assertion_group)

    file = assertion_lib.get_file_from_assertion_group(assertion_group)
    file_folder = file.file_path.parent
    base_path = file_folder / assertion_group_name

    method_missing_assertions_file = base_path / "method_without_assertion_group.dfy"
    method_with_missing_assertions = ""

    with open(method_missing_assertions_file, "r", encoding="utf-8") as f:
       method_with_missing_assertions = f.read()

    assertion_on_test = "assert 1==1;"
    program_lines = method_with_missing_assertions.splitlines(keepends=True)
    max_lines = len(program_lines)
    try_location = 0
    lines_that_fix = []
    for cur_line_fix_candidate in range(max_lines):
      new_program = []
      for id, line in  enumerate(program_lines):
        new_program.append(line)
        if(id == cur_line_fix_candidate):
           new_program.append(assertion_on_test + "\n")
      new_method_str = "".join(new_program)

      _, full_file_text = file.substitute_method_with_text(method, new_method_str)
      status, stdout_msg, _ = dafny_runner.run_dafny_from_text(gl.DAFNY_EXEC, full_file_text, gl.TEMP_FOLDER, option="resolve")
      if(not ("parse errors detected" in stdout_msg)):
           # lines that represent good places for assertions! 
           lines_that_fix.append(cur_line_fix_candidate)

    all_lines_that_are_syntatic_valid_file = base_path / "all_lines_that_are_syntatic_valid.json"
    with open(all_lines_that_are_syntatic_valid_file , "w", encoding="utf-8") as f:
            json.dump(lines_that_fix, f, indent=2, ensure_ascii=False)



from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
def expand_assertion_groups_with_original_error_info(dataset_dir, parallel):
    """
    Enriches each assertion group with its original verification error and oracle fix position.
    
    Parameters:
    - DATASET_DIR: Path to the dataset directory.
    - parallel: If 1, runs in parallel using ThreadPoolExecutor. If 0, runs sequentially.
    """
    assertion_dataset = assertion_lib.Dataset(dataset_dir)
    assertion_groups = assertion_dataset.get_all_assertion_groups()
    l = len(assertion_groups)
    PHYSICAL_CORES = os.cpu_count() // 2  # Logical cores usually = 2x physical
    SAFE_THREADS = max(1, PHYSICAL_CORES - 1)  # Prevent over-subscription
    if parallel == 0:
        for number_tested, assertion_group in enumerate(
                tqdm(assertion_groups, desc="Processing assertion groups", total=l)):
            #str_id = assertion_lib.get_assertion_group_string_id(assertion_group)
            #if(str_id == "method_start_1354_as_start_1887_end_1934_as_start_1940_end_1973"):
            get_orignal_error_of_assertion_group(assertion_group)
    else:
        print("Expanding Dataset information with original error and fix position localization of the oracle")
        results = []
        with ThreadPoolExecutor(max_workers=SAFE_THREADS) as executor:
            futures = {executor.submit(get_orignal_error_of_assertion_group, ag): ag for ag in assertion_groups}
            for future in tqdm(as_completed(futures), total=l, desc="Processing assertion groups"):
                result = future.result()
                results.append(result)

import os
import random
def expand_assertion_groups_with_all_fix_positions(dataset_dir, parallel):
    """
    Get all possible positions for the fix that makes everything working.
    
    Parameters:
    - DATASET_DIR: Path to the dataset directory.
    - parallel: If 1, runs in parallel using ThreadPoolExecutor. If 0, runs sequentially.
    """
    assertion_dataset = assertion_lib.Dataset(dataset_dir)
    assertion_groups = assertion_dataset.get_all_assertion_groups()
    random.shuffle(assertion_groups)
    
    l = len(assertion_groups)
    PHYSICAL_CORES = os.cpu_count() // 2  # Logical cores usually = 2x physical
    SAFE_THREADS = max(1, PHYSICAL_CORES)  # Prevent over-subscription
    if parallel == 0:
        for assertion_group in tqdm(assertion_groups, desc="Computing fix positions", total=l):
          #str_id = assertion_lib.get_assertion_group_string_id(assertion_group)
          #if(str_id == "method_start_715_as_start_3540_end_3600_as_start_2211_end_2268"):
            get_all_valid_positions_to_fix_assertion_group_with_oracle_assertions(assertion_group)
    else: 
        with ThreadPoolExecutor(max_workers=SAFE_THREADS) as executor:
            futures = {executor.submit(get_all_valid_positions_to_fix_assertion_group_with_oracle_assertions, ag): ag for ag in assertion_groups}
            for future in tqdm(as_completed(futures), total=l, desc="Computing fix positions"):
                future.result()

def expand_assertion_groups_with_all_syntatic_valid_positions_for_assertions(dataset_dir, parallel):
    """
    Get all possible positions for the fix that makes everything working.
    
    Parameters:
    - DATASET_DIR: Path to the dataset directory.
    - parallel: If 1, runs in parallel using ThreadPoolExecutor. If 0, runs sequentially.
    """
    assertion_dataset = assertion_lib.Dataset(dataset_dir)
    assertion_groups = assertion_dataset.get_all_assertion_groups()

    l = len(assertion_groups)
    PHYSICAL_CORES = os.cpu_count() // 2  # Logical cores usually = 2x physical
    SAFE_THREADS = max(1, PHYSICAL_CORES )  # Prevent over-subscription
    if parallel == 0:
        for assertion_group in tqdm(assertion_groups, desc="Computing syntatic valid positions", total=l):
            get_all_syntatix_valid_positions_of_assertions_on_this_group(assertion_group)
    else:
        with ThreadPoolExecutor(max_workers=SAFE_THREADS) as executor:
            futures = {executor.submit(get_all_syntatix_valid_positions_of_assertions_on_this_group, ag): ag for ag in assertion_groups}
            for future in tqdm(as_completed(futures), total=l, desc="Computing syntatic valid positions"):
                future.result()


DATASET_DIR = gl.DAFNY_ASSERTION_DATASET

def dafny_dataset_all_positions_gatherer():
    print("Retrieving original error of the Verification and save")
    expand_assertion_groups_with_original_error_info(DATASET_DIR, 1)
    print("Compute All syntatic valid positions to insert assertions")
    expand_assertion_groups_with_all_syntatic_valid_positions_for_assertions(DATASET_DIR, 1)
    print("Compute All valid positions for assertions for w/o 1 case")
    expand_assertion_groups_with_all_fix_positions(DATASET_DIR, 1)
