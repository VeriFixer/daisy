# Steps 
import dafny.dafny_runner as dafny_runner
import utils.global_variables as global_variables
from utils.run_parallel_or_seq import run_parallel_or_seq


import os 
import re
from pathlib import Path

def dafny_file_get_all_assertions(dafny_exec : Path, dafny_program : Path, dafny_destination_dataset_path : Path, temp_dir : Path):
    
    if not os.path.isfile(dafny_exec):
        raise FileNotFoundError(f"The file '{dafny_exec}' does not exist.")

    if not os.path.isfile(dafny_program):
        raise FileNotFoundError(f"The file '{dafny_program}' does not exist.")
        
    if not os.path.isabs(dafny_exec) or not os.path.isabs(dafny_program):
        raise ValueError("All paths must be absolute.")
    
    with open(dafny_program, 'r') as dafny_file:
        dafny_file_text = dafny_file.read()

    program_name: str = dafny_program.name
    program_converted_name = program_name[:-4]+"_dfy"

    (_, stdout, _) = dafny_runner.run_dafny_from_text(dafny_exec, dafny_file_text, temp_dir, option="asserttree")
    
    program_folder = dafny_destination_dataset_path  / program_converted_name
    os.makedirs(program_folder, exist_ok=True)

    assert_xml = program_folder / "assert.xml"
    match = re.search(r"<program>(.*?)</program>", stdout, re.DOTALL)

    if match:
        parsed_output = match.group(0)  # Keep <program> tags
    else:
        parsed_output = ""  # If no match, write an empty file
        print(f"ERROR PROCESSING {dafny_program}")

    with open(assert_xml, "w") as f:
        f.write(parsed_output)

    program_dfy_path = program_folder / "program.dfy"
    with open(program_dfy_path, "w") as f:
       f.write(dafny_file_text)


def dafny_get_all_assertions():
    dataset_path = global_variables.DAFNY_DATASET
    dafny_exec = global_variables.DAFNY_MODIFIED_EXEC_FOR_ASSERTIONS
    base_assertion_dataset = global_variables.DAFNY_BASE_ASSERTION_DATASET
    temp_folder = global_variables.TEMP_FOLDER

    files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f))
    ]

    print("Gathering all assertions from DafnyBench")

    def process_file(file_path : Path, dafny_exec : Path, base_assertion_dataset : Path, temp_folder : Path):
        return dafny_file_get_all_assertions(
            dafny_exec, file_path, base_assertion_dataset, temp_folder
        )

    return run_parallel_or_seq(
        files,
        process_file,
        "Processing Dafny files",
        dafny_exec,
        base_assertion_dataset,
        temp_folder,
        parallel=global_variables.GATHERER_DATASET_PARALLEL,
    )