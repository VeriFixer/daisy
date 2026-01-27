# Steps 
from pathlib import Path
import utils.dafny_read_assertions_xml as dafny_read_assertions_xml 
import dafny.dafny_runner as dafny_runner
import utils.global_variables as global_variables
import os 
import utils.get_assertion_to_test as get_assertion_to_test
from utils.run_parallel_or_seq import run_parallel_or_seq

def validate_folder_path(file_path: Path, error_message : str) -> None:
    if not os.path.isdir(file_path):
        raise FileNotFoundError(error_message)
    if not os.path.isabs(file_path):
        raise ValueError("All paths must be absolute.")
    

def validate_file_path(file_path : Path, error_message : str) -> None:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(error_message)
    if not os.path.isabs(file_path):
        raise ValueError("All paths must be absolute.")

def read_file(file_path : Path) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def prepare_output_file(program_name: str, destination_path: Path) -> str:
    program_converted_name: str = program_name.replace(".", "_").replace(" ", "_")
    return os.path.join(destination_path, f"{program_converted_name}.json")

def create_temp_directories(base_path: Path, id : str) -> tuple [str,str]:
    temp_source = os.path.join(base_path, f"src__{id}")
    temp_result = os.path.join(base_path, f"res__{id}")
    os.makedirs(temp_source, exist_ok=True)
    os.makedirs(temp_result, exist_ok=True)
    return temp_source, temp_result


from utils.assertion_method_classes import FileInfo
def process_assertions(dafny_exec : Path, file_info : FileInfo, source_dafny_path: Path , program_dst_folder: Path, max_assertions_to_remove : int):
    for method_info in  file_info.methods: 
        get_assertion_to_test.process_assertions_method(dafny_exec, program_dst_folder,  source_dafny_path, method_info, max_assertions_to_remove)

# number_of_assertions_to_remove if -1 remove all
def dafny_file_dataset_generator(dafny_exec : Path, dafny_program_assertion_folder: Path, dafny_destination_dataset_path: Path, temp_dir: Path, max_assertions_to_remove: int) -> None:
    validate_file_path(dafny_exec, f"The file '{dafny_exec}' does not exist.")
    validate_folder_path(dafny_program_assertion_folder, f"The file '{dafny_program_assertion_folder}' does not exist.")

    assert_xml_file = dafny_program_assertion_folder  / "assert.xml"
    program_file = dafny_program_assertion_folder  / "program.dfy"
    
    assertions_text = read_file(assert_xml_file)
    file_assertions_dict = dafny_read_assertions_xml.extract_assertion(assertions_text, program_file)

    dafny_program = program_file

    dafny_folder_base_name = os.path.basename(dafny_program_assertion_folder)
    dafny_program_dst = dafny_destination_dataset_path / dafny_folder_base_name

    #print(f"Folder is: {dafny_program_dst}")
    os.makedirs(dafny_program_dst, exist_ok=True) 

    dafny_file_text = read_file(dafny_program)
    
    status, stdout_content, stderr_content = dafny_runner.run_dafny_from_text(dafny_exec, dafny_file_text, temp_dir)
    #print("Dafny runnes with")
    #print(status)
    #print(stderr_content)
    if status != dafny_runner.Status.VERIFIED:
        #print(f"Folder is status base program not verifies skiping it: {dafny_program_dst}")
        #print("Returning first status not verified")
        #print(stdout_content)
        #print(stderr_content)
        return

    with open(dafny_program_dst / "original_program.dfy", 'w') as f:
        f.write(dafny_file_text)
    
    #print(f"{dafny_folder_base_name }: Checking assertions")
    process_assertions(dafny_exec, file_assertions_dict, dafny_program, dafny_program_dst, max_assertions_to_remove)


def run_dataset_generation(max_to_assert_to_remove : int):
    base_dir = global_variables.DAFNY_BASE_ASSERTION_DATASET
    dafny_exec = global_variables.DAFNY_EXEC
    assertion_dataset = global_variables.DAFNY_ASSERTION_DATASET
    temp_folder = global_variables.TEMP_FOLDER

    # Only process subdirectories (excluding hidden ones)
    dirs: list [Path] = [
        base_dir / name
        for name in os.listdir(base_dir)
        if (base_dir / name).is_dir() and not name.startswith(".")
    ]

    def process_dir(dafny_dir : Path, dafny_exec: Path, assertion_dataset: Path, temp_folder: Path, max_remove: int):
        return dafny_file_dataset_generator(
            dafny_exec, dafny_dir, assertion_dataset, temp_folder, max_remove
        )

    return run_parallel_or_seq(
        dirs,
        process_dir,
        "Generating Dafny dataset",
        dafny_exec,
        assertion_dataset,
        temp_folder,
        max_to_assert_to_remove,
        parallel=global_variables.GATHERER_DATASET_PARALLEL
    )

def dafny_dataset_generator():
  print("Creating all w/o-1 and w/o-2 testcases")
  run_dataset_generation(2) # this gets single and dual dataset
  print("Creating all w/o-all testcases")
  run_dataset_generation(-1) # this removes all assertions