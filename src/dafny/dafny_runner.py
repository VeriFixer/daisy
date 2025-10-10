import os
import shutil
import subprocess

import utils.global_variables as gl
import tempfile
import threading

# Run dafny from text can have collisons for same program 
# By default dafny from text runs in a fresh to erase directory
def run_dafny_from_text(dafny_exec, dafny_code, destination_path_folder, option="verify", tmp=1):
    # Always use a unique per-thread temporary directory
    if tmp:
        # Create a uniquely named temp directory using thread ID
        thread_safe_dir = tempfile.mkdtemp(prefix=f"dafny_thread_{os.getpid()}_")
        used_destination_path_folder = thread_safe_dir
    else:
        used_destination_path_folder = destination_path_folder
        os.makedirs(destination_path_folder, exist_ok=True)

    # Write to a uniquely named temp file within the thread-safe directory
    temp_file_path = os.path.join(used_destination_path_folder, f"temp_{os.getpid()}_{threading.get_ident()}.dfy")
    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
        temp_file.write(dafny_code)

    # Run Dafny from the thread-safe file
    status, stdout_content, stderr_content = run_dafny_temp_file_folder(
        dafny_exec, temp_file_path, option
    )

    # Cleanup
    if tmp:
        try:
            os.remove(temp_file_path)
            shutil.rmtree(used_destination_path_folder, ignore_errors=True)
        except Exception as e:
            print(f"Cleanup error: {e}")
    return status, stdout_content, stderr_content

def run_dafny_temp_file_folder(dafny_exec, dafny_program, option="verify"):
    """
    Runs a Dafny program, capturing stdout and stderr, with a specific operation mode.

    Returns: tuple (status, stdout_content, stderr_content)
      - status: 'ERROR', 'VERIFIED', 'NOT_VERIFIED', 'MEMORY_ERROR'
    """
    if not os.path.isabs(dafny_exec) or not os.path.isabs(dafny_program):
        raise ValueError("All paths must be absolute.")

    valid_options = ['resolve', 'verify', 'build', 'run', 'asserttree']
    if option not in valid_options:
        raise ValueError(f"Invalid option '{option}'. Must be one of {valid_options}.")

    if option == "verify":
        command = ["systemd-run", "--user", "--scope", "-p" ,f"MemoryMax={gl.VERIFIER_MAX_MEMORY}G", dafny_exec, option, dafny_program, "--cores", "1", "--verification-time-limit", "300"]
    else:
        command = [dafny_exec, option, dafny_program, "--cores", "1"]

    # Run process
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_content = result.stdout
    stderr_content = result.stderr

    # Otherwise continue with original logic
    status = "ERROR"  # default
    found_line = 0
    for line in stdout_content.splitlines():
        if "Dafny program verifier finished" in line:
            found_line = 1
            if "time out" in line:
                status = "ERROR"
            elif "0 errors" in line:
                status = "VERIFIED"
            else:
                status = "NOT_VERIFIED"
            break
    if(found_line == 0): # Memory shortage on program
        status = "MEMORY_ERROR"

    return (status, stdout_content, stderr_content)


def run_dafny(dafny_exec, dafny_program, destination_path_folder, option="verify"):
    """
    Runs a Dafny program, capturing the standard output and error, with the specified operation mode.

    Parameters:
    - dafny_exec (str): Absolute path to the Dafny executable.
    - dafny_program (str): Absolute path to the Dafny program file.
    - destination_path_folder (str): Absolute path to the destination folder where the Dafny program will be copied and executed.
    - option (str): Operation mode for the Dafny execution. Must be one of:
        - 'resolve': Only check for parse and type resolution errors.
        - 'verify': Verify the program.
        - 'build': Produce an executable binary or a library.
        - 'run': Execute the program (default).
    Returns: tuple (status, stdout_content, stderr_content)
      - status can be ERROR, VERIFIED, NOT_VERIFIED
    """
    if not os.path.isabs(dafny_exec) or not os.path.isabs(dafny_program) or not os.path.isabs(destination_path_folder):
        raise ValueError("All paths must be absolute.")

    valid_options = ['resolve', 'verify', 'build', 'run', 'asserttree']
    if option not in valid_options:
        raise ValueError(f"Invalid option '{option}'. Must be one of {valid_options}.")

    os.makedirs(destination_path_folder, exist_ok=True)

    destination_dafny_program = os.path.join(destination_path_folder, os.path.basename(dafny_program))
    if(dafny_program != destination_dafny_program):
      shutil.copy(dafny_program,  destination_dafny_program)

    stdout_path = os.path.join(destination_path_folder, "stdout.txt")
    stderr_path = os.path.join(destination_path_folder, "stderr.txt")

    if(option == "verify"):
       command = [dafny_exec, option, destination_dafny_program, "--cores","1","--verification-time-limit","300"]
    else:
       command = [dafny_exec, option, destination_dafny_program, "--cores","1"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_content = result.stdout
    stderr_content = result.stderr

    # Error only triggers if the program did not fhished as expected Parse errors are other kind of errors
    # Send erro in case ot time out also
    status = "ERROR"  # default status
    for line in stdout_content.splitlines():
        if "Dafny program verifier finished" in line:
            if "time out" in line:
                status = "ERROR"
            elif "0 errors" in line:
                 status = "VERIFIED"
            else:
                 status = "NOT_VERIFIED"
            break

    return (status, stdout_content, stderr_content)
    
