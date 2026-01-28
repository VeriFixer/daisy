import os
import shutil
import subprocess

import utils.global_variables as gl
import tempfile
import threading
from pathlib import Path

from enum import Enum
import resource

class Status(Enum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    ERROR = "ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"

VALID_OPTIONS = ['resolve', 'verify', 'build', 'run', 'asserttree']

def build_dafny_command(dafny_exec: Path, dafny_program: Path, option: str = "verify") -> list[str]:
    """
    Build the Dafny execution command and determine if memory limits should be applied.
    Returns (command, use_mem_limit)
    """
    if option not in VALID_OPTIONS:
        raise ValueError(f"Invalid option '{option}'. Must be one of {VALID_OPTIONS}.")

    cmd: list[str] = []

    if option == "verify":
        if gl.IS_SYSTEMD_AVAILABLE:
            cmd = [
                "systemd-run", "--user", "--scope",
                "-p", f"MemoryMax={gl.VERIFIER_MAX_MEMORY}G",
                str(dafny_exec), option, str(dafny_program),
                "--cores", "1",
                "--verification-time-limit", str(gl.VERIFIER_TIME_LIMIT)
            ]
        else:
            # Direct execution with memory limit

            cmd = [
                str(dafny_exec), option, str(dafny_program),
                "--cores", "1",
                "--verification-time-limit", str(gl.VERIFIER_TIME_LIMIT),
                f"--solver-option:O:memory_max_size={gl.VERIFIER_MAX_MEMORY*1000}"
            ]
    else:
        cmd = [str(dafny_exec), option, str(dafny_program), "--cores", "1"]

    return cmd

# -------------------------
# Helper: run a command with optional memory limit
# -------------------------
def run_command(command: list[str], use_mem_limit: bool = False, mem_gb: int = 24) -> tuple[int, str, str]:
    """
    Run a subprocess command with optional hard memory limit.
    Returns (returncode, stdout, stderr)
    """
    result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return result.returncode, result.stdout, result.stderr
    
# -------------------------
# Helper: parse Dafny output
# -------------------------
def parse_dafny_output(stdout: str, stderr: str, command: list[str]) -> Status:
    """
    Parse Dafny stdout/stderr to determine verification status
    """
    status = Status.ERROR  # default
    found_line = 0
    for line in stdout.splitlines():
        if "Dafny program verifier finished" in line:
            found_line = 1
            if "time out" in line:
                status = Status.ERROR
            elif "0 errors" in line:
                status = Status.VERIFIED
            else:
                status = Status.NOT_VERIFIED
            break
        if"resolution/type errors detected in" in line:
            found_line = 1
            status = Status.ERROR
            break

        if"parse errors detected in" in line:
            found_line = 1
            status = Status.ERROR
            break
    if(found_line == 0): # Memory shortage on program
        status = Status.MEMORY_ERROR
    return status

# -------------------------
# Main: run Dafny on a file
# -------------------------
def run_dafny(dafny_exec: Path, dafny_program: Path, destination_folder: Path, option: str = "verify") -> tuple[Status, str, str]:
    """
    Run Dafny on a program file, copying it to destination folder if needed.
    """
    if not os.path.isabs(dafny_exec) or not os.path.isabs(dafny_program) or not os.path.isabs(destination_folder):
        raise ValueError("All paths must be absolute.")

    os.makedirs(destination_folder, exist_ok=True)
    dest_file = destination_folder / dafny_program.name
    if dafny_program != dest_file:
        shutil.copy(dafny_program, dest_file)

    cmd = build_dafny_command(dafny_exec, dest_file, option)
    returncode, stdout, stderr = run_command(cmd)
    if(returncode == -1):
        status = Status.ERROR
    else:
        status = parse_dafny_output(stdout, stderr, cmd)
    return status, stdout, stderr

# -------------------------
# Run Dafny from code string in temp file
# -------------------------
def run_dafny_from_text(dafny_exec: Path, dafny_code: str, destination_folder: Path = gl.TEMP_FOLDER, option: str = "verify", tmp: bool = True) -> tuple[Status, str, str]:
    """
    Run Dafny from a code string, optionally in a unique temp folder per thread
    """
    if tmp:
        thread_dir = tempfile.mkdtemp(prefix=f"dafny_thread_{os.getpid()}_")
        used_folder = Path(thread_dir)
    else:
        used_folder = Path(destination_folder)
        os.makedirs(used_folder, exist_ok=True)

    temp_file = used_folder / f"temp_{os.getpid()}_{threading.get_ident()}.dfy"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(dafny_code)

    status, stdout, stderr = run_dafny(dafny_exec, temp_file, used_folder, option)

    if tmp:
        try:
            temp_file.unlink(missing_ok=True)
            shutil.rmtree(used_folder, ignore_errors=True)
        except Exception as e:
            print(f"Cleanup error: {e}")

    return status, stdout, stderr