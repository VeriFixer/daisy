# AUTOMATIC SETTTED
from pathlib import Path
import os
import shutil
import subprocess

# Function used to Find Base Path, path of .git repo
def find_repo_root(marker : str =".repo_multi_assertions_marker"):
    """Finds the root of the repository by looking for a marker (default: .git)."""
    current: Path = Path(__file__).resolve().parent
    while str(current) != current.root:
        if (current / marker).exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find repository root. Make sure you're running inside a valid repo.")

def is_systemd_available() -> bool:
    # 1. systemd-run binary exists
    if shutil.which("systemd-run") is None:
        return False

    # 2. systemd is PID 1
    try:
        with open("/proc/1/comm") as f:
            if f.read().strip() != "systemd":
                return False
    except OSError:
        return False

    # 3. Not inside Docker / container
    if os.path.exists("/.dockerenv"):
        return False

    # 4. systemd-run --user actually works
    try:
        subprocess.run(
            ["systemd-run", "--user", "--scope", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=True,
        )
    except Exception:
        return False

    return True



# OPTIONS -------------------------------------------------------
RUN_TEST_THAT_COST_MONEY : bool = False
GATHERER_DATASET_PARALLEL : bool = True
# Per default do not regenerate the dataset embeddings but instead read it from a picke file
GENERATE_DATASET_EMBEDDINGS : bool = False

VERIFIER_TIME_LIMIT : int = 60 # In seconds
VERIFIER_MAX_MEMORY : int = 24 # In Gigabytes

IS_SYSTEMD_AVAILABLE = is_systemd_available()
BASE_PATH: Path = find_repo_root()
TEMP_FOLDER: Path = BASE_PATH / "temp"
DAFNY_EXEC: Path = BASE_PATH/"external/dafny_fork/Binaries/Dafny"

DAFNY_MODIFIED_EXEC_FOR_ASSERTIONS: Path = DAFNY_EXEC
UNIT_TESTS_DIR : Path = BASE_PATH/"src/tests"
DAFNY_DATASET: Path = BASE_PATH /"external/DafnyBenchFork/DafnyBench/dataset/ground_truth"

DAFNY_BASE_ASSERTION_DATASET: Path  = BASE_PATH / "dataset/dafny_assertion_all"
DAFNY_ASSERTION_DATASET: Path  = BASE_PATH / "dataset/dafny_assertion_dataset"
DAFNY_ASSERTION_DATASET_TEST: Path  = BASE_PATH / "dataset/dafny_assertion_dataset_test"

LLM_RESULTS_DIR: Path  = BASE_PATH / "results/dafny_llm_results"
LLM_COSTS_DIR: Path  = BASE_PATH / "results/costs"
LLM_RESULTS_DIR_TEST: Path  = BASE_PATH / "results/dafny_llm_results_test"

PATH_TO_LAUREL: Path  = BASE_PATH / "external/dafny_laurel_repair/laurel"

ASSERTION_PLACEHOLDER : str = "/*<Assertion is Missing Here>*/"

BASE_PROMPT : str = """
Task:
For each location marked as needing assertions, return exactly 10 valid Dafny assertions that could fix the error at that point. 

Output:
- A JSON array of arrays, one inner array per missing assertion location.
- Each inner array must have exactly 10 strings, each string a valid Dafny assertion ending with a semicolon.
- Escape double quotes as \\".
- Do NOT output explanations, markdown, or any other text.

Examples:
# One missing position
[
  ["assert C;", "assert D;", "...", "assert J;"]
]

# Two missing positions
[
  ["assert A;", "assert B;", "...", "assert J;"],
  ["assert C;", "assert D;", "...", "assert L;"]
]
"""

LOCALIZATION_BASE_PROMPT : str = """
You are given a Dafny method with line numbers.
Return the line numbers AFTER which helper assertions should be inserted to fix verification errors.

FORMAT:
- JSON list only (e.g., [3], [5,7]).
- At least one number.
- Do NOT output any explanations.

RULES:
- Line numbers refer to the original program before insertions.
- Assertions are inserted independently after each listed line.
- Only insert inside the method body (between { and }).
- Never insert in signatures, requires, ensures or loop invariants
- The CODE section is your only source for line numbering. Disregard line numbers in the Error logs, as they do not match the local snippet.

INSERT EXAMPLE:

Original:
5: {
6: a := b;
7: c := d;
8: e := f;
9: }

Answer: [6,8]

Becomes:
5: {
6: a := b;
7: <assertion>
8: c := d;
9: e := f;
10: <assertion>
11: }

HEURISTICS (guidance, not mandatory):
These heuristics guide typical proof-repair behavior, but you may choose other valid placements 
- Failing assert → insert just before it.
- Postcondition/forall → near end of block.
- Loop invariant failures → end of loop body.
- Timeout/subset/domain → right before problematic stmt.
- Prefer after assignments, calls, swaps, updates.

Return ONLY the JSON list of line numbers.
"""


SYSTEM_PROMPT : str = """You are a dafny developer code expert"""

if(not os.path.exists(TEMP_FOLDER)):
  os.mkdir(TEMP_FOLDER)