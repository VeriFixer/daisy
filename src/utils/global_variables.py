# OPTION
GATHERER_DATASET_PARALLEL = 1
# Per defualt do not regernate the dataset embeddings but instead read it from a picke file
GENERATE_DATASET_EMBEDDINGS = 0
VERIFIER_MAX_MEMORY = 24

# AUTOMATIC SETTTED
from pathlib import Path

def find_repo_root(marker=".git"):
    """Finds the root of the repository by looking for a marker (default: .git)."""
    current = Path(__file__).resolve().parent
    while current != current.root:
        if (current / marker).exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find repository root. Make sure you're running inside a valid repo.")
# BASE_PATH is the root of the repository
BASE_PATH = find_repo_root()
#print(f"BASE_PATH: {BASE_PATH}")
# AUTOMATIC OTHER VARIABLES

# Depending on the version of dafny it can depend if it works or not
# Diferent version can have different behavious (to avoid that)
# Only the fork must be used to gatherer the dataset.

DAFNY_EXEC = BASE_PATH/"external/dafny_fork/Binaries/Dafny"
DAFNY_MODIFIED_EXEC_FOR_ASSERTIONS = DAFNY_EXEC

DAFNY_DATASET = BASE_PATH /"external/DafnyBenchFork/DafnyBench/dataset/ground_truth"
import os
TEMP_FOLDER = BASE_PATH / "temp"
if(not os.path.exists(TEMP_FOLDER)):
  os.mkdir(TEMP_FOLDER)

DAFNY_BASE_ASSERTION_DATASET = BASE_PATH / "dataset/dafny_assertion_all"
DAFNY_ASSERTION_DATASET = BASE_PATH / "dataset/dafny_assertion_dataset"

DAFNY_ASSERTION_DATASET_TEST = BASE_PATH / "dataset/dafny_assertion_dataset_test"

LLM_RESULTS_DIR = BASE_PATH / "results/dafny_llm_results"
LLM_RESULTS_DIR_TEST = BASE_PATH / "results/dafny_llm_results_test"

PATH_TO_LAUREL = BASE_PATH / "external/dafny_laurel_repair/laurel"

ASSERTION_PLACEHOLDER = "/*<Assertion is Missing Here>*/"

BASE_PROMPT = """
The Dafny code below fails verification due to missing helper assertions.
Locations needing assertions are marked. For each location, return a JSON array of exactly 10 valid Dafny assertions that could fix the error at that point.
Output: a list of JSON arrays, one per location. No explanations or markdown. Escape double quotes as \\".
Examples:
If one position:
[
  ["assert C;", "assert D;", ...] 
]
If two positions:
[
  ["assert A;", "assert B;", "assert str2 != \\"\\";", ...],
  ["assert C;", "assert D;", ...]
]
If more, continue the pattern (one inner list per position)
Now generate the output do not add any commentary, give only but only the required answer in json format:
"""

LOCALIZATION_BASE_PROMPT = """
You are given a Dafny method with line numbers. 
Your task: return 1 or 2 or more line numbers after which a missing helper assertion should be inserted to fix the program as json.
Format: 
- [3] → one assertion after line 3
- [3, 4] → assertions after lines 3 and 4
Constraints:
- Only insert assertions inside the method body, i.e., between the opening { and closing }.
- Do not insert assertions in:
 - function/predicate/method signatures
 - preconditions (requires)
 - postconditions (ensures)
 - loop invariants
- Your answer must be in JSON list format: e.g., [3] or [3, 4].
- Return no more than 2 lines.
Example:
0: method name(args)
1:   specification
2: {
3:   ...
4: }
→ All valid outputs: [2], [3], [2,3] (4 is outside the method)
-> If answer back [2] the new method would be
0: method name(args)
1:   specification
2: {
3:   /*<Assertion is Missing Here>*/ (added line)
4:   ...
5: }
Now, decide the best line(s) do not add any commentary, give only but only the required answer in json format:
You must send at least one number in the answer!
Only give at most two lines in the answer (one or two options are the only admissible candidates)!
"""

SYSTEM_PROMPT = """You are a dafny developer code expert"""

LOCALIZATION_BASE_PROMPT_WITHOUT_RESTRICTION = """
You are given a Dafny method with line numbers. 
Your task: return 1 or 2 or more line numbers after which a missing helper assertion should be inserted to fix the program as json.
Format: 
- [3] → one assertion after line 3
- [3, 4] → assertions after lines 3 and 4
Constraints:
- Only insert assertions inside the method body, i.e., between the opening { and closing }.
- Do not insert assertions in:
 - function/predicate/method signatures
 - preconditions (requires)
 - postconditions (ensures)
 - loop invariants
- Your answer must be in JSON list format: e.g., [3] or [3, 4].
Example:
0: method name(args)
1:   specification
2: {
3:   ...
4: }
→ All valid outputs: [2], [3], [2,3] (4 is outside the method)
-> If answer back [2] the new method would be
0: method name(args)
1:   specification
2: {
3:   /*<Assertion is Missing Here>*/ (added line)
4:   ...
5: }
Now, decide the best line(s) do not add any commentary, give only but only the required answer in json format:
You must send at least one number in the answer!
You can answer with more than two position as well like [2,3,4].
"""