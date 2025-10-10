# Dafny Fine-Tuned Repair

This repository contains all the artifacts required to infer helper annotations in Dafny code.

## Installation

### Clone the Repository
Set large file system of GIT to not load large files unless explicity required (files containing all experiments results):
```
export GIT_LFS_SKIP_SMUDGE=1
```
Clone this repository along with its submodules (that are placed under the external folder, if only need to replicate paper graphs no need to get submodules):

```
git clone --recurse-submodules https://github.com/VeriFixer/dafny_assertion_inference
```

If you need to access to a precomputed datasets and results generated from the scripts used in this paper, fetch the Git LFS files:

```
git lfs fetch --all
git lfs checkout
```

This will download the snapshot files:

* dataset/latest.tar.gz
* results/latest.tar.gz

You will need to uncompress these archives inside their respective directories to access the full contents (with the content to be directly under dataset and results like so)
* dataset/latest.tar.gz
* dataset/dafny_assertion_al (etc)


### Build the Custom Dafny Binary
#### Prerequisites
- Install .NET SDK version 8.0. Download and install from [Microsoft](https://dotnet.microsoft.com/en-us/download).
- Install z3 : on linux fedora: 
Version for reproduce results: Z3 version 4.15.2 - 64 bit

sudo dnf install -y z3

#### Build Dafny
```sh
cd external/dafny_fork
make
```
> **Note:** You may see messages like:
> ```
> FAILURE: Build failed with an exception.
> Execution failed for task ':javadoc'.
> ```
> These errors do not interfere with the actual Dafny binary build.

At the end of the process, you should see:
```
Build succeeded.
```

### Verify the Installation
Run at the top the unit tests for the project
```sh
PYTHONPATH=src python -m unittest discover -s src/tests -p "test_*.py" -v
```
This will run unit tests to verify that the Dafny executable is working. If all tests pass, the installation is successful.


### Building the Laurel and Laurel+ Position Inference Strategies

1. Navigate to the **original Laurel placeholder finder**:

   ```bash
   cd external/dafny_laurel_repair/dafny_laurel_repair/laurel/placeholder_finder
   ```

   Build the original Laurel placeholder inference algorithm:

   ```bash
   dotnet build placeholder_finder.csproj
   ```

2. Navigate to the **Laurel+ (improved) placeholder finder**:

   ```bash
   cd external/dafny_laurel_repair/dafny_laurel_repair/laurel/placeholder_finder_laurel_better
   ```

   Build the improved Laurel+ placeholder inference algorithm:

   ```bash
   dotnet build placeholder_finder_laurel_better.csproj
   ```

## Running the Project
Optional parts are not needed as the user can use the precomputed dataset.

### (Optional) Compute from scratch the full dataset and compute all helper metrics
To perform this step on the full dataset took around 48 hours with (6 parallel cores, 24Gb RAM capped)
```sh
cd src
python full_dataset_creator.py
```

The above steps are optional as their results are already present in:
- `dataset/dafny_assertion_all/` – contains assertions for every file in the dataset.
- `dataset/dafny_assertion_dataset/` – contains only the extracted helper assertions.

### Data Structure
#### `dataset/dafny_assertion_all`
Contains folders for each dataset file:
```
{file_folder}/assert.xml
{file_folder}/program.dfy
```
- `program.dfy`: The actual Dafny code.
- `assert.xml`: Contains extracted assertions.

#### Example `assert.xml`
```xml
<program>
  <name>example.dfy</name>
  <Method>
    <name>_module._default.example_method</name>
    <start_pos>0</start_pos>
    <end_pos>1448</end_pos>
    <assertion>
      <type>Regular_assertion</type>
      <start_pos>1073</start_pos>
      <end_pos>1118</end_pos>
    </assertion>
  </Method>
</program>
```

#### `dataset/dafny_assertion_dataset`
Contains assertion groups:
```
{file_folder}/original_program.dfy
{file_folder}/assertion_group_{id}/info.xml
{file_folder}/assertion_group_{id}/method_without_assertion_group.dfy
{file_folder}/assertion_group_{id}/program_without_assertion_group.dfy
{file_folder}/assertion_group_{id}/verifier_output.txt
```
- `info.xml`: Contains metadata about the assertion group.

#### Example `info.xml`
```xml
<method>
  <name>_module._default.testBinarySearch</name>
  <start_pos>946</start_pos>
  <end_pos>1302</end_pos>
  <assertion_group>
    <id>0</id>
    <number_assertions>2</number_assertions>
    <assertion>
      <type>Regular_assertion</type>
      <start_pos>1018</start_pos>
      <end_pos>1050</end_pos>
    </assertion>
  </assertion_group>
</method>
```

# Replicating paper results without recomputing dataset 
If you did not recompute the dataset and the results:
Use the complete jupyter notebook contain all data analysys

The figures used in the paper are generated using three scripts, all of which output their results under the `images/` folder:

* **`src/data_analysys.ipynb`**

the data that allowed to create the graphs are also shown there


# Replicating paper 
Explore the script

* **`src/main.py`**

This script demonstrates how to **estimate costs and run experiments** for replicating the results of the associated paper.
It also contains all code to run all runned experiments for the paper
## Overview

The script is divided into two main parts:

1. **Cost Estimation (before `exit()`)**
   * Uses a cost-stub LLM (`LLM_COST_STUB_RESPONSE_IS_PROMPT`) to simulate LLM queries and collect cost statistics.
   * Iterates over all combinations of tests in the paper
   * A debug llm_without_api exist there in order for the user to see the prompts in a full pipeline.
  
You must comment both parts before the exit() to test with a real LLM

2. **Actual Experiment Execution (after `exit()`)**

   * Preferred method for **running one experiment efficiently**.
   * The process is divided into **three passes**:
     * **Part 1 – Localization**: prompts the LLM for assertion positions.
     * **Part 2 – Assertion Candidate Generation**: prompts the LLM for assertion candidates.
     * **Part 3 – Verification**: verifies results in parallel (safe to parallelize).

   This separation speeds up experiments by avoiding bottlenecks in the verification step. Comment the exit() and the cost and llm_without_api calls to run all paper experiments.

## Switching from Stub to Real LLM

By default, the script uses the cost stub (`llm_cost_stub`) to simulate responses.
To replicate experiments with actual models, **replace the stub with a real LLM**:

```python
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

llm_openai = llm_open_ai.OpenAI_LLM(
    "gpt_4.1", 
    model="gpt-4.1-2025-04-14", 
    max_context_size=128000, 
    openaiKey=openai_api_key, 
    verbose=0
)

# Then pass `llm_openai` instead of `llm_cost_stub` into evaluate_all(...)
```

Therefore you must have a valid OPENAI_API_KEY in your environment


# Configurations

## Configuring other LLMS
New LLMs can be added in `src/llm/llm_configurations.py`.
Each LLM must implement the following class:
```python
class LLM:
    def __init__(self, name):
        self.name = name

    def get_response(self, prompt):
        raise NotImplementedError("Subclasses must implement the get_response method")

    def get_name(self):
        return self.name

    def set_system_prompt(self):
        raise NotImplementedError("Subclasses must implement the set_system_prompt method")
```

### Example LLM Implementations
Implementations present in llm_configuration.py
- `LLM_COST_STUB_RESPONSE_IS_PROMPT`: Estimates utilization costs.
- `LLM_YIELD_RESULT_WITHOUT_API`: Requires manual ChatGPT interaction.
- `LLM_EMPTY_RESPONSE_STUB`: Always returns empty responses.

Example usage:
```python
llm_stub = llm_configurations.LLM_COST_STUB_RESPONSE_IS_PROMPT("stub")
llm_without_api = llm_configurations.LLM_YIELD_RESULT_WITHOUT_API("without_api")
llm_stub_empty = llm_configurations.LLM_EMPTY_RESPONSE_STUB("stub_empty")
```

### Running LLM Evaluations
```python
evaluate_all(llm_without_api, global_options, run_options)
```
## Available Options for llm pipeline and other global options

See `src/llm/llm_pipeline.py` for examples and a full list of options to run the pipeline.
`llm_pipeline.py` supports two modes when called as a main script:

see `src/utils/global_variables.py` to see other options (like prompts)

# Logging and output files

All prompts and responses are logged in:

```shell
dataset/dafny_llm_results/{llm_name}/{file_folder}/{assertion_group_id}/{round_indication}
```

Inside this directory, several files will be generated:

- **previous_program.dfy** → The complete program that was tested with missing assertions.
- **new_program.dfy** → The program modified with the LLM-generated assertions.
- **oracle.dfy** → The fully corrected method or program.
- **prompt.txt** → The full prompt sent to the LLM.
- **response.txt** → The full response from the LLM.
- **verifier_output.txt** → The complete output from the Dafny verifier.


