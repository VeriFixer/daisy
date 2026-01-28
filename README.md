# Daisy

This repository contains all the artifacts required to infer helper annotations in Dafny code.

## Installation (skip if using dockerfile, only need to build docker file and extract experimental results part)
Note this project has a Dockerfile with all dependencies and instalation information, all requisities are prelisted there. Even that being the case here some major things are going to be explained. You can skip the installation part if using the Dockerfile, moreover in any quesiton on the instalation process you can see the Dockerfile and see exactly what is done there

### Extract Experimental results 
All the experimnt results are compressed on these paths. 
Run at the top of the proejects the following command to extract them (it will require close to 10Gb)
```shell
./extract_saved_results_tars.sh
```

For information these are the folders that are compressed
* dataset/dafny_assertion_dataset.tar.gz
* dataset/dafny_assertion_dataset_test.tar.gz

* results/dafny_llm_results_pre_test__testing_different_models.tar.gz
* results/dafny_llm_results_rq1__best_overall.tar.gz
* results/dafny_llm_results_rq2__loc_strategy.tar.gz
* results/dafny_llm_results_rq3__example_gatherer.tar.gz
* results/dafny_llm_results_rq4__different_llms.tar.gz

### Build the Custom Dafny Binary (Optional if not using docker)
#### Prerequisites
- Install .NET SDK version 8.0. Download and install from [Microsoft](https://dotnet.microsoft.com/en-us/download).
- Install z3 : on linux fedora: 
Version for reproduce results: Z3 version 4.15.4 - 64 bit
sudo dnf install -y z3

#### Build Dafny (Optional if not using docker)
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
### Install python dependencies for the project 
``shell
pip install -r src/requirements.txt
``shell

### Verify the Installation
Run at the top the unit tests for the project
```sh
PYTHONPATH=src python -m unittest discover -s src/tests -p "test_*.py" -v
```
This will run unit tests to verify that the Dafny executable is working. If all tests pass, the installation is successful.
Note: Some tests can be shown as skipped (the ones that actually use a provider LLM as those tests cost money). To actually run them go to: `src/utils/global_variables.py` and turn to True the variable (RUN_TEST_THAT_COST_MONEY : bool)


## Replicating Paper Results 
To get all tables on the paper 


# Replicating paper results without recomputing dataset 
If you did not recompute the dataset and the results:
You can use the jupyter notebooks to get all info used on the paper on data analysys

The figures used in the paper are generated using three scripts, all of which output their results under the `images/` folder:

With 

* src/data_analysys_dataset_overview.ipynb
The analyses to reach figures 2 and 3 of the paper can be seen

* src/data_analysys_pre_tests.ipynb
The analyses to reach Table two of the paper in terms of accuracy (the cost was manyally added with the infomration of consumed tokens)

* src/data_analysys_cost_statistics.ipynb
The analyses to reach Table 3 of the paper

Answer RQ1

* src/data_analysys_rq1_best_overall.ipynb
Anayses to answer rq1

* src/data_analysys_rq2_loc_strategy.ipynb
Anayses to answer rq2

* src/data_analysys_rq1_best_overall.ipynb
Anayses to answer rq3

* src/data_analysys_rq4_different_llms.ipynb
Anayses to answer rq4

To launch the notebooks by docker follow section 7 of (README_DOCKER.md). You can run each one of the jupyter notebooks files.
(If any erros with files not found happers you forgot to do step  Extract Experimental results, do that inside the docker)

# Replicating paper recomputing inference results
Explore the main scripts

* src/main_rq1_best_overall.py
* src/main_rq2_fault_localization.py
* src/main_rq3_example_retrieval.py

Those scripts demonstrates how to **estimate costs and run experiments** for replicating the results of the associated paper.
It also contains all code to run all runned experiments for the paper. (Note there is not a main rq4, as rq1 uses the data from rq3, but the data analyses is different, more focus on fault localization)

## Overview

The script is divided into two main parts:

1. **Cost Estimation (before `exit()`)**
   * Uses a cost-stub LLM to simulate LLM queries and collect cost statistics.
   * A debug llm_without_api exist there in order for the user to see the prompts in a full pipeline (to debug and see interactivly what is passed to the LLMs)
  
You must comment both both evalute_all before exit(), specially the llm_without_api to run with actual models (as llm_without_api is interactive and blocking waiting for user input).

You must also comment the exit() to run

2. **Actual Experiment Execution (after `exit()`)**

Runs the experimnts performing for each configuration, localization infernce, assetion inference and verification. Verification multicore. 

## Switching from Stub to Real LLM

Note in order to use a real LLM, for Bedrock you must have as a environmental variable a: 


Therefore you must have a valid OPENAI_API_KEY in your environment AWS_BEARER_TOKEN_BEDROCK and AWS_DEFAULT_REGION set. 

For using gpt you have to set a OPENAI_API_KEY on th environment. Fail to do so a mock that only answer "Mock Reply" is used.


## Replicating Everything Inclusing dataset Creation 
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



# Configurations (Extend the work with more LLMs)

## Configuring other LLMS
New LLMs can be added , see `src/llm/llm_configurations.py`.
Each LLM must implement the following functions extended the LLM class:
```python
class LLM_my_new_llm(LLM): 
    # dummy  llm that only responds with "Dummy response"
    def _get_response(self, prompt:str): 
       return "Dummy response"
```
After that we provide a factory for the LLMs so it is best to add to `src/llm/llm_configurations.py` and entry to the created llm such as in MODEL_REGISTRY add: 

```python
    "my_new_llm": ModelInfo(
        provider="debug",
        model_id="my_new_llm",
        max_context=128_000,
        cost_1M_in=0,
        cost_1M_out=0
    ),
```

With that we can create the llm directly with
```python
my_new_llm = LLM_my_new_llm("some_name", MODEL_REGISTRY["my_new_llm"])
```
Optional you can extend the function 
on `src/llm/llm_create.py`, create_llm, adding a case to your llm, and then the creation is like so:
```python
my_new_llm = create_llm("my_new_llm")
``` 

after that to use is like the several examples that do exist on the main_rq*.py files.

## Running LLM Evaluations
```python
evaluate_all(llm_without_api, global_options, run_options)
```
### Available Options for llm pipeline and other global options

See `src/llm/llm_pipeline.py` for examples and a full list of options to run the pipeline.
`llm_pipeline.py` supports two modes when called as a main script:

see `src/utils/global_variables.py` to see other options (like prompts)

## Logging and output files

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

`
# Consierations for full reproducability

Note When running a main_rq* the results are dumped on a diretcory with name results/dafny_llm_results 

when ending they have to be manually copied to a folder that matches the specific data_analyses we want to perform 

Also note that the results from rq4, correspond to some models that are run on rq1 and have to be copied manually.

Also for fully replicate the paper around 6 compute days on a 6 core machine were needed. 

# Docker Limitations 

Sometimes the verifier enter a loop where its memory requiremnts are unbounded, to protect the script when running verification from crashing without memory, systemd-run was being used to stop a process if it used too much memory without 
craching the launching script:
See `src/dafny/dafny_runner.py` to find the line in question
```python
     command = ["systemd-run", "--user", "--scope", "-p" ,f"MemoryMax={gl.VERIFIER_MAX_MEMORY}G", str(dafny_exec), option, str(dafny_program), "--cores", "1", "--verification-time-limit", str(gl.VERIFIER_TIME_LIMIT)]
```
If you need to change the max memory to beteer represent your system capacities you can change here: `src/utils/global_variables.py` (VERIFIER_MAX_MEMORY : int = 24 # In Gigabytes) 
The variable must be less than your computer memory, in order for the program to be able to kill the program.

The limitaion is that this command does not work in docker the systemd-run, therefore when running in docker a option is passed directly to z3 to enforce this constraint, but it is not as stable.

