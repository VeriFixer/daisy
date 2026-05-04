# Daisy FMCAD Artifact Reviewer Guide

This README is written for artifact reviewers. It explains what Daisy does, how to run a quick sanity check, what outputs to expect, and how to reproduce the paper’s reported results from cached data.

## Artifact Summary

Daisy is an assertion-repair tool for Dafny. Given a Dafny program that does not verify, Daisy localizes missing helper assertions, proposes candidate assertions, and checks them against the Dafny verifier until it finds a corrected version or reports that no fix was found.

The artifact supports the paper’s main claims in three ways:

1. It demonstrates the end-to-end repair pipeline on individual Dafny files.
2. It includes cached outputs for the research-question runs, so the published tables and figures can be regenerated without rerunning every experiment.
3. It includes the code, dataset, and analysis notebooks needed to inspect the results and verify the workflow.

## What Reviewers Should Verify

The artifact is intended to be easy to inspect without prior knowledge of the codebase. A reviewer should be able to:

1. Build or load the Docker image.
2. Run Daisy on one example Dafny file.
3. Inspect the generated artifact files in the results directory.
4. Extract the cached datasets and results.
5. Regenerate the paper’s tables and figures from the notebooks.

## Quick Start for Reviewers

If you only want to sanity-check the artifact, use this path.

### 1. Build the Docker image
It is necessary to have Docker installed.

To reproduce the paper a self contained Dockerfile is available, morever the image is also tar on this repository:

Run to load the image

```sh
docker load -i dafny_research_latest.tar.gz
```

### 2. Run one example through the CLI

The easiest path is the wrapper script at the repository root:

```sh
./docker_run_cli.sh dataset/extracted/dafny_assertion_dataset/SENG2011_tmp_tmpgk5jq85q_exam_ex4_dfy/method_start_0_as_start_197_end_231/program_without_assertion_group.dfy -b
```

If you prefer to call the CLI directly inside the container, use to go inside the container:

```sh
docker run --rm -it \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
  -p 8888:8888 \
  -w /app \
  dafny_research:latest bash
```

```sh
python -m src.cli dataset/extracted/dafny_assertion_dataset/SENG2011_tmp_tmpgk5jq85q_exam_ex4_dfy/method_start_0_as_start_197_end_231/program_without_assertion_group.dfy --model cost_stub_almost_real --localization LLM
```

### 3. What output to expect

For a successful repair, Daisy prints the verification workflow, the number of combinations tested, and the corrected method text. It also saves artifacts under a run directory and reports its location.

However this stub will not be able to repairit, if you want to test an actual LLM pass a OPENROUTER API KEY and you be able to test a free model. 

Typical saved files include:

1. `corrected_method.txt`
2. `corrected_file.dfy`
3. `timings.json`
4. `verification_errors.txt`
5. `selected_method.txt`
6. `localization_positions.json`

If no fix is found, Daisy still writes the run artifacts and exits with a failure message after reporting how many combinations were tested.

## Artifact Layout

The repository already follows a consistent layout that reviewers can inspect directly.

### Dataset archives

The packaged datasets are stored as tarballs in [dataset/](dataset/):

1. [dataset/dafny_assertion_dataset.tar.gz](dataset/dafny_assertion_dataset.tar.gz)
2. [dataset/dafny_assertion_dataset_test.tar.gz](dataset/dafny_assertion_dataset_test.tar.gz)

### Cached results

The research-question result archives are stored as tarballs in [results/](results/):

1. [results/dafny_llm_results_pre_test__testing_different_models.tar.gz](results/dafny_llm_results_pre_test__testing_different_models.tar.gz)
2. [results/dafny_llm_results_rq1__best_overall.tar.gz](results/dafny_llm_results_rq1__best_overall.tar.gz)
3. [results/dafny_llm_results_rq2__loc_strategy.tar.gz](results/dafny_llm_results_rq2__loc_strategy.tar.gz)
4. [results/dafny_llm_results_rq3__example_gatherer.tar.gz](results/dafny_llm_results_rq3__example_gatherer.tar.gz)
5. [results/dafny_llm_results_rq4__different_llms.tar.gz](results/dafny_llm_results_rq4__different_llms.tar.gz)

### Analysis notebooks

The paper’s tables and figures are generated from notebooks under [src/research_questions/](src/research_questions/):

1. [src/research_questions/data_analysys_dataset_overview.ipynb](src/research_questions/data_analysys_dataset_overview.ipynb)
2. [src/research_questions/data_analysys_pre_tests.ipynb](src/research_questions/data_analysys_pre_tests.ipynb)
3. [src/research_questions/data_analysys_cost_statistics.ipynb](src/research_questions/data_analysys_cost_statistics.ipynb)
4. [src/research_questions/data_analysys_rq1_best_overall.ipynb](src/research_questions/data_analysys_rq1_best_overall.ipynb)
5. [src/research_questions/data_analysys_rq2_loc_strategy.ipynb](src/research_questions/data_analysys_rq2_loc_strategy.ipynb)
6. [src/research_questions/data_analysys_rq3_example_gatherer.ipynb](src/research_questions/data_analysys_rq3_example_gatherer.ipynb)
7. [src/research_questions/data_analysys_rq4_different_llms.ipynb](src/research_questions/data_analysys_rq4_different_llms.ipynb)

## How to Extract the Cached Data

Extract the packaged tarballs in place:

```sh
./extract_saved_results_tars.sh
```

This restores the dataset and cached result directories that the analysis notebooks expect.

## How to Reproduce the Paper Results

The paper’s figures and tables are generated from cached results, not from fresh reruns.

### Fast reproduction path

1. Extract the dataset and result tarballs.
2. Open the relevant notebook under [src/research_questions/](src/research_questions/).
3. Run all cells.

This is the recommended path for reviewers because it avoids long API-driven reruns.

### Full recomputation path

If you want to rerun the research-question pipelines, use the entry points in [src/research_questions/](src/research_questions/):

```sh
python -m src.research_questions.main_rq1
python -m src.research_questions.main_rq2
python -m src.research_questions.main_rq3
```

These runs are expensive and may take days on a modest machine. Use them only if you want to reproduce the pipeline itself rather than just the published outputs.

### Dataset recreation

If you want to recreate the dataset from scratch, the repository provides:

```sh
python -m src.datasets.full_dataset_creator
```

The repository notes this as a long-running process.

## Expected Output and Result Files

Daisy writes its run artifacts to a per-run directory under [results/](results/). The output path structure is used consistently across the codebase and is covered by the test suite.

The artifact files commonly include:

1. `verification_errors.txt`
2. `selected_method.txt`
3. `localization_positions.json`
4. `method_with_placeholders.txt`
5. `corrected_method.txt`
6. `corrected_file.dfy`
7. `timings.json`

If a run is successful, the CLI prints the corrected method. If it fails, it prints a no-fix message after testing the available candidates.

## Validation

The repository includes tests that can be used as a lightweight sanity check:

```sh
python -m pytest src/tests/ -q
```

These tests cover key behaviors such as output-path structure, example retrieval, and property-based invariants.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run the CLI as a module from the repository root:

```sh
python -m src.cli <file.dfy> --model <model> --localization <strategy>
```

Do not execute the file directly as `python src/cli.py` unless you have intentionally configured `PYTHONPATH`.

### Cached data missing

If notebooks cannot find the expected inputs, make sure you extracted the tarballs with [extract_saved_results_tars.sh](extract_saved_results_tars.sh).

### Docker or verification instability

The repository documents a Dafny/Z3 memory caveat inside Docker. If verification behaves unexpectedly, use the container workflow exactly as documented and prefer cached results for artifact evaluation.

### LLM-backed runs need credentials

If you use a real model instead of a debug stub, provide the relevant API key environment variables before launching the container.

## Expected Review Outcome

A successful review should confirm that Daisy can repair at least one Dafny example, that the cached result archives can be extracted, and that the notebooks regenerate the paper’s reported tables and figures from those cached inputs.
