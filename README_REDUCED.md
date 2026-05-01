# Daisy — Quick Start (Reduced)

Essential steps to install, run via Docker, execute an example strategy inside Docker, launch the research notebooks, and reproduce results/datasets.

Installation (local / dev)
- Clone repository (with submodules if needed):

```bash
git clone --recurse-submodules <repo>
cd daisy
```

- Create a virtualenv and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r src/requirements.txt
# Recommended: make package importable
pip install -e .
```

Run scripts (preferred)
- Run scripts as modules from the repo root using `-m` so Python resolves the `src` package correctly:

```bash
python -m src.research_questions.main_rq1
python -m src.research_questions.main_rq2
python -m src.research_questions.main_rq3
```
Docker (build & run)
- Build image:

```bash
docker build -t daisy:latest .
```

- Run interactive container with repo mounted (recommended):

```bash
docker run --rm -it -v $PWD:/app -w /app daisy:latest /bin/bash
```

- The `Dockerfile` installs the package editable during image build (`pip install -e /app`), so `src` imports will work inside the container. Prefer module invocation inside container:

```bash
python -m src.research_questions.main_rq1
```

Convenience wrapper
- A small wrapper `docker_run_cli.sh` is included at the repo root to simplify running a single `.dfy` file through the Docker image. It handles mounts and results mapping.

Usage examples:

```bash
# Build image (first time)
docker build -t daisy:latest .

# Run a file inside Docker, saving artifacts to ./results
./docker_run_cli.sh path/to/program_without_assertion_group.dfy

# Specify model/localization and capture stdout to a file
./docker_run_cli.sh path/to/program.dfy -m openrouter-free -l LLM -o ./results/myrun_stdout.txt

# If your file is outside the repo, mount/build automatically:
./docker_run_cli.sh /abs/path/to/myfile.dfy -b
```

Jupyter notebooks (research analysis)
- Start Jupyter Lab inside container:

```bash
# inside container
pip install -r src/requirements.txt
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

- Open `http://localhost:8888` and run notebooks under `src/` (e.g., `src/data_analysys_rq1_best_overall.ipynb`).

Reproduce results & datasets
- Many scripts expect cached inference results. To restore caches provided with the repo, run:

```bash
./extract_saved_results_tars.sh
```

- To recreate datasets from scratch (long-running):

```bash
python -m src.datasets.full_dataset_creator
```

- To recompute inference/verification for research questions, run the RQ modules after datasets and resources are available; these runs are computationally heavy and may require days of CPU time.

Why `ModuleNotFoundError: No module named 'src'` happened
- Running `python src/research_questions/main_rq1.py` executes a file directly — Python's import path does not include the repo root by default. Solutions:
  - Run as module: `python -m src.research_questions.main_rq1` (preferred)
  - Install editable: `pip install -e .`
  - Set `PYTHONPATH` to repo root (less recommended)

Notes
- I removed the temporary `run` launcher shim and the per-script `sys.path` bootstraps. Project now expects module execution or editable install.

If you want, I can update the `Dockerfile` to drop root usage and run as a non-root `researcher` user by default, or add a small `Makefile` with common commands. Which would you prefer?
