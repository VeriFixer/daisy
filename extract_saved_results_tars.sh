#!/bin/bash
# extract_specific_tars.sh
# Extract specific .tar.gz files in place

set -euo pipefail

# List of files to extract (relative to current directory)
tars=(
    "dataset/dafny_assertion_dataset_test.tar.gz"
    "dataset/dafny_assertion_dataset.tar.gz"
    "results/dafny_llm_results_pre_test__testing_different_models.tar.gz"
    "results/dafny_llm_results_rq4__different_llms.tar.gz"
    "results/dafny_llm_results_rq3__example_gatherer.tar.gz"
    "results/dafny_llm_results_rq2__loc_strategy.tar.gz"
    "results/dafny_llm_results_rq1__best_overall.tar.gz"
)

for tarfile in "${tars[@]}"; do
    if [ -f "$tarfile" ]; then
        echo "Extracting $tarfile ..."
        dir="$(dirname "$tarfile")"
        tar -xzf "$tarfile" -C "$dir"
        echo "Done: $tarfile"
    else
        echo "File not found: $tarfile"
    fi
done

echo "All listed tar.gz files extracted."
