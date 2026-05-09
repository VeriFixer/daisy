# Daisy FMCAD Artifact Reviewer Guide

This README is written for artifact reviewers. It explains what Daisy does, how to run a quick sanity check, what outputs to expect, and how to reproduce the paper's reported results from cached data.

## Artifact Summary

Daisy is an assertion-repair tool for Dafny. Given a Dafny program that does not verify, Daisy localizes missing helper assertions, proposes candidate assertions, and checks them against the Dafny verifier until it finds a corrected version or reports that no fix was found.

The artifact supports the paper's main claims in three ways:

1. It demonstrates the end-to-end repair pipeline on individual Dafny files.
2. It includes cached outputs for the research-question runs, so the published tables and figures can be regenerated without rerunning every experiment.
3. It includes the code, dataset, and analysis notebooks needed to inspect the results and verify the workflow.

## Quick Start for Reviewers

If you only want to sanity-check the artifact, use this path.
Note the docker image was tested on a x86_64 cpu (Intel(R) Core(TM) Ultra 9 285H) with 32Gb of RAM running Linux Nobara operating system version 43 (KDE Plasma Desktop Edition).

### 1. Build the Docker image
It is necessary to have Docker installed.

https://docs.docker.com/engine/install/

Make sure Docker is running before proceeding.

To reproduce the paper a self contained Dockerfile is available, morever the image is also tar on this repository:

Run the following command to load the image.
Loading the image may take some time, varying significantly depending on the host hardware. In our tests, the process took approximately 15 minutes.

The loaded image occupies approximately 21 GB of storage space.

```sh
# Load the docker image
docker load -i dafny_research_latest.tar.gz
```

If there exist a need to manually build the image that can also be achieved with 

```sh
# Build the docker image
docker build -t dafny_research:latest .
```
### 2. Extract precomputed  dataset and results
Go inside docker 

```sh
# Run the docker image
docker run --rm -it \
  -p 8888:8888 \
  -w /app \
  dafny_research:latest bash
```

Inside docker
```sh
# Extract dataset + results (~15GB)
./extract_saved_results_tars.sh
```

### 4. Run the pipeline manually (without_api mode)

The `without_api` model is a debug mode that lets you act as the LLM: the tool prints the exact prompts it would send to a model, and you paste the responses manually. This is useful for:

- Inspecting the prompts that Daisy sends to models at each stage.
- Testing the pipeline end-to-end without API keys or costs.
- Trying the latest chat models via their free web interfaces (copy the prompt, paste the response).
- Debugging localization or assertion-inference behavior.

The example we will run is this:
```txt
=== TASK === 
 Verifier error:
 /tmp/dafny_repair_0mkej8to/tmpwzt796xq.dfy(6,11): Error: a postcondition could not be proved on this return path
  |
6 |     } else {
  |            ^

/tmp/dafny_repair_0mkej8to/tmpwzt796xq.dfy(2,20): Related location: this is the postcondition that could not be proved
  |
2 | ensures (n*(n-1))%2 == 0
  |         ^^^^^^^^^^^^^^^^


Dafny program verifier finished with 1 verified, 1 error

 Program (numbered):
0: lemma {:induction false} Divby2(n: nat)
1: ensures (n*(n-1))%2 == 0
2: {
3:     if n == 0 {
4:         assert (1*(1-1))%2 == 0; // base case
5:     } else {
6:         Divby2(n - 1); // proved in case n - 1
7:          // expanded case n - 1
8:     }
9: }
```

**Expected runtime:** The verification phases (Dafny calls) typically take <4 minuted, but it depends on the answeres provided by the model.

Run he following command to run our strategy on the provided example
```sh
# Run the python command
python -m src.cli \
  dataset/dafny_assertion_dataset/SENG2011_tmp_tmpgk5jq85q_exam_ex4_dfy/method_start_0_as_start_197_end_231/program_without_assertion_group.dfy \
  --localization HYBRID \
  --assertion LLM_EXAMPLE \
  --model without_api \
  --n-examples-pos 3 \
  --n-examples-inf 3 \
  --s-examples-pos DYNAMIC \
  --s-examples-inf DYNAMIC 
```

**CLI parameters explained:**
The localization names on the paper differ from the CLI. Their mappings are:

| CLI (`--localization`) | Paper Name |
|------------------------|------------|
| `LLM` | LLMFL |
| `LLM_EXAMPLE` | ExLLMFL |
| `LAUREL` | DiagFL |
| `LAUREL_BETTER` | DiagFL+ |
| `HYBRID` | BothFL |
| `ORACLE` | ExactFL |

Same for the example retrieval strategies:

| CLI (`--s-examples-pos` / `--s-examples-inf`) | Paper Name |
|-----------------------------------------------|------------|
| `NONE` (i.e. `--assertion LLM` without examples) | NoEx |
| `RANDOM` | Random |
| `TFIDF` | TF-IDF |
| `EMBEDDED` | CodeEmb |
| `DYNAMIC` | MixEmb (with alpha variants: 0.25, 0.50, 0.75, 1.00) |


| Parameter | Description |
|-----------|-------------|
| `<file>` | Path to the `.dfy` file to repair. |
| `--localization` | Position-inference strategy. Options: `LLM`, `LLM_EXAMPLE`, `LAUREL`, `LAUREL_BETTER`, `HYBRID`, `NONE`. |
| `--assertion` | Assertion-inference strategy. Options: `LLM`, `LLM_EXAMPLE`. |
| `--model` | Model to use. Use `without_api` for manual mode, or a real model name (see list below). |
| `--num-assertions` | Number of candidate assertions to generate per position (default: 10). |
| `--n-examples-pos` | Number of few-shot examples for the localization prompt. |
| `--n-examples-inf` | Number of few-shot examples for the assertion-inference prompt. |
| `--s-examples-pos` | Example retrieval strategy for localization: `NONE`, `RANDOM`, `TFIDF`, `EMBEDDED`, `DYNAMIC`. |
| `--s-examples-inf` | Example retrieval strategy for assertion inference: `NONE`, `RANDOM`, `TFIDF`, `EMBEDDED`, `DYNAMIC`. |
| `--rounds` | Number of assertion-inference rounds (default: 1). |

The method to repair is auto-detected from the Dafny verifier errors. If the file contains multiple methods, Daisy selects the one whose name appears in the error output.

When running examples this warning may appear (it is harmless and comes from internal dependencies):
```txt
/opt/venv/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
/opt/huggingface/modules/transformers_modules/jinaai/jina_hyphen_bert_hyphen_v2_hyphen_qk_hyphen_post_hyphen_norm/3baf9e3ac750e76e8edd3019170176884695fb94/configuration_bert.py:29: UserWarning: optimum is not installed. To use OnnxConfig and BertOnnxConfig, make sure that `optimum` package is installed
  warnings.warn("optimum is not installed. To use OnnxConfig and BertOnnxConfig, make sure that `optimum` package is installed")
```

#### 4.1 Verification Phase
The tool first runs Dafny verification and detects the failing method.
Example output:

```txt
═══ Dafny Assertion Repair ═══
  File:          dataset/dafny_assertion_dataset/SENG2011_tmp_tmpgk5jq85q_exam_ex4_dfy/method_start_0_as_start_197_end_231/program_without_assertion_group.dfy
  Model:         without_api (without_api)
  Localization:  HYBRID
  Assert Infer:  LLM_EXAMPLE
  Assertions:    10 per position, 1 round(s)
  N Example (Pos, Inf): (3, 3)

── Verification ──
  Status: NOT_VERIFIED
  Errors:
    /tmp/dafny_repair_0mkej8to/tmpwzt796xq.dfy(6,11): Error: a postcondition could not be proved on this return path
      |
    6 |     } else {
      |            ^
    
    /tmp/dafny_repair_0mkej8to/tmpwzt796xq.dfy(2,20): Related location: this is the postcondition that could not be proved
      |
    2 | ensures (n*(n-1))%2 == 0
      |         ^^^^^^^^^^^^^^^^
    
    
    Dafny program verifier finished with 1 verified, 1 error

  Selected method: _module._default.Divby2
```
#### 4.2 Localization Phase 
The HYBRID localization strategy combines a heuristic-based tool (LAUREL_BETTER) with an LLM-based predictor. The LLM receives:

- The verifier error
- The numbered Dafny method
- Several in-context examples (controlled by `--n-examples-pos` and `--s-examples-pos`)

The model must predict line numbers where assertions should be inserted.

```txt
── Localization ──
 Strategy: HYBRID
/opt/venv/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
/opt/huggingface/modules/transformers_modules/jinaai/jina_hyphen_bert_hyphen_v2_hyphen_qk_hyphen_post_hyphen_norm/3baf9e3ac750e76e8edd3019170176884695fb94/configuration_bert.py:29: UserWarning: optimum is not installed. To use OnnxConfig and BertOnnxConfig, make sure that `optimum` package is installed
  warnings.warn("optimum is not installed. To use OnnxConfig and BertOnnxConfig, make sure that `optimum` package is installed")
The prompt is Prompt

System Prompt: You are a dafny developer code expert

Main Prompt: 
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
Consider these examples: 
=== EXAMPLE ===
Error:
/tmp/dafny_thread_116087_twyep69i/temp_116087_140607468558016.dfy(6,11): Error: a postcondition could not be proved on this return path
  |
6 |     } else {
  |            ^

/tmp/dafny_thread_116087_twyep69i/temp_116087_140607468558016.dfy(2,20): Related location: this is the postcondition that could not be proved
  |
2 | ensures (n*(n-1))%2 == 0
  |         ^^^^^^^^^^^^^^^^


Dafny program verifier finished with 1 verified, 1 error

CODE:
0: lemma {:induction false} Divby2(n: nat)
1: ensures (n*(n-1))%2 == 0
2: {
3:     if n == 0 {
4:          // base case
5:     } else {
6:         Divby2(n - 1); // proved in case n - 1
7:          // expanded case n - 1
8:     }
9: }
OUTPUT:
[3, 5]
=== END ===
=== EXAMPLE ===
Error:
/tmp/dafny_thread_116087_z6970udl/temp_116087_140607451772608.dfy(6,11): Error: a postcondition could not be proved on this return path
  |
6 |     } else {
  |            ^

/tmp/dafny_thread_116087_z6970udl/temp_116087_140607451772608.dfy(2,20): Related location: this is the postcondition that could not be proved
  |
2 | ensures (n*(n-1))%2 == 0
  |         ^^^^^^^^^^^^^^^^


Dafny program verifier finished with 1 verified, 1 error

CODE:
0: lemma {:induction false} Divby2(n: nat)
1: ensures (n*(n-1))%2 == 0
2: {
3:     if n == 0 {
4:         assert (1*(1-1))%2 == 0; // base case
5:     } else {
6:         Divby2(n - 1); // proved in case n - 1
7:          // expanded case n - 1
8:     }
9: }
OUTPUT:
[6]
=== END ===
=== EXAMPLE ===
Error:
/tmp/dafny_thread_116087_g_bq83se/temp_116087_140607443379904.dfy(32,16): Error: a postcondition could not be proved on this return path
   |
32 |     if x%2 == 1 {
   |                 ^

/tmp/dafny_thread_116087_g_bq83se/temp_116087_140607443379904.dfy(30,18): Related location: this is the postcondition that could not be proved
   |
30 |     ensures x % 2 == 0
   |             ^^^^^^^^^^


Dafny program verifier finished with 10 verified, 1 error

CODE:
0: lemma cubEven_Lemma (x:int)
1:     requires (x*x*x + 5) % 2 == 1
2:     ensures x % 2 == 0
3: {
4:     if x%2 == 1 {
5:         var k := (x-1)/2;
6:     }
7: }
OUTPUT:
[5, 5]
=== END ===

=== TASK === 
 Verifier error:
 /tmp/dafny_repair_0mkej8to/tmpwzt796xq.dfy(6,11): Error: a postcondition could not be proved on this return path
  |
6 |     } else {
  |            ^

/tmp/dafny_repair_0mkej8to/tmpwzt796xq.dfy(2,20): Related location: this is the postcondition that could not be proved
  |
2 | ensures (n*(n-1))%2 == 0
  |         ^^^^^^^^^^^^^^^^


Dafny program verifier finished with 1 verified, 1 error

 Program (numbered):
0: lemma {:induction false} Divby2(n: nat)
1: ensures (n*(n-1))%2 == 0
2: {
3:     if n == 0 {
4:         assert (1*(1-1))%2 == 0; // base case
5:     } else {
6:         Divby2(n - 1); // proved in case n - 1
7:          // expanded case n - 1
8:     }
9: }
 OUTPUT: JSON array of line numbers ONLY, e.g. [2,5] (NO OTHER TEXT OR EXPLANATION)
```
#### 4.3 Receiving model response 
In `without_api` mode, the user provides the model response. You can copy the prompt above into any chatbot and paste the answer back. In this case we entered `[3]`:
```
Enter your response (write #END# to end):
[3]
#END#
```
The framework combines heuristic positions (from LAUREL_BETTER) with model-predicted positions:
```
Predicted lines: [7, 3]
```
#### 4.4 Assertion inference phase

Next, the assertion inference model generates candidate assertions for each predicted position.

The model receives:

- verifier errors
- the method with placeholders at predicted positions
- several few-shot examples (controlled by `--n-examples-inf` and `--s-examples-inf`)

```
── Assertion Inference ──
  Strategy: LLM_EXAMPLE
The prompt is Prompt

System Prompt: You are a dafny developer code expert

Main Prompt: 
Task:
For each location marked as needing assertions, return exactly 10 valid Dafny assertions that could fix the error at that point. 

Output:
- A JSON array of arrays, one inner array per missing assertion location.
- Each inner array must have exactly 10 strings, each string a valid Dafny assertion ending with a semicolon.
- Escape double quotes as \".
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
Consider these examples: 
=== EXAMPLE ===
Error:
/tmp/dafny_thread_116087_z6970udl/temp_116087_140607451772608.dfy(6,11): Error: a postcondition could not be proved on this return path
  |
6 |     } else {
  |            ^

/tmp/dafny_thread_116087_z6970udl/temp_116087_140607451772608.dfy(2,20): Related location: this is the postcondition that could not be proved
  |
2 | ensures (n*(n-1))%2 == 0
  |         ^^^^^^^^^^^^^^^^


Dafny program verifier finished with 1 verified, 1 error
CODE:
lemma {:induction false} Divby2(n: nat)
ensures (n*(n-1))%2 == 0
{
    if n == 0 {
        assert (1*(1-1))%2 == 0; // base case
    } else {
        Divby2(n - 1); // proved in case n - 1
        /*<Assertion is Missing Here>*/ // expanded case n - 1
    }
}
OUTPUT (as this is oracle only one option is shown the one that fixes the problem): 
 [['assert (n-1)*(n-2) == n*n -3*n + 2;']]
=== END ===
=== EXAMPLE ===
Error:
/tmp/dafny_thread_116087_twyep69i/temp_116087_140607468558016.dfy(6,11): Error: a postcondition could not be proved on this return path
  |
6 |     } else {
  |            ^

/tmp/dafny_thread_116087_twyep69i/temp_116087_140607468558016.dfy(2,20): Related location: this is the postcondition that could not be proved
  |
2 | ensures (n*(n-1))%2 == 0
  |         ^^^^^^^^^^^^^^^^


Dafny program verifier finished with 1 verified, 1 error
CODE:
lemma {:induction false} Divby2(n: nat)
ensures (n*(n-1))%2 == 0
{
    if n == 0 {
        /*<Assertion is Missing Here>*/ // base case
    } else {
        Divby2(n - 1); // proved in case n - 1
        /*<Assertion is Missing Here>*/ // expanded case n - 1
    }
}
OUTPUT (as this is oracle only one option is shown the one that fixes the problem): 
 [['assert (1*(1-1))%2 == 0;'], ['assert (n-1)*(n-2) == n*n -3*n + 2;']]
=== END ===
=== EXAMPLE ===
Error:
/tmp/dafny_thread_116087_g_bq83se/temp_116087_140607443379904.dfy(32,16): Error: a postcondition could not be proved on this return path
   |
32 |     if x%2 == 1 {
   |                 ^

/tmp/dafny_thread_116087_g_bq83se/temp_116087_140607443379904.dfy(30,18): Related location: this is the postcondition that could not be proved
   |
30 |     ensures x % 2 == 0
   |             ^^^^^^^^^^


Dafny program verifier finished with 10 verified, 1 error
CODE:
lemma cubEven_Lemma (x:int)
    requires (x*x*x + 5) % 2 == 1
    ensures x % 2 == 0
{
    if x%2 == 1 {
        var k := (x-1)/2;
        /*<Assertion is Missing Here>*/
        /*<Assertion is Missing Here>*/
    }
}
OUTPUT (as this is oracle only one option is shown the one that fixes the problem): 
 [['assert x*x*x + 5 == (2*k+1)*(2*k+1)*(2*k+1) + 5\n                == 8*k*k*k + 12*k*k + 6*k + 6\n                == 2*(4*k*k*k + 6*k*k + 3*k + 3);'], ['assert false;']]
=== END ===

 === TASK === 

ERROR:
/tmp/dafny_repair_a2g6bcvw/tmpnovo6v5b.dfy(6,11): Error: a postcondition could not be proved on this return path
  |
6 |     } else {
  |            ^

/tmp/dafny_repair_a2g6bcvw/tmpnovo6v5b.dfy(2,20): Related location: this is the postcondition that could not be proved
  |
2 | ensures (n*(n-1))%2 == 0
  |         ^^^^^^^^^^^^^^^^


Dafny program verifier finished with 1 verified, 1 error

CODE:
lemma {:induction false} Divby2(n: nat)
ensures (n*(n-1))%2 == 0
{
    if n == 0 {
/*<Assertion is Missing Here>*/
        assert (1*(1-1))%2 == 0; // base case
    } else {
        Divby2(n - 1); // proved in case n - 1
         // expanded case n - 1
/*<Assertion is Missing Here>*/
    }
}
OUTPUT:
Enter your response as a JSON array of arrays (containing the assertions to fix the program) ONLY, no extra text. (NO OTHER TEXT OR EXPLANATION)
```

##### 4.5 Response from the model
Again in `without_api` mode, you paste the model's response. Here we used a chatbot to generate candidate assertions:

```txt
Enter your response (write #END# to end):
[["assert n == 0;","assert (n*(n-1))%2 == 0;","assert (0*(0-1))%2 == 0;","assert (1*(1-1))%2 == 0;","assert n*n - n == 0;","assert n <= 0;","assert 0 % 2 == 0;","assert n*(n-1) == 0;","assert (n-1) == -1;","assert true;"],["assert (n-1)*(n-2) == n*n -3*n + 2;","assert ((n-1)*(n-2))%2 == 0;","assert n*(n-1) == (n-1)*(n-2) + 2*(n-1);","assert 2*(n-1)%2 == 0;","assert (n*n - n) == (n-1)*(n-2) + 2*(n-1);","assert n > 0;","assert (n-1) >= 0;","assert (n-1)*(n-2) + 2*(n-1) == n*(n-1);","assert ((n-1)*(n-2) + 2*(n-1))%2 == 0;","assert true;"]]
#END#
```
#### 4.6 Candidate verification

The framework tests combinations of generated assertions against the Dafny verifier. In the testing machine this phase took less than 1 minute. (overall the part that thats longer is the example gatherer that uses expensive embeddings to find the best matches).

```txt
── Verification ──
  Tested 30 combinations, 8 verified ✓

── Corrected Method ──
lemma {:induction false} Divby2(n: nat)
ensures (n*(n-1))%2 == 0
{
    if n == 0 {
assert (n*(n-1))%2 == 0;
        assert (1*(1-1))%2 == 0; // base case
    } else {
        Divby2(n - 1); // proved in case n - 1
         // expanded case n - 1
assert ((n-1)*(n-2))%2 == 0;
    }
}

Full artifacts saved at: /app/results/cli_runs/20260507_193805_program_without_assertion_group_without_api
```

### 4.7 On using real LLMs

When using a real (integrated) LLM, the pipeline is fully automated — the model receives the same prompts shown above and responds programmatically. No manual input is needed.

To use a real model, provide an API key inside the Docker container. For example, using the free models from OpenRouter:

```sh
# Export a key
export OPENROUTER_API_KEY="your-key"
```

> **Disclaimer:** The `openrouter-free` and `qwen3-coder-free` models are provided for testing convenience only. These free-tier models have lower capability and rate limits. The paper's results were obtained with commercial models which produce significantly better repair rates. For faithful reproduction of the paper's numbers, use the same commercial models listed in the paper.

**Example 1** — Same benchmark as the manual walkthrough, now fully automated:

```sh
# Run the python command
python -m src.cli \
  dataset/dafny_assertion_dataset/SENG2011_tmp_tmpgk5jq85q_exam_ex4_dfy/method_start_0_as_start_197_end_231/program_without_assertion_group.dfy \
  --localization HYBRID \
  --assertion LLM_EXAMPLE \
  --model openrouter-free \
  --n-examples-pos 3 \
  --n-examples-inf 3 \
  --s-examples-pos DYNAMIC \
  --s-examples-inf DYNAMIC 
```

**Example 2** — Using the LLM-only localization (no heuristic) with fewer examples:

(In a run we perform the free model did not found a fix, and the verifier tested all assertions still taking 2-3 minutes). Your results may be different, also the free version may be rate limited.

```sh
# Run the python command
python -m src.cli \
  dataset/dafny_assertion_dataset/SENG2011_tmp_tmpgk5jq85q_exam_ex4_dfy/method_start_0_as_start_197_end_231/program_without_assertion_group.dfy \
  --localization LLM_EXAMPLE \
  --assertion LLM_EXAMPLE \
  --model openrouter-free \
  --n-examples-pos 1 \
  --n-examples-inf 1 \
  --s-examples-pos RANDOM \
  --s-examples-inf RANDOM 
```

**Example 3** — Using the LAUREL_BETTER heuristic for localization (no LLM needed for position inference, only for assertion generation):

```sh
# Run the python command
python -m src.cli \
  dataset/dafny_assertion_dataset/SENG2011_tmp_tmpgk5jq85q_exam_ex4_dfy/method_start_0_as_start_197_end_231/program_without_assertion_group.dfy \
  --localization LAUREL_BETTER \
  --assertion LLM \
  --model openrouter-free \
  --num-assertions 15
```

**Expected runtime:** Each run takes 1–5 minutes depending on the model's response time and the number of candidate combinations to verify.

**Expected output:** On success, the CLI prints the corrected method and saves all artifacts to `results/cli_runs/`. On failure, it reports "No fix found" after exhausting candidates.

You can list all available models by running with an invalid model name: 

```sh
# Run the python command
python -m src.cli \
  dataset/dafny_assertion_dataset/SENG2011_tmp_tmpgk5jq85q_exam_ex4_dfy/method_start_0_as_start_197_end_231/program_without_assertion_group.dfy \
  --model NOT_EXIST 
```
Expected output:
```txt
Error: unknown model 'NOT_EXIST'.
Valid models:
  claude-haiku-4.5               provider=bedrock  model_id=us.anthropic.claude-haiku-4-5-20251001-v1:0
  claude-opus-4.5                provider=bedrock  model_id=us.anthropic.claude-opus-4-5-20251101-v1:0
  claude-sonnet-4.5              provider=bedrock  model_id=us.anthropic.claude-sonnet-4-5-20250929-v1:0
  cost_stub_almost_real          provider=debug  model_id=cost_stub_almost_real
  cost_stub_response_dafnybench  provider=debug  model_id=cost_stub_response_dafnybench
  deepseek-r1                    provider=bedrock  model_id=us.deepseek.r1-v1:0
  gpt-4.1                        provider=openai  model_id=gpt-4.1-2025-04-14
  gpt-5-mini                     provider=openai  model_id=gpt-5-mini
  gpt-5.2                        provider=openai  model_id=gpt-5.2
  llama-3.3-70b                  provider=bedrock  model_id=meta.llama3-3-70b-instruct-v1:0
  openrouter-free                provider=openrouter  model_id=openrouter/free
  qwen3-coder-30b                provider=bedrock  model_id=qwen.qwen3-coder-30b-a3b-v1:0
  qwen3-coder-480b               provider=bedrock  model_id=qwen.qwen3-coder-480b-a35b-v1:0
  qwen3-coder-free               provider=openrouter  model_id=qwen/qwen3-coder:free
  without_api                    provider=debug  model_id=without_api
```

#### 4.8 Generated Files

Each run produces the following artifacts in its output directory:

```txt
assertions_list
corrected_file.dfy
corrected_method.txt
localization/
localization_positions.json
method_with_placeholders.txt
methods.txt
selected_method.txt
timings.json
verification_errors.txt
```

These artifacts allow you to inspect:

- localization predictions
- generated assertions
- repaired programs
- verification outcomes
- timing information
- intermediate prompts and outputs

And were also generated for the complete pipeline.

# Artifact Layout

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

The paper's tables and figures are generated from notebooks under [src/research_questions/](src/research_questions/):

1. [src/research_questions/data_analysys_dataset_overview.ipynb](src/research_questions/data_analysys_dataset_overview.ipynb)
2. [src/research_questions/data_analysys_pre_tests.ipynb](src/research_questions/data_analysys_pre_tests.ipynb)
3. [src/research_questions/data_analysys_cost_statistics.ipynb](src/research_questions/data_analysys_cost_statistics.ipynb)
4. [src/research_questions/data_analysys_rq1_best_overall.ipynb](src/research_questions/data_analysys_rq1_best_overall.ipynb)
5. [src/research_questions/data_analysys_rq2_loc_strategy.ipynb](src/research_questions/data_analysys_rq2_loc_strategy.ipynb)
6. [src/research_questions/data_analysys_rq3_example_gatherer.ipynb](src/research_questions/data_analysys_rq3_example_gatherer.ipynb)
7. [src/research_questions/data_analysys_rq4_different_llms.ipynb](src/research_questions/data_analysys_rq4_different_llms.ipynb) (Not used on the final paper)


## How to Reproduce the Paper Results

The paper's figures and tables are generated from cached results, not from fresh reruns.

### Fast reproduction path

1. Extract the dataset and result tarballs.
2. Open the relevant notebook under [src/research_questions/](src/research_questions/).

To launch Jupyter inside docker: 
```sh
# Run jupyter lab
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

On the host machine find the local IP after the phrase "Or copy and paste one of those URLS". For example:
http://127.0.0.1:8888/lab?token=5da37094e6e911d7b4d01d2b71b486ef80532e7bf97eea49
(your token will differ)

Navigate to `src/research_questions` and open the notebooks for each research question.

3. Open a given research question (RQ1–RQ3) and run all cells to regenerate the results tables and figures presented in the paper.

**RQ1** (`data_analysys_rq1_best_overall.ipynb`): Generates the main comparison table and significance tests for the best overall configuration.

**RQ2** (`data_analysys_rq2_loc_strategy.ipynb`): Generates tables and significance tests for the localization strategy comparison.

**RQ3** (`data_analysys_rq3_example_gatherer.ipynb`): Generates tables and significance tests for the example-retrieval strategy comparison.

This is the recommended path for reviewers because it avoids long API-driven reruns.

### Full recomputation path

If you want to rerun the research-question pipelines, use the entry points in [src/research_questions/](src/research_questions/):

```sh
# Run the python commands
python -m src.research_questions.main_rq1
python -m src.research_questions.main_rq2
python -m src.research_questions.main_rq3
```

These runs are expensive and may take days on a modest machine. Use them only if you want to reproduce the pipeline itself rather than just the published outputs.

### Dataset recreation

If you want to recreate the dataset from scratch, the repository provides:

```sh
# Run the python command
python -m src.datasets.full_dataset_creator
```

However, this is a long-running process (approximately 2 days). On the other hand, it does not rely on LLMs, so no API costs are incurred.

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

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run the CLI as a module from the repository root:

```sh
# Run the python command
python -m src.cli <file.dfy> --model <model> --localization <strategy>
```

Do not execute the file directly as `python src/cli.py` unless you have intentionally configured `PYTHONPATH`.

### Cached data missing

If notebooks cannot find the expected inputs, make sure you extracted the tarballs by running  
```sh
# Extract dataset + results (~15GB)
./extract_saved_results_tars.sh
```
### LLM-backed runs need credentials

If you use a real model instead of a debug stub, provide the relevant API key environment variables before launching the container or export the key inside the container.

For example:
```bash
# Export a Key
export OPENROUTER_API_KEY="your-key"
```

## Expected Review Outcome

A successful review should confirm that Daisy can repair at least one Dafny example, that the cached result archives can be extracted, and that the notebooks regenerate the paper's reported tables and figures from those cached inputs.
