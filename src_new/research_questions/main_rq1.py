"""RQ1: Best overall — evaluate different localization strategies.

Tests LLM, LAUREL, LAUREL_BETTER, HYBRID, ORACLE, and LLM_EXAMPLE
localization strategies with assertion inference and verification.

Three-phase pattern per (model, strategy) combo:
  1. Localization pass
  2. Assertion inference pass
  3. Verification pass

All results must be pre-cached; raises CacheMissError on miss.
"""

from __future__ import annotations

import os
from pathlib import Path

from src_new.config import (
    ASSERTION_PLACEHOLDER,
    DAFNY_ASSERTION_DATASET,
    DAFNY_EXEC,
    LLM_RESULTS_DIR,
    TEMP_FOLDER,
    AssertionInfererConfig,
    ExampleStrategy,
    LocStrategy,
    PositionInfererConfig,
    VerificationConfig,
)
from src_new.daisy.assertion_inference import LLMAssertionStrategy
from src_new.daisy.position_inference import (
    HybridPositionStrategy,
    LAURELBetterPositionStrategy,
    LAURELPositionStrategy,
    LLMExamplePositionStrategy,
    LLMPositionStrategy,
    OraclePositionStrategy,
)
from src_new.daisy.verification import ParallelComboVerification
from src_new.llm.llm_configurations import LLM
from src_new.llm.llm_create import create_llm
from src_new.research_questions import CacheMissError
from src_new.utils.assertion_method_classes import (
    assertionGroup,
    get_assertion_group_string_id,
    get_file_from_assertion_group,
    get_method_from_assertion_group,
)
from src_new.utils.dataset_class import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model_dir_name(
    llm_name: str,
    loc: LocStrategy,
    example_type: ExampleStrategy = ExampleStrategy.NONE,
    example_type_pos: ExampleStrategy = ExampleStrategy.NONE,
    num_examples: int = 0,
    example_weight: float = 0.5,
) -> str:
    """Replicate the old ``get_one_line_main_options`` naming convention."""
    s = "_"
    s += "nAssertions_ALL_"
    s += "nRounds_1_"
    s += "nRetries_1_"
    s += "addError_True_"
    if example_type != ExampleStrategy.NONE or example_type_pos != ExampleStrategy.NONE:
        s += f"addExamp_{num_examples}_"
    if example_type == ExampleStrategy.DYNAMIC or example_type_pos == ExampleStrategy.DYNAMIC:
        s += f"alpha_{example_weight}_"
    s += f"ExType_{example_type.value}_"
    s += f"loc_{loc.value}"
    return llm_name + s


def _group_cache_key(group: assertionGroup) -> str:
    """Build the cache sub-path for an assertion group: ``{prog_folder}/{group_id}``."""
    file = get_file_from_assertion_group(group)
    prog_folder = os.path.basename(file.file_path.parent.name)
    group_id = get_assertion_group_string_id(group)
    return f"{prog_folder}/{group_id}"


def _check_cache_completeness(
    groups: list[assertionGroup],
    pos_inferer,
    assert_inferer,
    phase: str,
) -> None:
    """Raise CacheMissError if any group lacks cached results."""
    if phase == "localization":
        missing = [
            _group_cache_key(g)
            for g in groups
            if not pos_inferer.check_cache(_group_cache_key(g))
        ]
    elif phase == "assertion":
        missing = [
            _group_cache_key(g)
            for g in groups
            if not assert_inferer.check_cache(_group_cache_key(g))
        ]
    else:
        return

    if missing:
        preview = missing[:10]
        raise CacheMissError(
            f"Missing {phase} cache for {len(missing)} groups: {preview}"
            + ("..." if len(missing) > 10 else ""),
            missing_entries=missing,
        )


def _prepare_method(group: assertionGroup, remove_empty_lines: bool = True):
    """Prepare method text with placeholders and full file text."""
    file = get_file_from_assertion_group(group)
    method = get_method_from_assertion_group(group)
    method_with_placeholders = method.get_method_with_assertion_group_changed(
        group, remove_empty_lines, ASSERTION_PLACEHOLDER,
    )
    _, full_file_text = file.substitute_method_with_text(method, method_with_placeholders)
    return method, method_with_placeholders, full_file_text


# ---------------------------------------------------------------------------
# Three-phase execution
# ---------------------------------------------------------------------------

def _run_localization_pass(
    groups: list[assertionGroup],
    pos_inferer,
) -> None:
    """Phase 1: localization — reads from cache."""
    for group in groups:
        key = _group_cache_key(group)
        method = get_method_from_assertion_group(group)
        pos_inferer.infer_positions(method.segment_str, "", cache_key=key)


def _run_assertion_pass(
    groups: list[assertionGroup],
    assert_inferer,
) -> None:
    """Phase 2: assertion inference — reads from cache."""
    for group in groups:
        key = _group_cache_key(group)
        _, method_with_placeholders, _ = _prepare_method(group)
        assert_inferer.infer_assertions(method_with_placeholders, "", cache_key=key)


def _run_verification_pass(
    groups: list[assertionGroup],
    pos_inferer,
    assert_inferer,
    verifier: ParallelComboVerification,
) -> list:
    """Phase 3: verification — combines cached localization + assertions."""
    results = []
    for group in groups:
        key = _group_cache_key(group)
        _, method_with_placeholders, full_file_text = _prepare_method(group)
        candidates = assert_inferer.infer_assertions(
            method_with_placeholders, "", cache_key=key,
        )
        result = verifier.verify_assertions(
            full_file_text, method_with_placeholders, candidates,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Per-strategy runner
# ---------------------------------------------------------------------------

def run_strategy(
    llm: LLM,
    loc_strategy: LocStrategy,
    groups: list[assertionGroup],
    results_dir: Path,
    dataset_path: Path,
    example_type: ExampleStrategy = ExampleStrategy.NONE,
    example_type_pos: ExampleStrategy = ExampleStrategy.NONE,
    num_examples: int = 0,
    num_examples_pos: int = 0,
    example_weight: float = 0.5,
    example_weight_pos: float = 0.5,
) -> list:
    """Run three-phase evaluation for one (model, strategy) combo."""
    model_dir_name = _build_model_dir_name(
        llm.get_name(), loc_strategy, example_type, example_type_pos,
        num_examples, example_weight,
    )
    model_dir = results_dir / model_dir_name
    os.makedirs(model_dir, exist_ok=True)

    pos_config = PositionInfererConfig(
        example_retrieval_type=example_type_pos,
        num_examples=num_examples_pos,
        example_weight=example_weight_pos,
    )
    assert_config = AssertionInfererConfig(
        example_retrieval_type=example_type,
        num_examples=num_examples,
        example_weight=example_weight,
    )
    verif_config = VerificationConfig()

    # Create inferers with cache_dir
    pos_inferer = _create_pos_inferer(
        llm, loc_strategy, pos_config, model_dir, dataset_path,
    )
    assert_inferer = LLMAssertionStrategy(
        llm=llm, config=assert_config, cache_dir=model_dir,
    )
    verifier = ParallelComboVerification(config=verif_config)

    # Check cache completeness
    _check_cache_completeness(groups, pos_inferer, assert_inferer, "localization")
    _check_cache_completeness(groups, pos_inferer, assert_inferer, "assertion")

    # Three-phase execution
    print(f"\n  Localization pass ({loc_strategy.value})...")
    _run_localization_pass(groups, pos_inferer)

    print(f"  Assertion inference pass...")
    _run_assertion_pass(groups, assert_inferer)

    print(f"  Verification pass...")
    results = _run_verification_pass(groups, pos_inferer, assert_inferer, verifier)

    verified_count = sum(1 for r in results if r.verified)
    print(f"  Done: {verified_count}/{len(results)} verified")
    return results


def _create_pos_inferer(
    llm: LLM,
    loc: LocStrategy,
    config: PositionInfererConfig,
    cache_dir: Path,
    dataset_path: Path,
):
    """Factory for position inferers with cache_dir set."""
    if loc == LocStrategy.LLM:
        return LLMPositionStrategy(llm=llm, config=config, cache_dir=cache_dir)
    if loc == LocStrategy.LLM_EXAMPLE:
        return LLMExamplePositionStrategy(llm=llm, config=config, cache_dir=cache_dir)
    if loc == LocStrategy.LAUREL:
        return LAURELPositionStrategy(config=config, cache_dir=cache_dir)
    if loc == LocStrategy.LAUREL_BETTER:
        return LAURELBetterPositionStrategy(config=config, cache_dir=cache_dir)
    if loc == LocStrategy.ORACLE:
        return OraclePositionStrategy(dataset_path=dataset_path, cache_dir=cache_dir)
    if loc == LocStrategy.HYBRID:
        laurel = LAURELBetterPositionStrategy(config=config, cache_dir=None)
        llm_inf = LLMPositionStrategy(llm=llm, config=config, cache_dir=None)
        return HybridPositionStrategy(
            laurel_better_inferer=laurel, llm_inferer=llm_inf, cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown localization strategy: {loc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """RQ1: Evaluate different localization strategies."""
    dataset = Dataset.from_dataset_assertion_groups(DAFNY_ASSERTION_DATASET)
    groups = dataset.get_all_assertion_groups()
    print(f"Loaded {len(groups)} assertion groups")

    results_dir = LLM_RESULTS_DIR
    dataset_path = DAFNY_ASSERTION_DATASET

    # --- Haiku with all localization strategies ---
    name = "claude-haiku-4.5"
    llm = create_llm(name, name)
    print(f"\n=== {name} ===")

    for loc in [
        LocStrategy.LAUREL,
        LocStrategy.LAUREL_BETTER,
        LocStrategy.ORACLE,
        LocStrategy.LLM,
    ]:
        print(f"\n--- {loc.value} ---")
        run_strategy(llm, loc, groups, results_dir, dataset_path)

    # LLM_EXAMPLE with DYNAMIC examples
    print(f"\n--- LLM_EXAMPLE ---")
    run_strategy(
        llm, LocStrategy.LLM_EXAMPLE, groups, results_dir, dataset_path,
        example_type_pos=ExampleStrategy.DYNAMIC,
        num_examples_pos=3,
        example_weight_pos=0.25,
    )

    # HYBRID with DYNAMIC examples
    print(f"\n--- HYBRID ---")
    run_strategy(
        llm, LocStrategy.HYBRID, groups, results_dir, dataset_path,
        example_type_pos=ExampleStrategy.DYNAMIC,
        num_examples_pos=3,
        example_weight_pos=0.25,
    )

    # --- Opus with selected strategies ---
    name = "claude-opus-4.5"
    llm = create_llm(name, name)
    print(f"\n=== {name} ===")

    for loc in [LocStrategy.LAUREL_BETTER, LocStrategy.ORACLE]:
        print(f"\n--- {loc.value} ---")
        run_strategy(llm, loc, groups, results_dir, dataset_path)

    print(f"\n--- LLM_EXAMPLE ---")
    run_strategy(
        llm, LocStrategy.LLM_EXAMPLE, groups, results_dir, dataset_path,
        example_type_pos=ExampleStrategy.DYNAMIC,
        num_examples_pos=3,
        example_weight_pos=0.25,
    )

    print(f"\n--- HYBRID ---")
    run_strategy(
        llm, LocStrategy.HYBRID, groups, results_dir, dataset_path,
        example_type_pos=ExampleStrategy.DYNAMIC,
        num_examples_pos=3,
        example_weight_pos=0.25,
    )


if __name__ == "__main__":
    main()
