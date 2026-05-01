"""Unit tests for example retrieval helpers."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ExampleStrategy, PositionInfererConfig
from src.llm.retrieve_examples import retrieve_examples


class TestRetrieveExamples:
    def test_retrieve_examples_uses_retrieval_and_limits_results(self):
        cfg = PositionInfererConfig(
            example_retrieval_type=ExampleStrategy.DYNAMIC,
            num_examples=1,
            example_weight=0.25,
        )

        entries = [{"id": 1}, {"id": 2}]
        retrieved = [{"example": "first"}, {"example": "second"}]

        with patch("src.llm.retrieve_examples.generate_example_model") as mock_model, patch(
            "src.llm.retrieve_examples.retrieve_by_error_and_code",
            return_value=retrieved,
        ) as mock_retrieve, patch(
            "src.llm.retrieve_examples.extract_error_blocks",
            return_value="filtered error",
        ) as mock_extract:
            mock_model.return_value = (entries, MagicMock(), "cpu", "tfidf_vec", "tfidf_mat")

            result = retrieve_examples(
                cfg,
                "method Foo() {}",
                "error: assertion might not hold",
                "Prog",
                "Group",
            )

        assert result == retrieved[:1]
        mock_extract.assert_called_once_with("error: assertion might not hold")
        mock_retrieve.assert_called_once_with(
            new_error="filtered error",
            new_code="method Foo() {}",
            entries=entries,
            top_k=-1,
            method=ExampleStrategy.DYNAMIC,
            α=0.25,
            prog_original="Prog",
            group_original="Group",
            model=mock_model.return_value[1],
            device="cpu",
            diferent_methods=1,
            tfidf_vectorizer="tfidf_vec",
            tfidf_matrix="tfidf_mat",
        )