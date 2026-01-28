import tempfile
import unittest
import json
from pathlib import Path
import utils.global_variables as gl
from datasets.dafny_dataset_generator import dafny_file_dataset_generator
from datasets.dafny_dataset_all_positions_gatherer import (
    expand_assertion_groups_with_all_syntatic_valid_positions_for_assertions,
    expand_assertion_groups_with_all_fix_positions,
    expand_assertion_groups_with_original_error_info,
    get_all_method_with_assertions_at_ind_relocated
)

def write_file(path: Path, content: str) -> Path:
    path.write_bytes(content.encode("utf-8"))
    return path

class TestAssertMethodClass(unittest.TestCase):
    def setUp(self):
        # make a temporary directory for files and datasets
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)

        # Use a test-specific subfolder under the global TEMP_FOLDER so we can inspect / remove it
        self.outputdir = gl.TEMP_FOLDER / "letssee"
        self.output_path = self.outputdir

        # this is the provided sample dataset directory under your repo's unit test fixtures
        self.data_path_assertive: Path = (
            gl.UNIT_TESTS_DIR
            / "files/dataset_creator_assertion_group_test"
            / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        )

        # ensure output parent exists
        self.outputdir.mkdir(parents=True, exist_ok=True)

        # Generate dataset artifacts into the output_path using the dafny wrapper
        dafny_file_dataset_generator(
            gl.DAFNY_EXEC,
            self.data_path_assertive,
            self.output_path,
            self.tmp_path,
            1,
        )

        # The dataset generator is expected to create a directory named after the program
        self.expected_dir = (
            self.output_path
            / "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        )

    def tearDown(self):
        # cleanup tempdir
        self.tempdir.cleanup()
        # cleanup created outputdir (if it exists)
 

    def test_generate_all_syntatic_valid_positions(self):
        expand_assertion_groups_with_all_syntatic_valid_positions_for_assertions(
            self.output_path, True
        )
        group_dir = self.expected_dir / "method_start_1798_as_start_2028_end_2058"
        json_file = group_dir / "all_lines_that_are_syntatic_valid.json"

        self.assertTrue(
            json_file.exists(),
            msg=f"Expected JSON file not found: {json_file}",
        )

        with open(json_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.assertEqual(
            data,
            [6],
            msg=f"Expected content [6] in {json_file}, got: {data}",
        )

    def test_generate_all_valid_positions(self):
        expand_assertion_groups_with_original_error_info(self.output_path,  True)

        expand_assertion_groups_with_all_fix_positions(self.output_path, True)
        group_dir = self.expected_dir / "method_start_0_as_start_67_end_127"
        json_file = group_dir / "all_lines_that_fix_file.json"

        self.assertTrue(
            json_file.exists(),
            msg=f"Expected JSON file not found: {json_file}",
        )
        with open(json_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.assertEqual(
            data,
            [[2,3]],
            msg=f"Expected content [[2,3]] in {json_file}, got: {data}",
        )
  
    def test_get_all_method_with_assertions_at_ind_relocated(self):

        methodWithoutAssertionLines = ["line 0", "line 1"]
        assertions = ["assert 1","assert 2","assert 3"] 
        assertionOracleLines = [0,0,1]
        ind = 0

        methods = get_all_method_with_assertions_at_ind_relocated( methodWithoutAssertionLines, assertions, 
                                                     assertionOracleLines, ind)
        assert (len(methods) == 2)
        assert(methods[0] == ["line 0","assert 1", "assert 2","line 1", "assert 3"])
        assert(methods[1] == ["line 0", "assert 2","line 1", "assert 1", "assert 3"])

if __name__ == "__main__":
    unittest.main()