import tempfile
import unittest
import os 
from pathlib import Path
import utils.global_variables as gl
from datasets.dafny_dataset_generator import dafny_file_dataset_generator

def write_file(path: Path, content: str) -> Path:
    path.write_bytes(content.encode("utf-8"))
    return path

class TestAssertMethodClass(unittest.TestCase):
    def setUp(self):
        # make a temporary directory for files and datasets
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)

        #self.outputdir = tempfile.TemporaryDirectory()
        #self.output_path = Path(self.outputdir.name)

        self.outputdir = gl.TEMP_FOLDER / "test_dataset_creator"
        self.output_path = self.outputdir

        self.data_path_630: Path = gl.UNIT_TESTS_DIR / "files/dataset_creator_assertion_group_test/630-dafny_tmp_tmpz2kokaiq_Solution_dfy"
        self.data_path_assertive: Path = gl.UNIT_TESTS_DIR / "files/dataset_creator_assertion_group_test/assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy"
        self.data_path_find: Path = gl.UNIT_TESTS_DIR / "files/dataset_creator_assertion_group_test/assertive-programming-assignment-1_tmp_tmp3h_cj44u_FindRange_dfy"

    def tearDown(self):
        self.tempdir.cleanup()
        #self.outputdir.cleanup()

    def test_generate_assertion_groups_630(self):
        dafny_file_dataset_generator(gl.DAFNY_EXEC, 
                                     self.data_path_630, 
                                     self.output_path, 
                                     self.tmp_path,
                                     1)
        
        expected_dir = os.path.join(self.output_path, "630-dafny_tmp_tmpz2kokaiq_Solution_dfy")
        self.assertTrue(os.path.isdir(expected_dir), f"Expected directory {expected_dir} does not exist")
        files = os.listdir(expected_dir)
        self.assertEqual(len(files), 1, f"Expected exactly one file in {expected_dir}, found {len(files)}")
        self.assertIn("original_program.dfy", files, 
                      f"Expected 'original_program.py' in {expected_dir}, found {files}")
        print(f"Output verified in: {expected_dir}") 
        print(self.output_path)

    def test_generate_assertion_groups_assertive(self):
        dafny_file_dataset_generator(gl.DAFNY_EXEC, 
                                     self.data_path_assertive, 
                                     self.output_path, 
                                     self.tmp_path,
                                     1)
        
        expected_dir = os.path.join(self.output_path, "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy")
        self.assertTrue(os.path.isdir(expected_dir), f"Expected directory {expected_dir} does not exist")
        
        files = os.listdir(expected_dir)
        expected_assertions_helper = ["method_start_0_as_start_67_end_127", "method_start_0_as_start_96_end_125",  "method_start_1798_as_start_2028_end_2058"]

        for exp in expected_assertions_helper: 
           self.assertIn(exp, files, 
                      f"Expected folder {exp} in {expected_dir}, found {files}")

    def test_generate_assertion_groups_assertive_2(self):
        dafny_file_dataset_generator(gl.DAFNY_EXEC, 
                                     self.data_path_assertive, 
                                     self.output_path, 
                                     self.tmp_path,
                                     2)
        
        expected_dir = os.path.join(self.output_path, "assertive-programming-assignment-1_tmp_tmp3h_cj44u_SearchAddends_dfy")
        self.assertTrue(os.path.isdir(expected_dir), f"Expected directory {expected_dir} does not exist")
        
        files = os.listdir(expected_dir)
        expected_assertions_helper = ["method_start_0_as_start_67_end_127_as_start_96_end_125"]

        for exp in expected_assertions_helper: 
           self.assertIn(exp, files, 
                      f"Expected folder {exp} in {expected_dir}, found {files}")

    def test_generate_assertion_groups_assertive_unlimit(self):
        dafny_file_dataset_generator(gl.DAFNY_EXEC, 
                                     self.data_path_find, 
                                     self.output_path, 
                                     self.tmp_path,
                                     10**10)
        
        expected_dir = os.path.join(self.output_path, "assertive-programming-assignment-1_tmp_tmp3h_cj44u_FindRange_dfy")
        self.assertTrue(os.path.isdir(expected_dir), f"Expected directory {expected_dir} does not exist")
        
        files = os.listdir(expected_dir)
        expected_assertions_helper = ["method_start_0_as_start_342_end_382_as_start_386_end_419_as_start_423_end_439"]

        for exp in expected_assertions_helper: 
           self.assertIn(exp, files, 
                      f"Expected folder {exp} in {expected_dir}, found {files}")

if __name__ == "__main__":
    unittest.main()