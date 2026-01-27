from dafny.dafny_runner import *
import utils.global_variables as gl
import os
import unittest

class TestRunDafny(unittest.TestCase):
    def setUp(self):
        self.dafny_exec = gl.DAFNY_EXEC
        self.destination_path_folder = gl.TEMP_FOLDER / "dafny_test"

        test_location = gl.BASE_PATH / "src/tests/files"
        self.test_error_program = test_location / "factorial_error.dfy"
        self.test_verify_program = test_location / "factorial_verified.dfy"
        self.test_not_verified_program = test_location / "factorial_not_verified.dfy"

        if not os.path.exists(self.dafny_exec):
            self.fail(f"DAFNY_EXEC not found: {self.dafny_exec}")
        if not os.access(self.dafny_exec, os.X_OK):
            self.fail(f"DAFNY_EXEC is not executable: {self.dafny_exec}")


    def test_program_verifies(self):
        self.setUp()
        # Text without any assertions
        ret =  run_dafny(self.dafny_exec, self.test_verify_program, self.destination_path_folder, option="verify")
        (program_status, _ , _) = ret 
        print(ret)
        self.assertEqual(program_status, "VERIFIED")
    def test_program_not_verifies(self):
        # Text with a single valid assertion
        (program_status, _ , _) =  run_dafny(self.dafny_exec, self.test_not_verified_program, self.destination_path_folder, option="verify")
        self.assertEqual(program_status, "NOT_VERIFIED")
    def test_program_error(self):
        # Text with a single valid assertion
        (program_status, _ , _) =  run_dafny(self.dafny_exec, self.test_error_program, self.destination_path_folder, option="verify")
        self.assertEqual(program_status, "ERROR")
        #test_location = gl.BASE_PATH / "tests" / "assertiveProgramming_Find_substring_should_error.dfy"
        #ret =  run_dafny(gl.DAFNY_EXEC, test_location, gl.TEMP_FOLDER / "dafny_test", option="verify")
        #print(ret) 
        #test_location_2 = gl.BASE_PATH / "tests" / "assertiveProgramming_Find_substring_should_corrected_differently.dfy"
        #ret =  run_dafny(gl.DAFNY_EXEC, test_location, gl.TEMP_FOLDER / "dafny_test", option="verify")
        #print(ret) 
if __name__ == "__main__":
    unittest.main()  