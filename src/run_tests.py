#import tests.test_assertion_method_classes as t
#import tests.test_dataset_creator as t
#import tests.test_dataset_extensions_loc as t
#import tests.test_llm_dummy as t
import tests.test_llm_model_creation as t
import unittest

if __name__ == "__main__":
    test_instance = t.TestAllModels()
    test_instance.setUp()

    #test_instance.test_debug_and_mock_models()
    #test_instance.test_openai_models_real()
    test_instance.test_bedrock_models_real()



    #test_instance.te()# <-- replace with the failing test
    #test_instance.test_run_llm_localization_no_example()
    #test_instance.test_llm_inference_no_example()
    #test_instance.test_run_verification_on_changed_assertions()

    #test_instance.test_run_llm_localization_example()
    #test_instance.test_llm_inference_example()
    #test_instance.tearDown()

    # Load all tests from the module
    #suite = unittest.defaultTestLoader.loadTestsFromModule(t)
    
    # Run the tests
    #runner = unittest.TextTestRunner()
    #runner.run(suite)