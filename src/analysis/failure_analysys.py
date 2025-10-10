import re
import utils.global_variables as gl
import pandas as pd
from get_dataframe_from_results import get_pandas_dataset

if __name__ == '__main__':
    RESULT_DIR = gl.LLM_RESULTS_DIR
    DATASET_DIR = gl.DAFNY_ASSERTION_DATASET
    verif_data_pd = get_pandas_dataset(DATASET_DIR, RESULT_DIR)
    # Global results 
    # Uncomment to check only for a given number of assertions
    #verif_data_pd = verif_data_pd[
    #  (verif_data_pd['benchmark'] == "w/o-1") #&
    #]
    verif_data_pd = verif_data_pd[
      (verif_data_pd['number_oracle_assertions'] == 1) &
      (verif_data_pd['benchmark'] != "w/o-2 one w/o-1")  &
      (verif_data_pd['benchmark'] == "w/o-1")  
    ]

    col = list(verif_data_pd.columns)
    verif_data_pd = verif_data_pd[
    verif_data_pd["llm"].isin([
        #"gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_LLM"
        #"gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_LLM_EXAMPLE"
        "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_LAUREL_BETTER",
        #"gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_LLM_EXAMPLE"
        #"gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_ORACLE"
    ])
]
    # Failure analysys oracle assertions
    verif_data_pd  = verif_data_pd.assign(success=lambda d: d['verif_sucess'] > 0)


    
    df_pairs =  verif_data_pd.groupby(
        ['prog','group'],
        as_index=False
    ).agg(
        assertion_type=('assertion_type', lambda x: str(x.iloc[0])),
        oracle_here_would_fix=('oracle_here_would_fix', lambda x: any(x)),
        assertion_here_syntatic_valid=('assertion_here_syntatic_valid', lambda x: any(x)),
        number_expected_assertions=('number_expected_assertions', lambda x: sum(x)),
        success=('success','any')
    )

    print("To analyse")
    print(df_pairs)


    sucesses = df_pairs[df_pairs["success"] == True]
    pd.set_option('display.max_rows', None)
    print(sucesses[['prog', 'group']])
    print(len(sucesses[['prog', 'group']]))

    #failures = df_pairs[df_pairs["success"] == False]
    #pd.set_option('display.max_rows', None)
    #print(failures[['prog', 'group']])
    #print(len(failures[['prog', 'group']]))
   
    # 1) They are as expected, TEST and INDEX a lot easier to reach than the others , MULTI by far the hardest)
    verif_data_pd  = verif_data_pd.assign(success=lambda d: d['verif_sucess'] > 0)
    
    df_pairs =  verif_data_pd.groupby(
        ['llm','prog','group'],
        as_index=False
    ).agg(
        assertion_type=('assertion_type', lambda x: str(x.iloc[0])),
        oracle_here_would_fix=('oracle_here_would_fix', lambda x: any(x)),
        assertion_here_syntatic_valid=('assertion_here_syntatic_valid', lambda x: any(x)),
        number_expected_assertions=('number_expected_assertions', lambda x: sum(x)),
        success=('success','any')
    )

    # Union
    df_pairs_together =  verif_data_pd.groupby(
        ['prog','group'],
        as_index=False
    ).agg(
        assertion_type=('assertion_type', lambda x: str(x.iloc[0])),
        oracle_here_would_fix=('oracle_here_would_fix', lambda x: any(x)),
        assertion_here_syntatic_valid=('assertion_here_syntatic_valid', lambda x: any(x)),
        number_expected_assertions=('number_expected_assertions', lambda x: sum(x)),
        success=('success','any')
    )

    counts = df_pairs.groupby(["llm", 'success']).size().reset_index(name='count')
    print("Counts per assertion type:")
    print(counts)


    counts = df_pairs_together.groupby(['success']).size().reset_index(name='count')
    print("Counts per assertion type:")
    print(counts)

    
    print("Position")
    
    counts = df_pairs.groupby(["llm", 'oracle_here_would_fix']).size().reset_index(name='count')
    print("Counts position sucess individually:")
    print(counts)


    counts = df_pairs_together.groupby(['oracle_here_would_fix']).size().reset_index(name='count')
    print("Counts the conjugation of both:")
    print(counts)
 