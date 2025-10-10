from analysis.get_dataframe_from_results import get_pandas_dataset
import utils.global_variables as gl

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

green = "#117733"
yellow = "#DDCC77"
red = "#8A1414"

green_darker = "#0d5927"
yellow_darker = "#a39b5b"
red_darker = "#450a0a"

title_prefix = ""

def bar_chart_program_method_success_df(df: pd.DataFrame, size = "BIG", desired_order=None):
    """
    Creates a stacked bar chart per LLM showing:
    - successful pairs (verification succeeded at least once)
    - failed pairs (had something to verify but never succeeded)
    - verification_to_do (no verification existed to perform)
    """
    df = df.assign(
        success=lambda d: d['verif_sucess'] > 0,
        verif_done=lambda d: d['verif_exist'] > 0
    )
    df_pairs = (
        df.groupby(['llm', 'prog', 'group'])['success']
          .any()
          .reset_index()
    )

    summary = []
    for llm, group in df_pairs.groupby('llm'):
        success_count = group['success'].sum()
        fail_count = ((~group['success'])).sum()
        summary.append({
            'llm': llm,
            'success': success_count,
            'fail': fail_count,
        })
    agg_pairs = pd.DataFrame(summary)
    if desired_order:
        existing = set(agg_pairs['llm'])
        missing = [x for x in desired_order if x not in existing]
        if missing:
            print(f"Warning: Missing from data: {missing}")
        agg_pairs = agg_pairs.set_index('llm').reindex(desired_order).reset_index()
   
    labels = agg_pairs['llm'].tolist()
    success = agg_pairs['success'].tolist()
    fail = agg_pairs['fail'].tolist()

    print(labels)
    print(success)

    x = range(len(labels))

    if(size == "SINGLE"):
        fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)
    elif(size == "DOUBLE"):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    elif(size == "BIG"):
        fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(x, success, label='Verify', color=green)
    ax.bar(x, fail, bottom=success, label='Fail', color=red)

    for i, (s, f) in enumerate(zip(success, fail)):
        total = s + f 
        print(total)
        ax.text(i, total + 0.8, f"{total}", ha='center')

    for i, (s, f) in enumerate(zip(success, fail)):
        total = s + f
        if total == 0:
            continue  
        pct_s = s / total * 100
        if(s!=0 and pct_s>5):
          ax.text(
            i,               
            s / 2,           
            f"{pct_s:.0f}%",  
            ha='center',
            va='center',
            color='white',
        )

        pct_f = f / total * 100
        if(f != 0 and pct_f>5):
          ax.text(
            i,
            s + f / 2,      
            f"{pct_f:.0f}%",
            ha='center',
            va='center',
            color='white',
          )

    ax.set_ylabel('Program/Assertion Group Pairs')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01,1))

    plt.tight_layout()
    plt.savefig(title_prefix + "bar_chart_program_method_success.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    #plt.show()

import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  
def get_expected_pass_at_k_by_llm(df: pd.DataFrame) -> dict[str, dict[int, int]]:
    """
    For each (llm, program_folder, assertion_group), compute:
        total_assertions = number of rows
        successes = sum(verified > 0)
        expected_k = ceil(total_assertions / successes)
    Then build a histogram per LLM: key = expected_k, value = count of groups.
    """
    df = df.assign(success=lambda d: d['verif_sucess'] > 0)

    grp = (
        df.groupby(['llm', 'prog', 'group'])
          .agg(
              total_assertions=('success', 'size'), 
              successes=('success', 'sum')   
          )
          .reset_index()
    )
    grp = grp[grp['successes'] > 0]
    grp['k'] = (grp['total_assertions'] / grp['successes']).apply(np.ceil).astype(int)
    histo = {}
    for llm, sub in grp.groupby('llm'):
        counts = sub['k'].value_counts().to_dict()
        histo[llm] = dict(sorted(counts.items()))
    return histo

def line_plot_expected_kpass_df(df: pd.DataFrame, size="BIG"):
    histo = get_expected_pass_at_k_by_llm(df)
    """
    Plots cumulative number of groups “fixed” at or before expected_k for each LLM.
    """
    max_k = max((max(d.keys()) for d in histo.values()), default=0)
    if(size == "SINGLE"):
        fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)
    elif(size == "DOUBLE"):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    elif(size == "BIG"):
        fig, ax = plt.subplots(figsize=(14, 7))

    for llm, bucket in histo.items():
        xs = list(range(1, max_k + 1))
        ys = []
        cum = 0
        for k in xs:
            cum += bucket.get(k, 0)
            ys.append(cum)
        ax.plot(xs, ys, marker='o', label=llm)

    ax.set_xlabel('Expected k (total_assertions / successes)')
    ax.set_ylabel('Program/Assertion Group Pairs')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01,1))
    plt.tight_layout()
    plt.savefig(title_prefix + "line_plot_expected_kpass.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    #plt.show()

def bar_chart_fix_position_analysis_df(df: pd.DataFrame, size="BIG", desired_order=None, localization="" ):
    """
    Stacked bar chart per LLM categorizing program/assertion_group pairs by fix position type:
      - 'Valid Position': found position intersects oracle positions (green)
      - 'Syntactic Valid but Invalid For Oracle Assertion': found intersects syntactic but not possible options (yellow)
      - 'Syntatic Invalid Position': found not in any valid positions (red)
    Adds percentages in the middle of each stack.
    """
    df_pairs = df.groupby(
        ['llm','prog','group'],
        as_index=False
    ).agg(
        oracle_here_would_fix=('oracle_here_would_fix', lambda x: any(x)),
        assertion_here_syntatic_valid=('assertion_here_syntatic_valid', lambda x: any(x)),
        number_expected_assertions=('number_expected_assertions', lambda x: sum(x)),
    )

    def classify(row):
        if(row['number_expected_assertions'] == 0):
            return "No Pos"
        elif row['oracle_here_would_fix']:
            #return 'Valid Position'
            return "Valid"
        elif row['assertion_here_syntatic_valid']:
            #return 'Syntactic Valid but Invalid For Oracle Assertion'
            return "Partial"
        else:
            #return 'Syntatic Invalid Position'
            return "Invalid"

    df_pairs['category'] = df_pairs.apply(classify, axis=1)

    agg = df_pairs.pivot_table(
        index='llm',
        columns='category',
        aggfunc='size',
        fill_value=0
    )

    cats   = ['Valid',
              'Partial',
              'Invalid',
              'No Pos']
    
    colors = [green, 
              yellow, 
              red,
              '#777777'] 


    agg = agg.reindex(columns=cats, fill_value=0)

    if desired_order is not None:
        missing = [x for x in desired_order if x not in agg.index]
        if missing:
            print(f"Warning: these LLMs not found in data and will be skipped: {missing}")
        agg = agg.reindex(desired_order, fill_value=0)

    labels = agg.index.tolist()
    data   = [agg[cat].tolist() for cat in cats]

    x = range(len(labels))
    if(size == "SINGLE"):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    elif(size == "DOUBLE"):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    elif(size == "BIG"):
        fig, ax = plt.subplots(figsize=(14, 7))

    bottom = [0] * len(labels)

    totals = agg.sum(axis=1).tolist()

    for cat, vals, color in zip(cats, data, colors):
        ax.bar(x, vals, bottom=bottom, label=cat, color=color)
        for i, (b, v, tot) in enumerate(zip(bottom, vals, totals)):
            pct = v / tot * 100
            if v > 0 and pct>5:

                ax.text(
                    i,               
                    b + v / 2,     
                    f'{pct:.0f}%', 
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='white'
                )
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_ylabel('Testcases')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha='center')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01,1))
    plt.tight_layout()
    plt.savefig(localization , dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)



def filter_df(df: pd.DataFrame,
              name_contains: List[str] = [], remove_matches=1
             ) -> pd.DataFrame:
    """
    Return a filtered DataFrame where:
      - llm_names containing `name_contains` are removed (if name_contains != "")
      - if name_contains is empty, returns the original df
    """
    if not name_contains:
        return df

    pattern = "|".join(re.escape(s) for s in name_contains)
    if(remove_matches):
        mask = ~df['llm'].str.contains(pattern, regex=True)
    else:
        mask = df['llm'].str.contains(pattern, regex=True)
    return df[mask]

def bar_chart_dual_success_fix(df: pd.DataFrame,
                               size: str = "BIG",
                               width: float = 0.35,
                               desired_order: list[str] = None,
                               localization =""
                               ):
    """
    For each LLM, draw two bars:
      • Left:   overall Success / Fail (2‐segment stack, with % inside).
      • Right:  Success split by fix‑position (4‐segment), then Fail likewise.
    """
    df = df.assign(success=lambda d: d['verif_sucess'] > 0)
    agg = (
        df.groupby(['llm','prog','group'], as_index=False)
          .agg(
            success=('success','any'),
            oracle_fix=('oracle_here_would_fix','any'),
            synt_valid=('assertion_here_syntatic_valid','any'),
            num_expect=('number_expected_assertions','sum'),
          )
    )
    def cls(r):
        if r.num_expect == 0: return 'No Pos'
        if r.oracle_fix:      return 'Valid'
        if r.synt_valid:      return 'Partial'
        return 'Invalid'
    agg['fix_cat'] = agg.apply(cls, axis=1)

    cats   = ['Valid','Partial','Invalid','No Pos']
    pivot = (
        agg.pivot_table(
            index='llm',
            columns=['success','fix_cat'],
            aggfunc='size',
            fill_value=0
        )
        .reindex(
            pd.MultiIndex.from_product([[True,False],cats],
                                       names=['success','fix_cat']),
            axis=1, fill_value=0
        )
    )

    if desired_order is not None:
        missing = [x for x in desired_order if x not in pivot.index]
        if missing:
            print(f"Warning: these LLMs not in data and will be skipped: {missing}")
        pivot = pivot.reindex(desired_order, fill_value=0)
    

    llms     = pivot.index.tolist()
    succ_df  = pivot.xs(True,  level='success', axis=1)
    fail_df  = pivot.xs(False, level='success', axis=1)
    succ_tot = succ_df.sum(axis=1).tolist()
    fail_tot = fail_df.sum(axis=1).tolist()
    if(size == "SINGLE"):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    elif size=='DOUBLE':
        fig, ax = plt.subplots(figsize=(7,4), dpi=300)
    else:
        fig, ax = plt.subplots(figsize=(14,7))

    x = range(len(llms))
    lx = [xi - width/2 for xi in x]
    rx = [xi + width/2 for xi in x]

    h_s = ax.bar(lx, succ_tot, width, color=green)
    h_f = ax.bar(lx, fail_tot, width, bottom=succ_tot, color=red)

    for xi, s, f in zip(lx, succ_tot, fail_tot):
        tot = s + f
        s1 = s/tot*100
        f1 = f/tot*100
        if s and s1>5:
            ax.text(xi, s/2,   f"{s1:.0f}%", ha='center', va='center', color='white', fontsize=6)
        if f and f1>5:
            ax.text(xi, s + f/2, f"{f1:.0f}%", ha='center', va='center', color='white', fontsize=6)

    colors = {'Valid':green_darker,'Partial':yellow_darker,'Invalid':red_darker,'No Pos':'#777777'}

    bottom = [0]*len(llms)
    fix_handles = {}
    for cat in cats:
        vals = succ_df[cat].tolist()
        bars = ax.bar(rx, vals, width, bottom=bottom, color=colors[cat])
        fix_handles[cat] = bars[0]
        for xi, b, v, st in zip(rx, bottom, vals, succ_tot):
            p1 = v/st*100
            if v and p1>5:
                ax.text(xi, b + v/2, f"{p1:.0f}%", ha='center', va='center', color='white', fontsize=6)
        bottom = [b+v for b,v in zip(bottom, vals)]

    for cat in cats:
        vals = fail_df[cat].tolist()
        bars = ax.bar(rx, vals, width, bottom=bottom, color=colors[cat])
        for xi, b, v, ft in zip(rx, bottom, vals, fail_tot):
            p2 = v/ft*100
            if v and p2>5:
                ax.text(xi, b + v/2, f"{v/ft*100:.0f}%", ha='center', va='center', color='white', fontsize=6)
        bottom = [b+v for b,v in zip(bottom, vals)]

    leg1 = ax.legend(
        handles=[h_s, h_f],
        labels=['Verify','Fail'],
        title='Outcome',
        loc='upper left', bbox_to_anchor=(1.01,1))
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=[fix_handles[c] for c in cats],
        labels=cats,
        title='Fix Position',
        loc='lower left', bbox_to_anchor=(1.01,0))

    ax.set_xticks(x)
    ax.set_xticklabels(llms, rotation=10, ha='center')
    ax.set_ylabel('Testcases')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(localization , dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)



def bar_chart_cleaned(verif_data_pd, size, llms_to_plot):
    new_verif_data_pd = filter_df(verif_data_pd, llms_to_plot.keys(), remove_matches=0).copy()
    new_verif_data_pd["llm"] = new_verif_data_pd["llm"].apply(lambda x: llms_to_plot[x])
    bar_chart_program_method_success_df(new_verif_data_pd, size, desired_order=list(llms_to_plot.values()))


def line_plot_expected_kpass_df_cleaned(verif_data_pd, size, llms_to_plot):
    new_verif_data_pd = filter_df(verif_data_pd, llms_to_plot.keys(), remove_matches=0).copy()    
    new_verif_data_pd["llm"] = new_verif_data_pd["llm"].apply(lambda x: llms_to_plot[x])
    line_plot_expected_kpass_df(new_verif_data_pd, size)

def bar_chart_fix_position_cleaned(verif_data_pd, size, llms_to_plot, localization):
    new_verif_data_pd = filter_df(verif_data_pd, llms_to_plot.keys(), remove_matches=0).copy()
    new_verif_data_pd["llm"] = new_verif_data_pd["llm"].apply(lambda x: llms_to_plot[x])
    bar_chart_fix_position_analysis_df(new_verif_data_pd, size,desired_order=list(llms_to_plot.values()), localization=localization)


def sucess_vs_position_cleaned(verif_data_pd, size, llms_to_plot, localization):
    new_verif_data_pd = filter_df(verif_data_pd, llms_to_plot.keys(), remove_matches=0).copy()    
    new_verif_data_pd["llm"] = new_verif_data_pd["llm"].apply(lambda x: llms_to_plot[x])
    bar_chart_dual_success_fix(new_verif_data_pd, size, desired_order=list(llms_to_plot.values()), localization=localization)

if __name__ == '__main__':
    RESULT_DIR = gl.LLM_RESULTS_DIR
    #RESULT_DIR = gl.BASE_PATH / "dafny_llm_results_dataset_multi_generate_1_or_2"
    DATASET_DIR = gl.DAFNY_ASSERTION_DATASET
    #DATASET_DIR = gl.BASE_PATH / "dafny_assertion_dataset_multi_generate_1_or_2"
    verif_data_pd = get_pandas_dataset(DATASET_DIR, RESULT_DIR)

    # Uncomment to get results for a given number of assertions

    #verif_data_pd = verif_data_pd[verif_data_pd['llm'].str.contains("stub", na=False)]
    verif_data_pd = verif_data_pd[
      (verif_data_pd['benchmark'] != "w/o-2 one w/o-1") 
    ]
    col = list(verif_data_pd.columns)
    #verif_data_pd = filter_df(verif_data_pd, name_contains=["stub", "without"])
    #verif_data_pd = verif_data_pd[verif_data_pd["llm"] ==  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_LLM_EXAMPLE"]
    #verif_data_pd = verif_data_pd[verif_data_pd["llm"] ==  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_LLM"]
    #verif_data_pd = verif_data_pd[verif_data_pd["llm"] ==  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_ORACLE"]
    #verif_data_pd = verif_data_pd[verif_data_pd["llm"] ==  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_LAUREL_BETTER"]
    #verif_data_pd = verif_data_pd[verif_data_pd["assertion_type"].apply(len) == 0]
    #verif_data_pd = verif_data_pd[
    #  (verif_data_pd['number_expected_assertions'] == 1) &
    #  (verif_data_pd['number_oracle_assertions'] == 1)
    #]

    verif_data_pd = verif_data_pd[
      (verif_data_pd['number_oracle_assertions'] == 1)
    ]

    verif_data_pd  = verif_data_pd.assign(success=lambda d: d['verif_sucess'] > 0) 
    
    df_pairs =  verif_data_pd.groupby(
        ['llm','prog','group'],
        as_index=False
    ).agg(
        assertion_type=('assertion_type', lambda x: str(x.iloc[0])),
        oracle_here_would_fix=('oracle_here_would_fix', lambda x: any(x)),
        assertion_here_syntatic_valid=('assertion_here_syntatic_valid', lambda x: any(x)),
        number_expected_assertions=('number_expected_assertions', lambda x: x.iloc[0]),
        number_oracle_assertions=('number_oracle_assertions', lambda x: x.iloc[0]),
        success=('success','any')
    )

    #print(df_pairs)
    counts = df_pairs.groupby(["llm", "assertion_type", 'success']).size().reset_index(name='count')
    print("Counts per assertion type:")
    print(counts)

    counts = df_pairs.groupby(["llm", "number_oracle_assertions", 'number_expected_assertions','success' ]).size().reset_index(name='count')
    print(counts)
    # LLM example  0.5                                             llm      assertion_type  success  count
#0   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'INDEX']     True      3
#1   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'OTHER']     True      3
#2   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['INDEX']    False     31
#3   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['INDEX']     True     37
#4   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['MULTI', 'OTHER']     True      1
#5   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['MULTI']    False      8
#6   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['MULTI']     True      3
#7   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'INDEX']     True      3
#8   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'MULTI']    False      4
#9   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'OTHER']    False      4
#10  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'OTHER']     True     25
#11  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['OTHER']    False     67
#12  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['OTHER']     True     59
#13  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...            ['TEST']    False     10
#14  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...            ['TEST']     True     23
#15  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...                  []    False     23
#16  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...                  []     True      7
                    
    # ORACLE 0.5                                                llm      assertion_type  success  count
#0   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'INDEX']    False      1
#1   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'INDEX']     True      2
#2   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'OTHER']     True      3
#3   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['INDEX']    False     17
#4   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['INDEX']     True     51
#5   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['MULTI', 'OTHER']    False      1
#6   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['MULTI']    False     10
#7   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['MULTI']     True      1
#8   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'INDEX']    False      2
#9   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'INDEX']     True      1
#10  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'MULTI']    False      1
#11  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'MULTI']     True      3
#12  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'OTHER']    False      5
#13  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'OTHER']     True     24
#14  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['OTHER']    False     48
#15  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['OTHER']     True     78
#16  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...            ['TEST']    False      5
#17  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...            ['TEST']     True     28
#18  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...                  []    False     20
#19  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...                  []     True     10  
   
# gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_LAUREL_BETTER 
# STATIC 0.5                                                 llm      assertion_type  success  count
#0   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'INDEX']    False      1
#1   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'INDEX']     True      2
#2   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'OTHER']    False      1
#3   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['INDEX', 'OTHER']     True      2
#4   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['INDEX']    False     28
#5   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['INDEX']     True     40
#6   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['MULTI', 'OTHER']     True      1
#7   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['MULTI']    False      9
#8   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['MULTI']     True      2
#9   gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'INDEX']     True      3
#10  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'MULTI']    False      3
#11  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'MULTI']     True      1
#12  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'OTHER']    False      7
#13  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...  ['OTHER', 'OTHER']     True     22
#14  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['OTHER']    False     66
#15  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...           ['OTHER']     True     60
#16  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...            ['TEST']    False     12
#17  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...            ['TEST']     True     21
#18  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...                  []    False     24
#19  gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_...                  []     True      6

    for op in ["1_assertion_", "2_assertion_", "more_than_2_", "all_assertion_"]:
      if(op == "1_assertion_"):
        plot_data_pd = verif_data_pd[
         (verif_data_pd['number_oracle_assertions'] == 1) 
         ]
      elif(op == "2_assertion_"):
        plot_data_pd = verif_data_pd[
         (verif_data_pd['number_oracle_assertions'] == 2) 
         ]
      elif(op == "more_than_2_"):
        plot_data_pd = verif_data_pd[
         (verif_data_pd['number_oracle_assertions'] > 2) 
         ]
      else:
        plot_data_pd = verif_data_pd

      op = "images/" + op
      llms_to_plot={ 
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_LLM_EXAMPLE" : "LLM_EX_NO_RAG",
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_LAUREL_BETTER" : "STATIC_RAG",
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_LLM_EXAMPLE" : "LLM_EX_RAG", 
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_ORACLE" : "ORACLE_RAG" ,                         
                  }
    

        
      title_prefix = op + "_best_overall_"
      print(title_prefix)
      bar_chart_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)
      line_plot_expected_kpass_df_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)
      bar_chart_fix_position_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)
      sucess_vs_position_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)


   
      llms_to_plot={
                  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_ExType_RANDOM_loc_ORACLE" : "random",
                  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_ORACLE" : "no example",
                  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.25_ExType_DYNAMIC_loc_ORACLE" : "alpha_0.25",
                  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_1_ExType_DYNAMIC_loc_ORACLE" : "alpha_1",
                  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.75_ExType_DYNAMIC_loc_ORACLE" : "alpha_0.75",
                  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_ExType_TFIDF_loc_ORACLE"   : "tfidf",  
                  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_alpha_0.5_ExType_DYNAMIC_loc_ORACLE" : "alpha_0.50" ,   
                  "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_3_ExType_EMBEDDED_loc_ORACLE" : "EMBEDDED"
                                                                  
                  }
    
      title_prefix = op + "_evaluating_examples_retrieval_with_fixed_oracle_position_"
      print(title_prefix)
      bar_chart_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)
      line_plot_expected_kpass_df_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)
      bar_chart_fix_position_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)
      sucess_vs_position_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)

      llms_to_plot={ 
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_LAUREL" : "Laurel$_{fl}$",
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_LLM" : "Llm$_{fl}$", 
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_LAUREL_BETTER" : "Laurel$_{fl+}$",
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_LLM_EXAMPLE" : "LlmEx$_{fl}$",
                 "gpt_4.1__nAssertions_ALL_nRounds_1_nRetries_1_addError_1_addExamp_0_ExType_NONE_loc_ORACLE" : "GrTru$_{fl}$",               
                  }
    
      title_prefix = op + "_evaluating_position_inference_without_examples_"
      print(title_prefix)
      bar_chart_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)
      line_plot_expected_kpass_df_cleaned(plot_data_pd,"DOUBLE",   llms_to_plot)
      bar_chart_fix_position_cleaned(plot_data_pd,"SINGLE",   llms_to_plot)
      sucess_vs_position_cleaned(plot_data_pd,"SINGLE",   llms_to_plot)
      




