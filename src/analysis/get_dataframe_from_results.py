import re
import utils.global_variables as gl
from llm.parse_raw_response import parse_raw_response

from pathlib import Path
from typing import Optional
import json
from typing import List, Dict, Any
import pandas as pd

RESULT_DIR = gl.LLM_RESULTS_DIR
DATASET_DIR = gl.DAFNY_ASSERTION_DATASET

def _read_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _parse_localization(localization_file) -> List[int]:
    raw = _read_file(localization_file)
    if raw is None:
        return []
    try:
        return parse_raw_response(raw)
    except Exception:
        return []

def _parse_assertion_dir(assertion_dir):
    verif_text = _read_file(assertion_dir)
    result = {
        'verified': 0,
        'verification_errors': 0,
        'resolution_errors': 0,
        'parse_errors': 0,
        'time_out_errors': 0,
        'did_not_finish': 1
    }
    try:
      lines = verif_text.splitlines()
    except:
        return result
    for line in lines[-3:]:
        if "ERROR SKIPPED VERIFICATION" in line:
            result.update({'verification_errors': 1, 'did_not_finish': 0})
            break
        if m := re.search(r"(\d+) verified, (\d+) errors, (\d+) time out", line):
            result.update({
                'verified': int(m.group(1)),
                'verification_errors': int(m.group(2)),
                'time_out_errors': int(m.group(3)),
                'did_not_finish': 0
            })
            break
        if m := re.search(r"(\d+) verified, (\d+) error", line):
            result.update({
                'verified': int(m.group(1)),
                'verification_errors': int(m.group(2)),
                'did_not_finish': 0
            })
            break
        if m := re.search(r"(\d+) resolution/type errors detected", line):
            result.update({'resolution_errors': int(m.group(1)), 'did_not_finish': 0})
            break
        if m := re.search(r"(\d+) parse errors detected", line):
            result.update({'parse_errors': int(m.group(1)), 'did_not_finish': 0})
            break

    result['verif_sucess'] = not any([
        result['verification_errors'],
        result['resolution_errors'],
        result['parse_errors'],
        result['time_out_errors'],
        result['did_not_finish']
    ])
    return result

import re
def retrieve_information_from_dataset_mod(dataset_dir):
    # results: Dict[ExperimentKey, Dict[str, Any]] = {}
    rows = []
    pattern = re.compile(r"method_start_(\d+)_as_start_(\d+)_end_(\d+)")
    pattern2 = re.compile(r"method_start_(\d+)_as_start_(\d+)_end_(\d+)_as_start_(\d+)_end_(\d+)")
    for prog_dir in dataset_dir.iterdir():
        if not prog_dir.is_dir():
            continue
        # identified w/o-1 assertions
        without_1_assert = {}
        without_1_assert_name = {}
        for grp_dir in prog_dir.iterdir():
            if not grp_dir.is_dir() or grp_dir.name in ("bin", "obj"):
                continue
            match = pattern.fullmatch(grp_dir.name)
            if match:         
                method_start, assertion_start, assertion_end = map(int, match.groups())
                without_1_assert_name[grp_dir.name] = "w/o-1"
                without_1_assert.setdefault(method_start, []).append((assertion_start, assertion_end))

        # identified w/o-2 cases and named them 
        without_2_assert_name = {}
        for grp_dir in prog_dir.iterdir():
            if not grp_dir.is_dir() or grp_dir.name in ("bin", "obj"):
                continue
            match = pattern2.fullmatch(grp_dir.name)
            if match:         
                method_start, assertion1_start, assertion1_end, assertion2_start, assertion2_end  = map(int, match.groups())
                assert1_in_wo1 = False
                assert2_in_wo1 = False
                if(method_start in without_1_assert):
                    for (assert_test_start, assert_test_end) in without_1_assert[method_start]:
                        if(assert_test_start == assertion1_start and assert_test_end == assertion1_end):
                            assert1_in_wo1 = True
                        if(assert_test_start == assertion2_start and assert_test_end == assertion2_end):              
                            assert2_in_wo1 = True
                if(assert1_in_wo1 and assert2_in_wo1):
                    without_2_assert_name[grp_dir.name] = "w/o-2 both w/o-1"
                elif( (not assert1_in_wo1) and (not assert2_in_wo1)):
                    without_2_assert_name[grp_dir.name] = "w/o-2 none w/o-1" 
                else:
                    without_2_assert_name[grp_dir.name] = "w/o-2 one w/o-1" 


        for grp_dir in prog_dir.iterdir():
            if not grp_dir.is_dir() or grp_dir.name in ("bin", "obj"):
                continue
            row = {}
            row["prog"] = prog_dir.name

            row["group"] = grp_dir.name  
            # This is a json containing a python list
            try:
              with open(grp_dir / "all_lines_that_are_syntatic_valid.json", 'r', encoding='utf-8') as f:
                row["all_syntatic_valid_lines"] = json.load(f)
            except: # Case for all assertions
                row["all_syntatic_valid_lines"] = []

            try:
              with open(grp_dir / "all_lines_that_fix_file.json", 'r', encoding='utf-8') as f:
                row["all_lines_where_oracle_fixes_file"] = json.load(f)
            except: # Case for all assertions
                row["all_lines_where_oracle_fixes_file"] = []

            try:
              with open(grp_dir / "manual_assertions_type.json", 'r', encoding='utf-8') as f:
                row["assertion_type"] = json.load(f)
            except:
                row["assertion_type"] = []

            if(row["group"] in  without_1_assert_name):
                row["benchmark"] = without_1_assert_name[row["group"]]
            elif(row["group"] in  without_2_assert_name):
                row["benchmark"] = without_2_assert_name[row["group"]]
            else:
                row["benchmark"] = "w/o-all"
            rows.append(row.copy())
            row = {}
    return rows


def retrieve_information_from_dataset(dataset_dir):
    # results: Dict[ExperimentKey, Dict[str, Any]] = {}
    rows = []
    for prog_dir in dataset_dir.iterdir():
        if not prog_dir.is_dir():
            continue
        row = {}
        row["prog"] = prog_dir.name
        for grp_dir in prog_dir.iterdir():
            if not grp_dir.is_dir() or grp_dir.name in ("bin", "obj"):
                continue

            row["group"] = grp_dir.name  
            # This is a json containing a python list
            try:
              with open(grp_dir / "all_lines_that_are_syntatic_valid.json", 'r', encoding='utf-8') as f:
                row["all_syntatic_valid_lines"] = json.load(f)
            except:
                row["all_syntatic_valid_lines"] = []
            try:
              with open(grp_dir / "all_lines_that_fix_file.json", 'r', encoding='utf-8') as f:
                row["all_lines_where_oracle_fixes_file"] = json.load(f)
            except: # Case for all assertions
                row["all_lines_where_oracle_fixes_file"] = []

            try:
              with open(grp_dir / "manual_assertions_type.json", 'r', encoding='utf-8') as f:
                row["assertion_type"] = json.load(f)
            except:
                row["assertion_type"] = []
            rows.append(row.copy())
    return rows

def retrieve_information_from_results(results_dir: Path):
    # results: Dict[ExperimentKey, Dict[str, Any]] = {}
    rows = []
    for llm_dir in results_dir.iterdir():
        if not llm_dir.is_dir():
            continue
        row = {}
        row["llm"] = llm_dir.name
        for prog_dir in llm_dir.iterdir():
            if not prog_dir.is_dir():
                row = {}
                row["llm"] = llm_dir.name
                continue
            row["prog"] = prog_dir.name
            for grp_dir in prog_dir.iterdir():
                if not grp_dir.is_dir():
                    row = {}
                    row["llm"] = llm_dir.name
                    row["prog"] = prog_dir.name
                    continue
                row["group"] = grp_dir.name    
                verif_root = grp_dir / 'verification'
                loc_root   = grp_dir / 'localization'

                save_verif_exist = verif_root.exists()
                save_local_exist  = loc_root.exists()
                save_localization = _parse_localization(loc_root / "localization_raw_response.txt")

                row["verif_exist"] = save_verif_exist
                row["local_exist"] = save_local_exist
                row["localization"] = save_localization

                info_names = set()
                if not verif_root.exists():
                    rows.append(row.copy()) # Row without verification appended
                    row = {}
                    row["llm"] = llm_dir.name
                    row["prog"] = prog_dir.name
                    row["group"] = grp_dir.name    
                    continue
                elif(row["localization"] == []): # If not localization given means than there is nothing to verify
                    rows.append(row.copy()) # Row without verification appended
                    row = {}
                    row["llm"] = llm_dir.name
                    row["prog"] = prog_dir.name                    
                    row["group"] = grp_dir.name  
                    continue                     
                else:
                    info_names |= {d.name for d in verif_root.iterdir() if d.is_dir()}
            
                for assertion_str in info_names:
                    row.update(_parse_assertion_dir(verif_root / assertion_str / "verif_stdout.txt" ))
                    rows.append(row.copy())
                    row = {}
                    row["llm"] = llm_dir.name
                    row["prog"] = prog_dir.name
                    row["verif_exist"] = save_verif_exist
                    row["local_exist"] = save_local_exist
                    row["localization"] = save_localization
                    row["group"] = grp_dir.name          
    return rows


def merge_dataset_and_results(dataset_info, results_info):
    """
    Joins the two lists on matching (prog, group) keys.
    Only merges when the key exists in both.
    """
    # Build lookup by (prog, group)
    lookup = {(r["prog"], r["group"]): r for r in dataset_info}
    
    merged = []
    for res in results_info:
        key = (res["prog"], res["group"])
        if key in lookup:  # only merge if both exist
            ds = lookup[key]
            combined = {**ds, **res}
            merged.append(combined)
    return merged

def oracle_here_would_fix(n_oracle,found_fixes_position, all_options_fixes_position):
    n_found_fixes = len(found_fixes_position)
    if(all_options_fixes_position == []):
        return False
    
    if(n_oracle == 1 and  n_found_fixes == 1):
        pos = found_fixes_position[0]
        return pos in all_options_fixes_position
    if(n_oracle == 2 and  n_found_fixes == 2):
        pos0 = found_fixes_position[0]
        pos1 = found_fixes_position[1]
        if pos0 in all_options_fixes_position[0] and pos1 in all_options_fixes_position[1]:
           return True
    
        return False
    # These are partial true though (i will insert to be fair the comparation!)
    if(n_oracle == 1 and  n_found_fixes == 2):
        pos0 = found_fixes_position[0]
        pos1 = found_fixes_position[1]
        # Compare both as it is only necessary one to be in the correct posiiton in true
        return (pos0 in all_options_fixes_position) or (pos1 in all_options_fixes_position)
    if(n_oracle == 2 and  n_found_fixes == 1):
          pos = found_fixes_position[0]
          return pos in all_options_fixes_position[0] or pos in all_options_fixes_position[1]
    
def assertion_here_syntatic_valid( n_oracle,found_fixes_position, all_syntatic_fixes_position):
    n_found_fixes = len(found_fixes_position)

    if(n_oracle == 1 and  n_found_fixes == 1):
        pos = found_fixes_position[0]
        return pos in all_syntatic_fixes_position
    if(n_oracle == 2 and  n_found_fixes == 2):
        pos0 = found_fixes_position[0]
        pos1 = found_fixes_position[1]
        return pos0 in all_syntatic_fixes_position or pos1 in all_syntatic_fixes_position
    # These are partial true though
    if(n_oracle == 1 and  n_found_fixes == 2):
        pos0 = found_fixes_position[0]
        pos1 = found_fixes_position[1]
        # As I tru multiple assertion only need one to be syntatic balid
        return pos0 in all_syntatic_fixes_position or pos1 in all_syntatic_fixes_position
    if(n_oracle == 2 and  n_found_fixes == 1):
        pos = found_fixes_position[0]
        return pos in all_syntatic_fixes_position
    

def create_pandas_dataset_and_expand_it_with_computed_data(
    merged_rows: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Given merged_rows from merge_dataset_and_results(), compute additional
    statistics and return a pandas DataFrame.
    """
    expanded = []
    for row in merged_rows:
        # pull in the raw lists (they may be missing)
        all_options = row['all_lines_where_oracle_fixes_file']
        localization_positions = row['localization']
        all_syntatic = row['all_syntatic_valid_lines']
 
        n_oracle = row["group"].count("start")-1

        # 2) Number of found fixes
        n_found = len(localization_positions)

        # 3) Compute the two custom flags
        oracle_fix = oracle_here_would_fix(
            n_oracle,
            localization_positions,
            all_options
        )
        syntactic = assertion_here_syntatic_valid(
            n_oracle,
            localization_positions,
            all_syntatic
        )

        # 4) Tally up and attach to a copy of the row
        new_row = dict(row)  # shallow copy
        new_row.update({
            'number_oracle_assertions': n_oracle,
            'number_expected_assertions': n_found,
            'oracle_here_would_fix':     oracle_fix,
            'assertion_here_syntatic_valid': syntactic,
            "position_valid" : oracle_fix,
            "position_partial" : syntactic and (not oracle_fix),
            "position_invalid" : (not syntactic),
            "position_no_pos" : (n_found == 0)
        })


        expanded.append(new_row)

    return pd.DataFrame(expanded)
# Example usage:
def get_pandas_dataset(dataset_dir, result_dir):
    dataset_rows = retrieve_information_from_dataset_mod(dataset_dir)
    results_rows = retrieve_information_from_results(result_dir)
    merged = merge_dataset_and_results(dataset_rows, results_rows)
    df = create_pandas_dataset_and_expand_it_with_computed_data(merged)
    return df

if __name__ == '__main__':
      dataset_rows = retrieve_information_from_dataset_mod(gl.DAFNY_ASSERTION_DATASET)
      #dataset_rows = retrieve_information_from_dataset_mod(gl.BASE_PATH / "dafny_assertion_dataset_old_correct")
      df = pd.DataFrame(dataset_rows)
      print(df["benchmark"].value_counts())
      df43 = df[df["benchmark"] == "w/o-all"]

      i = 0
      for _, row in df43.iterrows():
         i += 1
         print(row["prog"] + "   " + row["group"])
      print(i)