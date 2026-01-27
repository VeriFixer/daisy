# Processes the sets of previous levels to give the assertions that are needed to text for this level
import dafny.dafny_runner as dafny_runner
import utils.dafny_read_assertions_xml as dafny_read_assertions_xml

import itertools
from utils.assertion_method_classes import AssertionInfo, MethodInfo
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pathlib import Path

def create_assertion_xml(number_to_remove: int, assertions_to_remove : list[AssertionInfo], method_info : MethodInfo, output_path : Path, group_id : int):
    root = ET.Element("method")
    ET.SubElement(root, "name").text = method_info.method_name
    ET.SubElement(root, "start_pos").text = str(method_info.start_pos)
    ET.SubElement(root, "end_pos").text = str(method_info.end_pos)

    assertion_group = ET.SubElement(root, "assertion_group")
    ET.SubElement(assertion_group, "id").text = str(group_id)
    ET.SubElement(assertion_group, "number_assertions").text = str(number_to_remove)

    for assertion in assertions_to_remove:
        assertion_elem = ET.SubElement(assertion_group, "assertion")
        ET.SubElement(assertion_elem, "type").text = assertion.type
        ET.SubElement(assertion_elem, "start_pos").text = str(assertion.start_pos)
        ET.SubElement(assertion_elem, "end_pos").text = str(assertion.end_pos)

    # Convert the tree to a string
    rough_string = ET.tostring(root, encoding="utf-8")

    # Use minidom to format with indentation and newlines
    parsed_xml = minidom.parseString(rough_string)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")  # Two spaces for indentation

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

from pathlib import Path
from utils.assertion_method_classes import MethodInfo, AssertionInfo, assertionGroup
# max_assertions_to_remove == -1 remove all

def write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def process_assertions_per_level(assertions_ids : list[int], number_to_remove : int):
    if(number_to_remove == -1):
        helper_assertion_to_test = [{i for i in assertions_ids}]
        return  helper_assertion_to_test

    helper_assertion_to_test = [set(c) for c in itertools.combinations(assertions_ids, number_to_remove)]
    return helper_assertion_to_test

# -1 max assertion to remove means remove all assertion
# infinity is also possible to be passed (no infinite loop will occur)
def process_assertions_method( dafny_exec : Path, program_dst_folder : Path,  program_path: Path, 
                                                         method_info: MethodInfo, max_assertions_to_remove: int) -> None:
    def process_assertion_set(assertion_infos : list[AssertionInfo], n_assertions_to_remove: int, init_grop_id : int) -> tuple[list[assertionGroup], int]:
        
        found_assertion_groups : list[assertionGroup] = []
        group_id = init_grop_id

        assertion_ids = list(range(len(assertion_infos)))
        assertions_sets_to_test = process_assertions_per_level(assertion_ids, n_assertions_to_remove)
        
        for assertion_set in assertions_sets_to_test:
            # Removing from the later to the begginning allows to remove one by one
            assertions_to_remove = [assertion_infos[i] for i in assertion_set]

            assertion_dir_suffix = (
                f"method_start_{method_info.start_pos}"
                + "".join(f"_as_start_{a.start_pos}_end_{a.end_pos}" for a in assertions_to_remove)
            )

            program_assertion_dst_dir = program_dst_folder / assertion_dir_suffix

            status, stdout_msg, _ , new_program_text, method_new_text =  test_new_program(dafny_exec, program_path, assertions_to_remove)
            if status not in {dafny_runner.Status.VERIFIED, dafny_runner.Status.NOT_VERIFIED}:
                print("Error running Dafny:", status)
                print(stdout_msg)

            if status in {dafny_runner.Status.NOT_VERIFIED}:
                # Save program files and outputs
                write_file(program_assertion_dst_dir / "method_without_assertion_group.dfy", method_new_text)
                write_file(program_assertion_dst_dir / "program_without_assertion_group.dfy", new_program_text)
                write_file(program_assertion_dst_dir / "verifier_output.txt", stdout_msg)

                # Save XML info
                create_assertion_xml(
                    n_assertions_to_remove, assertions_to_remove, method_info,
                   program_assertion_dst_dir / "info.xml", group_id
                )

                found_assertion_groups.append(assertions_to_remove)
                group_id += 1
           
            
        return found_assertion_groups, group_id
    
    group_id = 0

    # This comes from reading the all dataset , need to convert it into array
    assertion_groups_list = method_info.assertion_groups
    if(len(assertion_groups_list) == 0):
        return
    
    assertion_list = assertion_groups_list[0]

    if(max_assertions_to_remove != -1):
       helper_lvl_1, group_id = process_assertion_set(assertion_list, 1, 0)
       # Note all elemnts of lv1_1 have grops have exactly one assertion
       helper_lvl_1_assertions: list[AssertionInfo] = [x[0] for x in helper_lvl_1]
       actual_max_assertions = min(max_assertions_to_remove, len(helper_lvl_1_assertions))
       # Has helper_lvl_1_assertions represent the assertion at most that we can remove (remove all lvl 1 at the same time)
       for number_to_remove in range(2, actual_max_assertions + 1):
            _, group_id = process_assertion_set(helper_lvl_1_assertions, number_to_remove, group_id)

    else:
       _,_= process_assertion_set(assertion_list, -1, 0)

import os
def create_temp_directories(base_path : Path, id: int)-> tuple[Path, Path]:
    temp_source = os.path.join(base_path, f"src__{id}")
    temp_result = os.path.join(base_path, f"res__{id}")
    os.makedirs(temp_source, exist_ok=True)
    os.makedirs(temp_result, exist_ok=True)
    return Path(temp_source), Path(temp_result)

# removes the assertion_set of the program
def test_new_program(dafny_exec : Path, dafny_file : Path, assertions_group : list[AssertionInfo]) -> tuple[dafny_runner.Status,str,str,str,str]:
    dafny_new_program_text, method_new_text = dafny_read_assertions_xml.get_file_and_method_without_assertion_group(dafny_file, assertions_group, remove_empty_lines = 1)
    status, stdout_msg, stderr = dafny_runner.run_dafny_from_text (dafny_exec, dafny_new_program_text, tmp=True)
    return status, stdout_msg, stderr, dafny_new_program_text, method_new_text

