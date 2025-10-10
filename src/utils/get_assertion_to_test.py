# Processes the sets of previous levels to give the assertions that are needed to text for this level
import dafny.dafny_runner as dafny_runner
import utils.dafny_read_assertions_xml as dafny_read_assertions_xml

import itertools

def process_assertions_per_level_old(assertions_ids, helper_assertions_of_all_levels, nothelper_assertions_previous_level, number_to_remove):
    if(number_to_remove == -1):
        helper_assertion_to_test = [{i for i in assertions_ids}]
        return  helper_assertion_to_test
 
    # old implementation had restriction on assertions pairs (only paired if w/o-2 if did not had any)
    possibilites_for_next_level = []
    if(number_to_remove == 1):
        helper_assertion_to_test = [{i} for i in assertions_ids]
        return  helper_assertion_to_test

    for i in assertions_ids:
        for prev_level_assertion_set in nothelper_assertions_previous_level:
          # if assertion already on the set not needed to test
          if(i in prev_level_assertion_set):
              continue
          possibility = set(list(prev_level_assertion_set) + [i])
          # If already added that det not repeated
          if(possibility in possibilites_for_next_level):
              continue
          is_a_possibility = 1
          # Code responsible to stop assertions already identified as Helper to make pairs with others 
          # By removing this i believe the number of cases will explode lets run
          for helper_assertion_prev_level in  helper_assertions_of_all_levels:
              for helper_prev_sequences in helper_assertion_prev_level:
                  if(helper_prev_sequences.issubset(possibility)):
                      is_a_possibility = 0
                      break 
              if(is_a_possibility == 0):
                  break 
          if(is_a_possibility):
                 possibilites_for_next_level.append(possibility)
    return possibilites_for_next_level


def process_assertions_per_level(assertions_ids, number_to_remove):
    if(number_to_remove == -1):
        helper_assertion_to_test = [{i for i in assertions_ids}]
        return  helper_assertion_to_test

    helper_assertion_to_test = [set(c) for c in itertools.combinations(assertions_ids, number_to_remove)]
    return helper_assertion_to_test

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def create_assertion_xml(number_to_remove, assertions_to_remove, method_info, output_path, group_id):
    root = ET.Element("method")
    ET.SubElement(root, "name").text = method_info["name"]
    ET.SubElement(root, "start_pos").text = str(method_info["start_pos"])
    ET.SubElement(root, "end_pos").text = str(method_info["end_pos"])

    assertion_group = ET.SubElement(root, "assertion_group")
    ET.SubElement(assertion_group, "id").text = str(group_id)
    ET.SubElement(assertion_group, "number_assertions").text = str(number_to_remove)

    for assertion in assertions_to_remove:
        assertion_elem = ET.SubElement(assertion_group, "assertion")
        ET.SubElement(assertion_elem, "type").text = assertion["type"]
        ET.SubElement(assertion_elem, "start_pos").text = str(assertion["start_pos"])
        ET.SubElement(assertion_elem, "end_pos").text = str(assertion["end_pos"])

    # Convert the tree to a string
    rough_string = ET.tostring(root, encoding="utf-8")

    # Use minidom to format with indentation and newlines
    parsed_xml = minidom.parseString(rough_string)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")  # Two spaces for indentation

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

# max_assertions_to_remove == -1 remove all
def process_assertions_method( dafny_exec, program_dst_folder,  program_path, method_info, assertion_infos, temp_dir, max_assertions_to_remove):
    dataset_list = []
    if(max_assertions_to_remove == -1):
        number_to_remove = -1
    else:
        number_to_remove = 1
    assertion_ids = [i for i in range(len(assertion_infos))]
    method_name = method_info["name"]
    helper_assertions_of_all_levels = []
    nothelper_assertions_previous_level = []
    n_assertions = len(assertion_ids)

    group_id = 0
    while(1):
        if(number_to_remove != -1 and number_to_remove > max_assertions_to_remove):
            break
        assertions_sets_to_test = process_assertions_per_level(assertion_ids, number_to_remove)
        print("Method: " + method_name + "number: " + str(number_to_remove) + "assertion_to_test:" + str(assertions_sets_to_test))
        new_level_helper = []
        prev_level_not_helper = []
        for assertion_set in assertions_sets_to_test:
            assertion_set_list =  list(assertion_set)
            # If number of assertions is one or two this case is englobated already in the 1 and 2 exaustive search
            if(len(assertion_set_list) <= 2 and number_to_remove == -1):
                continue

            # Removing from the later to the begginning allows to remove one by one
            assertions_to_remove = []
            for assertion_id in assertion_set_list:
              assertions_to_remove.append(assertion_infos[assertion_id])

            assert_str = f"method_start_{method_info["start_pos"]}"
            for assertion in assertions_to_remove:
                assert_str += f"_as_start_{assertion["start_pos"]}_end_{assertion["end_pos"]}"
            program_assertion_dst_dir = program_dst_folder / assert_str

            status, stdout_msg, _ , new_program_text, method_new_text =  test_new_program(dafny_exec, program_path, assertions_to_remove,method_info , temp_dir, id)
            #print("-----------------------------------------------")
            #print(method_new_text)
            #print("----status msg ------")
            #print(stdout_msg)
            #print("-----------------------------------------------")
            if(status == "NOT_VERIFIED"):
                new_level_helper.append(assertion_set)
            elif(status == "VERIFIED"):
                prev_level_not_helper.append(assertion_set) 
            else:
                print("Error")
                print(status)
                print(stdout_msg)
    
            if(status == "Error" or status == "NOT_VERIFIED"):
              print(f"Creating {program_assertion_dst_dir }")
              os.makedirs(program_assertion_dst_dir, exist_ok=True) 

              method_without_assertions = program_assertion_dst_dir / "method_without_assertion_group.dfy"
              with open(method_without_assertions,"w") as f:
                  f.write(method_new_text)

              program_without_assertions = program_assertion_dst_dir / "program_without_assertion_group.dfy"
              with open(program_without_assertions,"w") as f:
                  f.write(new_program_text)
                  
              verifier_output_file = program_assertion_dst_dir / "verifier_output.txt"
              with open(verifier_output_file,"w") as f:
                  f.write(stdout_msg)

              output_path = program_assertion_dst_dir / "info.xml"
              create_assertion_xml(number_to_remove, assertions_to_remove, method_info, output_path, group_id)
              group_id += 1

        print("New level helper:" + str(new_level_helper))
        print("Not helper:" + str(prev_level_not_helper))  

        helper_assertions_of_all_levels.append(new_level_helper)
        nothelper_assertions_previous_level = prev_level_not_helper

        if(n_assertions <= number_to_remove):
            break

        if(number_to_remove == -1):
            break

        number_to_remove += 1

    return dataset_list

import os
def create_temp_directories(base_path, id):
    temp_source = os.path.join(base_path, f"src__{id}")
    temp_result = os.path.join(base_path, f"res__{id}")
    os.makedirs(temp_source, exist_ok=True)
    os.makedirs(temp_result, exist_ok=True)
    return temp_source, temp_result

# removes the assertion_set of the program
def test_new_program(dafny_exec, dafny_file, assertions_info, method_info, temp_dir, id):
    dafny_new_program_text, method_new_text = dafny_read_assertions_xml.get_file_and_method_without_assertions(dafny_file, assertions_info, method_info, remove_empty_lines = 1)
    status, stdout_msg, stderr = dafny_runner.run_dafny_from_text (dafny_exec, dafny_new_program_text, temp_dir)
    return status, stdout_msg, stderr, dafny_new_program_text, method_new_text

