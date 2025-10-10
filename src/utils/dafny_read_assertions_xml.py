# This file is reponsible by parsing a Dafny file retrieving the AST
# It must also implement functions allowying it to retrieve
  # Assertions source position
  # Insert assertions at a given position in another source file
import utils.global_variables as gl

import xml.etree.ElementTree as ET
import os 

def normalize_dict(data):
    def ensure_list(key):
        """Ensures that the given key in `data` is a list."""
        if key not in data:
            data[key] = []
        elif isinstance(data[key], dict):
            data[key] = [data[key]]

    def normalize_assertions(method_or_function):
        """Ensures the 'assertion' field inside a method or function is a list."""
        if 'assertion' not in method_or_function:
            method_or_function['assertion'] = []
        elif isinstance(method_or_function['assertion'], dict):
            method_or_function['assertion'] = [method_or_function['assertion']]

    for key in ['Method', 'Function', "assertion_group"]:
        ensure_list(key)
        for item in data[key]:
            normalize_assertions(item)
    return data

def xml_to_dict(element):
    """Recursively converts an XML element and its children into a dictionary."""
    result = {}
    
    if list(element):
        for child in element:
            child_data = xml_to_dict(child)
            if child.tag in result:
                if isinstance(result[child.tag], list):
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = [result[child.tag], child_data]
            else:
                result[child.tag] = child_data
    else:
        result = element.text.strip() if element.text else ""
    
    return result


def extract_assertion(dafny_assertion_ast_raw_text):
    """
    Gets the raw text , obtained after runninf modified dafny verify to extract allassetions in xml
    Reads it and parser giving back a dicionary containing all the information required to travel 
    the assertions
    """
    content = dafny_assertion_ast_raw_text
    if(content is None):
        return None
    root = ET.fromstring(content)

    xml_converted = xml_to_dict(root)
    return normalize_dict(xml_converted)
 
def extract_assertion_from_file(dafny_assertion_ast_file):
    raw_assert_string = ""
    with open(dafny_assertion_ast_file, "r") as f:
        raw_assert_string = f.read()

    return extract_assertion(raw_assert_string)

def replace_assertion_by(dafny_file_text, assertion_info, substitute =""):
    posi = int(assertion_info["start_pos"])
    pose = int(assertion_info["end_pos"])
    return dafny_file_text[:posi] + substitute + dafny_file_text[pose+1:]


def get_assertion_bytes_and_string(dafny_file_bytes, assertion_info):
    posi = int(assertion_info["start_pos"])
    pose = int(assertion_info["end_pos"])
    substring_bytes = dafny_file_bytes[posi:pose+1]
    substring_text = substring_bytes.decode("utf-8")
    return substring_bytes, substring_text


def get_method_bytes_and_string(dafny_file_bytes, method_info):
    posi = int(method_info["start_pos"])
    pose = int(method_info["end_pos"])
    substring_bytes = dafny_file_bytes[posi:pose+1]
    substring_text = substring_bytes.decode("utf-8")
    return substring_bytes, substring_text  

def replace_method_by(dafny_file_bytes, method_info, substitute):
    posi = int(method_info["start_pos"])
    pose = int(method_info["end_pos"])
    plus_padding = 1
    new_raw_bytes = dafny_file_bytes[:posi] + substitute.encode("utf-8") + dafny_file_bytes[pose+plus_padding:]
    return new_raw_bytes, new_raw_bytes.decode("utf-8")

def replace_assertion_by(dafny_file_bytes, assertion_info, substitute =""):

    posi = int(assertion_info["start_pos"])
    pose = int(assertion_info["end_pos"])

    plus_padding = 1

    new_raw_bytes = dafny_file_bytes[:posi] + substitute.encode("utf-8") + dafny_file_bytes[pose+plus_padding:]
    return new_raw_bytes, new_raw_bytes.decode("utf-8")

def replace_assertion_in_method_by(method_file_bytes, method_info, assertion_info, substitute =""):
    posi = int(assertion_info["start_pos"]) - int(method_info["start_pos"])
    pose = int(assertion_info["end_pos"]) - int(method_info["start_pos"])

    plus_padding = 1
    new_raw_bytes = method_file_bytes[:posi] + substitute.encode("utf-8") + method_file_bytes[pose+plus_padding:]
    return new_raw_bytes, new_raw_bytes.decode("utf-8")

# It is expected for the assertions positions to be sorted
# if method_info different than {} it also return method replaced info
def remove_empty_lines_function(text):
    return "\n".join(line for line in text.split("\n") if line.strip())
    
def get_file_and_method_without_assertions(dafny_file, assertions_info, method_info = {}, remove_empty_lines = 0):
        #Pick a method remove all assertions one by one and see if working
        with open(dafny_file, "rb") as f:
          content_bytes = f.read()

        # Assertions Sorted
        sorted_assertions_info = sorted(assertions_info, key=lambda x: int(x["start_pos"]))
        # If a assertion is of type By_assertion (i cannot remove assertions that are inside it if i will remove it:)
        # Logic to remove them
        assertions_to_remove = []
        inside_by_assertions = 0
        end_of_by_assertions = 0
        for assertion in sorted_assertions_info:
            if(not inside_by_assertions):
                assertions_to_remove.append(assertion)
            else:
                if(assertion["end_pos"] > end_of_by_assertions):
                    assertions_to_remove.append(assertion)
            if(assertion["type"] == "By_assertion"):
                inside_by_assertions = 1
                end_of_by_assertions = assertion["end_pos"]
        # The assertions should be in reverse order in order the remove one by one from the end
        assertions_to_remove = sorted(assertions_to_remove, key=lambda x: int(x["start_pos"]), reverse=True)

        new_file_bytes = content_bytes[:]
        for i,assertion in enumerate(assertions_to_remove):
            new_file_bytes, _ = replace_assertion_by(new_file_bytes, assertion)

        new_file_str = new_file_bytes.decode("utf-8")
        if(remove_empty_lines):
            new_file_str = remove_empty_lines_function(new_file_str)

        if(method_info == {}):
            return new_file_str, ""
        
        method_bytes, _ = get_method_bytes_and_string(content_bytes, method_info)
        for i,assertion in enumerate(assertions_to_remove):
            method_bytes, _ = replace_assertion_in_method_by(method_bytes,method_info,assertion)
        method_str = method_bytes.decode("utf-8")
        if(remove_empty_lines):
            method_str = remove_empty_lines_function(method_str)
        return new_file_str, method_str
