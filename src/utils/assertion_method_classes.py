import utils.global_variables as gl
import utils.dafny_read_assertions_xml as dafny_read_assertions_xml

import os 

def substitute_a_given_pos_by_text(start_pos, end_pos, new_text, current_bytes):
        plus_padding = 1
        s = start_pos 
        e = end_pos 
        new_bytes = (current_bytes[:s] + new_text.encode("utf-8") +
                         current_bytes[e + plus_padding:]) 
        return new_bytes , new_bytes.decode("utf-8")

def get_method_from_assertion_group(assertion_group):
    return assertion_group[0].method

def get_file_from_assertion_group(assertion_group):
    return assertion_group[0].method.file

def get_assertion_group_string_id(assertion_group):
    method = get_method_from_assertion_group(assertion_group)
    ret_string = f"method_start_{method.start_pos}"
    for assertion in assertion_group:
        ret_string += f"_as_start_{assertion.start_pos}_end_{assertion.end_pos}"
    return ret_string


class FileSegment:
    def __init__(self, start_pos, end_pos, file_path):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.file_path = file_path
        
        self.segment_bytes = b""
        self.segment_str = ""
        self.populate_bytes_and_string()

    def populate_bytes_and_string(self):
        with open(self.file_path, "rb") as f:
            file_bytes = f.read()
        
        self.segment_bytes = file_bytes[self.start_pos:self.end_pos+1]
        self.segment_str = self.segment_bytes.decode("utf-8")
    

class AssertionInfo(FileSegment):
    def __init__(self, start_pos, end_pos ,asstype, method ):
        super().__init__(start_pos, end_pos, method.file.file_path)
        self.type = asstype
        self.method = method

    def __str__(self):
        return "ASSERT:" + self.segment_str + "START_POS:" + str(self.start_pos)
    
    def __repr__(self):
        return self.__str__()

class MethodInfo(FileSegment):
  def __init__(self, start_pos, end_pos,method_name, file):
        super().__init__(start_pos, end_pos, file.file_path)
        self.method_name = method_name
        self.file = file
        self.assertion_groups = []

  # This adds a list of assertions corresponding to the helper
  # If helper level 1 [assertion_1]
  # If helper level 2 [assertion1,assertion_2]
  def add_assertion_group(self,assertion_group):
        self.assertion_groups.append(assertion_group)

  def get_method_with_assertion_group_changed(self, assertion_group, remove_empty_lines, change_text):
    sorted_assertions = sorted(assertion_group, key=lambda x: x.start_pos)
    # Identify assertions to remove, ensuring nested assertions in "By_assertion" are properly handled
    removal_list = []
    end_of_by_assertion = 0
    for assertion in sorted_assertions:
        if assertion.type == "By_assertion":
            end_of_by_assertion = assertion.end_pos
            removal_list.append(assertion)
        elif assertion.start_pos >= end_of_by_assertion:
            removal_list.append(assertion)
    # Remove assertions in reverse order to preserve indexing
    removal_list.sort(key=lambda x: x.start_pos, reverse=True)
    new_method_bytes = self.segment_bytes[:]
    for assertion in removal_list:
        new_method_bytes, _ = substitute_a_given_pos_by_text(
            assertion.start_pos - self.start_pos,
            assertion.end_pos - self.start_pos,
            change_text,
            new_method_bytes
        )
    new_method_str = new_method_bytes.decode("utf-8")
    return "\n".join(line for line in new_method_str.split("\n") if line.strip()) if remove_empty_lines else new_method_str


class FileInfo:
    def __init__(self,file_path, file_parent_folder_path):
        self.file_parent_folder_path = file_parent_folder_path
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
          self.file_bytes = f.read()
        self.file_text =  self.file_bytes.decode("utf-8")

        self.start_pos = 0
        self.end_pos = len(self.file_bytes)+1

        self.methods = []

    def add_method(self,method):
        self.methods.append(method)

    def parse_dict(self, data):
        method_infos = {}

        for assertion in data:
            method_dict = assertion["method_info"]
            method_name = assertion["method_func_name"]
            method_start_pos = int(method_dict["start_pos"])
            method_end_pos = int(method_dict["end_pos"])

            if method_name not in method_infos:
                method_infos[method_name] = MethodInfo(method_start_pos, method_end_pos, method_name, self)
            
            assertion_list = [
                AssertionInfo(int(a["start_pos"]), int(a["end_pos"]), a["type"],  method_infos[method_name])
                for a in assertion["assertion_info"]
            ]
            method_infos[method_name].add_assertion_group(assertion_list)

        self.methods.extend(method_infos.values())

    def substitute_method_with_text(self, method, new_text):
        return substitute_a_given_pos_by_text(
            method.start_pos, 
            method.end_pos, 
            new_text, 
            self.file_bytes)

class Dataset():
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.files = []
        for dafny_file_folder in os.listdir(self.dataset_path):
            dafny_file_folder_path = self.dataset_path / dafny_file_folder
            if dafny_file_folder_path.is_dir():
                all_subdirs  = [
                    subdir for subdir in os.listdir(dafny_file_folder_path)
                    if (dafny_file_folder_path / subdir).is_dir() and subdir.startswith("method_start_")
                ]
                dafny_file_path = dafny_file_folder_path / "original_program.dfy"
                if dafny_file_path.exists():
                   file_obj = FileInfo(dafny_file_path, dafny_file_folder_path)
                else:
                    continue
                for subdir in all_subdirs:

                    info_xml_path = dafny_file_folder_path / subdir / "info.xml"  # Example of the path to the XML file
                    if info_xml_path.exists():
                        with open(info_xml_path, "r") as xml_file:
                            xml_content = xml_file.read()  # Put XML content in a variable
                            xml_dict = dafny_read_assertions_xml.extract_assertion(xml_content)
                            method =  MethodInfo(int(xml_dict["start_pos"]), int( xml_dict["end_pos"]),xml_dict["name"], file_obj)                        
                            for assertion_group in xml_dict["assertion_group"]:
                              assertion_group_list = []
                              for assertion in assertion_group["assertion"]:
                                 assertion_obj = AssertionInfo(int(assertion["start_pos"]), int(assertion["end_pos"]), assertion["type"], method)
                                 assertion_group_list.append(assertion_obj)
                              method.add_assertion_group(assertion_group_list)
                            file_obj.add_method(method)
                self.files.append(file_obj)




    def get_all_assertion_groups(self):
        dataset_assertion_groups = []
        for file in self.files:
            for method in file.methods:
                for assertion_group in method.assertion_groups:
                     dataset_assertion_groups.append(assertion_group)
        return dataset_assertion_groups
