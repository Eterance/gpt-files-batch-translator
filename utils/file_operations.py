# Everything about file operations.
import json
import logging
import os
import pickle
import time
from typing import TypeVar

ListOrDictType = TypeVar("ListOrDictType", list, dict)

def save_variable_to_json_file(dict, fileName:str, indent:int=4, logger:logging.Logger=None):
    """
    Only Json serializable objects can be saved.
    If the object is Non Json serializable (such as Tensor), use save_variable_to_pickle_file() instead.
    """
    with open(fileName, 'w') as f:
        # use parameter: indent to beautify the json file
        start_time = time.time()
        dict_json = json.dumps(dict, indent=indent)
        dump_time = time.time()
        f.write(dict_json)
        end_time = time.time()
        if logger is not None:
            logger.info(f"save_variable_to_json_file(): dump_time: {dump_time-start_time}, write_time: {end_time-dump_time}, total_time: {end_time-start_time}")

def load_variable_from_json_file(fileName:str):
    with open(fileName, encoding='utf-8') as file:
        variable = json.load(file)
    return variable

def save_variable_to_pickle_file(variable, fileName:str):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)

def load_variable_from_pickle_file(fileName:str):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def find_all_file_paths_with_suffix(target_dir:str, target_suffix:str, recursive=False):
    """
    Find all files with the target suffix in the target directory.

    parameters:
        target_dir: str, the target directory.
        target_suffix: str, the target suffix, such as ".sql".

    return:
        list[str]: the list of file paths.
    """
    files:list[str] = []
    for root, dirs, file_names in os.walk(target_dir):
        if len(file_names) < 1:
            continue
        for file_name in file_names:
            if file_name.endswith(target_suffix):
                files.append(os.path.join(root, file_name))
        if not recursive:
            break
    return files

def search_files(path:str, search_string:str, recursive=False)->list[str]:
    """
    Search for files containing a given search string in the specified directory and optionally
    its subdirectories.
    
    Args:
        path (str): The directory to search in.
        search_string (str): The search string to look for in file names.
        recursive (bool, optional): Whether to search recursively in subdirectories. Defaults to False.
    
    Returns:
        List[str]: A list of file names that contain the search string.
    """
    matching_files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if search_string in filename:
                matching_files.append(os.path.join(root, filename))
        if not recursive:
            break
    return matching_files


def generate_predict_sql_file(sql_list:list[str], path:str):
    #print(len(list1))
    str1:str = ""
    for (index, item) in enumerate(sql_list):
        # codex 生成的 sql 可能会包含 \n
        # 因此需要将 \n 替换为空字符串
        if "\n" in item:
            #print(index)
            #print(item)
            item = item.replace("\n", " ")
            #print(item)
        if item == "":
            item = "<EMPTY>"
        str1 += item + "\n"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str1)
        f.close()
    
def read_prompt_file_and_remove_single_line_comment(prompt_file_path:str): 
    with open(prompt_file_path, encoding='utf-8') as f:
        lines:list[str] = [line for line in f.readlines()]
    result = ""
    for line in lines:
        if not line.startswith("#"):
            result = f"{result}\n{line}"
    return result


def load_and_merge_multiple_dicts_or_lists(folder_path:str, empty_list_or_dict:ListOrDictType):
    """
    Load all json files in the folder and merge them into a dict or list.

    Args:
        folder_path (str): The folder path.

    Returns:
        dict | List: The merged dict or list.
    """
    merged = empty_list_or_dict
    file_names = find_all_file_paths_with_suffix(folder_path, ".json", recursive=False)
    for file_name in file_names:
        file_dict_or_dict = load_variable_from_json_file(file_name)
        if isinstance(merged, list):
            merged.extend(file_dict_or_dict)
        else:
            merged.update(file_dict_or_dict)
    return merged