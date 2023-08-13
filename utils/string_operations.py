from ctypes import Union
import re
from sys import float_repr_style
from typing import Type
from typing import TypeVar, Generic

T = TypeVar('T', int, float)

def strip_quotes(enable:bool, string:str):
    """
    Strip quotes from a string.
    """
    if enable:
        string = string.strip("`").strip("'").strip("\"")
    return string

def get_all_numbers_in_string(string:str, t:Type = float):
    """
    Get all numbers in a string.
    
    Args:
        string (str): The string to be searched.
        type (Type): The type of the numbers to be returned.
    """
    if t != float and t != int:
        raise ValueError("Type must be float or int.")
    list1 = re.findall(r'\d+', string)
    numbers_list:list[str] = []
    numbers_list_origin:list[str] = []
    for item in list1:
        try:
            t(item)
            numbers_list.append(t(item))
            numbers_list_origin.append(item)
        except ValueError:
            pass
    return numbers_list, numbers_list_origin

def string2number(string:str, t:type = float):
    if t != float and t != int:
        raise ValueError("Type must be float or int.")
    try: 
        result = t(string)
        return result
    except ValueError:
        return None
    
def list_2_str_list(somethings:list) -> 'list[str]':
    result = []
    for sth in somethings:
        result.append(str(sth))
    return result

def str_list_all_lower(str_list:'list[str]') -> 'list[str]':
    return [item.lower() for item in str_list]
    #result = []
    #for item in str_list:
    #    result.append(item.lower())
    #return result

def add_annotate_sign_at_line_start(prompt:str, sign:str="#"):
    """
    Add a # at the start of each line.

    Args:
        prompt (str): _description_

    Returns:
        _type_: _description_
    """
    prompts = prompt.split("\n")
    for index, line in enumerate(prompts):
        prompts[index] = f"{sign} {line}"
    return "\n".join(prompts)