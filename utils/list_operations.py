

from copy import deepcopy
from typing import Any


def divide_list_into_n_parts(lst:list[Any], n:int, deep_copy:bool=False)->list[list]:
    """
    This function takes a list and an integer as input and returns a list of n sub-lists of equal or almost equal length.

    Args:
        lst (list): A list of elements to be divided into n parts.
        n (int): An integer representing the number of sub-lists to be created.
        deep_copy (bool): An optional boolean parameter indicating whether to return a deep copy of the sub-lists (default is False).
        
    Returns:
        list[list]: A list containing n sublists of equal or almost equal length, where each sublist is a portion of the original list.
    """
    quotient = len(lst) // n
    remainder = len(lst) % n

    result:list[list] = []
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient
        if deep_copy:
            result.append(deepcopy(lst[start:end]))
        else:
            result.append(lst[start:end])
        start = end

    return result

def completing_list_elements(lst:list[list], target_index:int):
    """ Completing the list elements to the target index.

    Args:
        lst (list[list]):
        target_index (int):

    Returns:
    """
    if len(lst)-1 < target_index:
        for _ in range(len(lst)-1, target_index):
            lst.append([])
    return lst