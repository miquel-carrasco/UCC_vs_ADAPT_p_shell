import numpy as np
import os
import random
from itertools import permutations


def labels_all_combinations(label_list: list[int]):
    """Returns all combinations of operators in the list."""
    
    n = len(label_list)
    combinations = list(permutations(label_list))
    combinations = [list(comb) for comb in combinations]
    return combinations


def asc_desc_order(mats_list,v0,H,V_ijlk):
    V_ijkl=np.array(V_ijlk)
    ascending_order=np.argsort(np.abs(V_ijlk))
    ascending=[V_ijkl[i] for i in ascending_order]
    asc_list=[mats_list[i] for i in ascending_order]
    desc_list=[mats_list[i] for i in ascending_order[::-1]]
    return asc_list,desc_list

