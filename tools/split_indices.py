import random

def split_indices(indices, ratio1, ways=''):
    if ways =='random':
        random.shuffle(indices)
        total_len = len(indices)
        len_1 = total_len * (100-ratio1) // 100
        list_1 = indices[:len_1]
        list_2 = indices[len_1:]
    else:
        total_len = len(indices)
        len_1 = total_len * (100-ratio1)//100
        list_1 = indices[:len_1]
        list_2 = indices[len_1:]

    return list_1, list_2
