from tools.find_max_degree import find_max_degree
from tools.split_2_co import split_2_co
ini = float('inf')
current_balls_num = 0


def split_ball_purity(graph, id_dict, C, total_degree_dict, total_balls_num, purity_threshold=1):
    cur_ball_num = len(C)
    while True:

        C.sort(key=lambda x: x[-1])
        if cur_ball_num >= total_balls_num or C[0][-1] >= purity_threshold:
            break

        GB = C.pop(0)
        value, index = find_max_degree(GB[1])
        cluster1, cluster2 = split_2_co(graph, id_dict, GB, index, total_degree_dict)
        temp_num = -1
        if len(cluster1) != 0:
            C.append(cluster1)
            temp_num += 1
        if len(cluster2) != 0:
            C.append(cluster2)
            temp_num += 1
        cur_ball_num += temp_num
    return C


def split_ball_further(graph, id_dict, C, total_degree_dict, total_balls_num,purity_threshold=1.0):
    cur_ball_num = len(C)
    while True:
        if cur_ball_num >= total_balls_num:
            break
        C.sort(key=lambda x: len(x[0]), reverse=True)
        GB = C.pop(0)
        value, index = find_max_degree(GB[1])
        cluster1, cluster2 = split_2_co(graph, id_dict, GB, index, total_degree_dict)
        temp_num = -1
        if len(cluster1) != 0:
            C.append(cluster1)
            temp_num += 1
        if len(cluster2) != 0:
            C.append(cluster2)
            temp_num += 1
        cur_ball_num += temp_num
    return C

