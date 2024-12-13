from tools.find_major import find_major
from tools.find_major_num import find_major_num


def add_purity(C):
    for ball in C:
        nodes = ball[0]
        major_label = find_major(nodes)
        major_label_num, num_len = find_major_num(nodes, major_label)
        ball.append(float(major_label_num/num_len))

    return C