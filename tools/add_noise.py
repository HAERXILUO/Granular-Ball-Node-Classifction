import numpy as np
def add_noise(data):
    class_nums = len(set(np.array(data.y)))
    change_nums = 0
    for i in range(0, len(data.train_mask)):
        if data.train_mask[i]:
            p = np.random.randint(0, 100)
            if p < 10:
                change_nums += 1
                data.y[i] = generate_random_except(data.y[i], class_nums)

    return data


def generate_random_except(excluded_number, class_nums):
    choice_list = []
    for i in range(0, class_nums):
        if i != excluded_number:
            choice_list.append(i)
    return np.random.choice(choice_list)
