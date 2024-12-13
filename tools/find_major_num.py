def find_major_num(nums, target):
    sum = 0
    num_len = 0
    for num in nums:
        if num[-2] == target:
            sum += 1
        if num[-2] != -1:
            num_len += 1

    return sum, num_len
