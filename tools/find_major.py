def find_major(nums):
    candidate = 0
    count = 0


    for num in nums:
        if num[-2] != -1:
            if count == 0:
                candidate = num[-2]
                count = 1
            elif num[-2] == candidate:
                count += 1
            else:
                count -= 1

    return candidate