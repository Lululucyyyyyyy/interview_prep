def bin_search(nums, x):
    l = 0
    r = len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] >= x:
            r = m - 1
        elif nums[m] < x:
            l = m + 1
    return r + 1