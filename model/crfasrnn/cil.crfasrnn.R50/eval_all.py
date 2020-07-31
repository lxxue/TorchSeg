import os
import numpy as np

# lrs = ['0.001', '1e-05', '1e-07', '1e-09', '1e-11', '1e-13', '1e-15']
lrs = ['1e-09']
# epoch_nums = [1600]
epoch_nums = [1200, 1300, 1400, 1500, 1600, 1700, 1800]
num_iters = [1, 2, 4, 8]
print(epoch_nums)
for lr in lrs:
    for e in epoch_nums:
        for n in num_iters:
            print("-----------------------------------------------------------------")
            print(lr, e, n)
            os.system("python eval.py -lr {}  -e {} -n {} -p results_eval/{}/epoch{}/iter{}/".format(lr, e, n, lr, e, n))
            print("-----------------------------------------------------------------")
            # break
        # break
    # break
