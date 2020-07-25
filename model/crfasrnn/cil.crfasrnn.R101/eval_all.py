import os
import numpy as np

lrs = ['1e-06', '1e-09', '1e-12']
epoch_nums = [100, 200, 300]
num_iters = [1, 2, 4, 8]
print(epoch_nums)
for lr in lrs:
    for e in epoch_nums:
        for n in num_iters:
            print("-----------------------------------------------------------------")
            print(lr, e, n)
            os.system("python eval.py -lr {}  -e {} -n {} -p results_eval/{}/epoch{}/iter{}/".format(lr, e, n, lr, e, n))
            print("-----------------------------------------------------------------")
