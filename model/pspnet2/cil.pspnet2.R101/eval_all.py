import os
import numpy as np

epoch_nums = np.arange(200, 4100, step=200)
print(epoch_nums)
for e in epoch_nums:
    print("-----------------------------------------------------------------")
    print(e)
    os.system("python eval.py -e {} -p results_eval/epoch{}/".format(e, e))
    print("-----------------------------------------------------------------")
