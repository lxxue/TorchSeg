import os
import numpy as np

epoch_nums = np.arange(100, 4100, step=100)
print(epoch_nums)
for e in epoch_nums:
    os.system("python eval.py -e {} -p results_eval/epoch{}/".format(e, e))
