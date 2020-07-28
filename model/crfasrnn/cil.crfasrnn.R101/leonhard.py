import os

lrs = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15]

for lr in lrs:
    print("bsub -n 4 -W 120:00 -R 'rusage[mem=10000, ngpus_excl_p=1]' python train.py -d 0 --snapshot_dir log/snapshot_{} --lr_crf {}".format(lr, lr)) 
    os.system("bsub -n 4 -W 120:00 -R 'rusage[mem=10000, ngpus_excl_p=1]' python train.py -d 0 --snapshot_dir log/snapshot_{} --lr_crf {}".format(lr, lr)) 


