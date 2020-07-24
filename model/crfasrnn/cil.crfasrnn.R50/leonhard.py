import os

# lrs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
lrs = [1e-13]

for lr in lrs:
    print("bsub -n 4 -W 24:00 -R 'rusage[mem=10000, ngpus_excl_p=1]' python train.py -d 0 --snapshot_dir log/snapshot_{} --lr_crf {}".format(lr, lr)) 
    os.system("bsub -n 4 -W 24:00 -R 'rusage[mem=10000, ngpus_excl_p=1]' python train.py -d 0 --snapshot_dir log/snapshot_{} --lr_crf {}".format(lr, lr)) 


