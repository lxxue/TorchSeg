1. install `crfasrnn`
   1. `cd crfasrnn`
   2. run `python setup.py install` (if error, try `CFLAGS='-stdlib=libc++' python setup.py build`)
2. main parameters to tune during training
   1. `config.py: C.lr`
   2. `config.py: C.train_num_iter`
   3. `crfasrnn/params.py: alpha, beta, gamma` (fixed)
   4. `crfasrnn/params.py: spatial_ker_weight, bilateral_ker_weight` (trainable)
3. main parameters to tune during eval
   1. `config.py: C.eval_num_iter`
   2. Caveat: if error during prediction, set `C.batch_size=1`.



### parameters found by Yu Fei

1. crf iter num=8
2. compat_sm=[-0.07720807 0.94233034 1.22431553 0.46881228]
3. compat_bi=[-0.44835076 1.41230527 0.7765879 -0.03361967]
4. w_sm=3.7606425465775133
5. w_bi=1.6835678682669473
6. sigma_sm=4.461952267619522
7. sigma_clr=1.7500746997459924
8. sigma_pos=40