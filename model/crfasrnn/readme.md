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

