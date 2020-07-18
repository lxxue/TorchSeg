1. install `crfasrnn`
   1. `cd crfasrnn`
   2. run `python setup.py install` (if error, try `CFLAGS='-stdlib=libc++' python setup.py build`)
2. Changes I made in `config.py`
   1. `C.num_workers=0`
   2. `C.pretrained_model = None`

