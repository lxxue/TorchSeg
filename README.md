# CIL Road Segmentation Project

Team Member: Lixin Xue, Yu Fei, Hongjie Chen, Shengze Jin

[github repo](https://github.com/lxxue/TorchSeg)

Code for [CIL Road Segmentation Project](https://www.kaggle.com/c/cil-road-segmentation-2020/) based on [TorchSeg](https://github.com/ycszen/TorchSeg), a semantic segmentation codebase in PyTorch.

## Environment Setup

### Create conda environment
```bash
# install pytorch as instructed on pytorch.org
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install easydict
conda install easydict tqdm opencv matplotlib
```

### Download ResNet weight

Download the following two checkpoints into `model_zoo` dir
'''bash
cd ${TorchSeg}
mkdir model_zoo
cd model_zoo
'''

* [ResNet50](https://drive.google.com/file/d/1iEshXXzI3tCexo2CH92TNNOyizf2R_db/view?usp=sharing)
* [ResNet101](https://drive.google.com/file/d/1iEshXXzI3tCexo2CH92TNNOyizf2R_db/view?usp=sharing)

## Data
```bash
${TorchSeg}
├── cil
    ├── training                # training set images & gt
    ├── eval                    # validation set images & gt
    ├── test_images             # test imags
    ├── zhang-suen-thining
        ├── edges               # edge gt for PSPNet2
        ├── midlines            # midline gt for PSPNet2
```

## Base Models
```bash
${TorchSeg}
├── baslines
    ├── linear model
    ├── tf cnn
├── model
    ├── pspnet
        ├── pspnet r50
        ├── pspnet r101
        ├── pspnet r101 without data augmentation
    ├── pspnet2
        ├── pspnet2 r50
        ├── pspnet2 r101
    ├── crfasrnn
        ├── crfasrnn r50
        ├── crfasrnn r101
```

### Training
For each model, you can train the model after properly setting up `config.py`
```bash
python train.py -d 0
# for crfasrnn we need to specify the checkpoint path and the learning rate for crf part
# python train.py -d 0 --snapshot_dir <ckpt_path> --lr_crf <CRF_learning_rate>
```

### Evaluation on validation set
The checkpoints would be saved and we can evaluate the performance on validation set
```bash
python eval.py -e <epoch_num> -p <save_path>
```

For validation over many checkpoints, we can make minor modification in `eval_all.py` and run it without cmd line arguments.

We provide one checkpoint for each model through [polybox](https://polybox.ethz.ch/index.php/s/G5wRD4Raoj5Txy7). After unzip the file, just put the checkpoint into its corresponding model's `log/snapshot` directory and rename it to `epoch-xxxx.pth`, for example
```
mv cil_ckpts/pspnet2_R101_epoch-2800.pth ${TorchSeg}/model/cil.pspnet2.R101/log/snapshot/epoch-2800.pth
cd ${TorchSeg}/model/cil.pspnet2.R101/
python eval.py -e 2800 -p results_eval/epoch2800
```


### Inference on test set
For make predictions on the test data, we can run the following script
```bash
python pred.py -e <epoch_num> -p <pred_save_path>
```
and the predicted probability maps would be saved.

### Post-processing

To be added.

## Submission to Kaggle
For make submissions on the kaggle server, go to the `model` directory
```
python mask_to_submission.py -p <pred_save_path> -n <submission_fname>
kaggle competitions submit -c cil-road-segmentation-2020 -f <submission_fname> -m <message>
``` 
