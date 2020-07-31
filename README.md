# CIL Road Segmentation Project

Team Member: Lixin Xue, Yu Fei, Hongjie Chen, Shengze Jin

[github repo](https://github.com/lxxue/TorchSeg)

Code for [CIL Road Segmentation Project](https://www.kaggle.com/c/cil-road-segmentation-2020/) based on [TorchSeg](https://github.com/ycszen/TorchSeg), a semantic segmentation codebase in PyTorch.

## Environment Setup

### Create conda environment
```shell
# install pytorch as instructed on pytorch.org
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install easydict
conda install easydict tqdm opencv matplotlib
```

### Download ResNet weight

Download the following two checkpoints into `model_zoo` dir

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

### Inference on test set
For make predictions on the test data, we can run the following script
```bash
python pred.py -e <epoch_num> -p <pred_save_path>
```
and the predicted probability maps would be saved.

## Post-processing

```bash
${TorchSeg}
├── post_processing
    ├── post_processing.py
    ├── tuning_parameters_for_post_processing.py
    ├── utils.py
```

The code for post-processing are put under directory `post_processing`, to apply the post_processing model you have to explicitly modify the code inside.

### Extract direction maps
The direciton-based kernel requires a direction map for each pixel. To process a image with post-processing modules, first generate direction maps by run

```python
    gen_dir_map_and_visualize(image_path= './images/',
                              vis_path='./vis_dir_blur_/',
                              dir_path='./dir_map_/',
                              process_all=True)
```

in `TorchSeg/post_processing/utils.py`.

### Parameters
The parameters of our post-processing module are hardcoded as a default setting in 

```python
    post_processing(image_path = './test_images/',vis_path = './results_eval/vis_crf_no_parking',
                    dir_path = './dir_map_test/', prob_path = './epoch3000_test/',
                    out_path = './results_eval/crf_no_parking', with_prop=False, with_kernel=False)
```

of the `TorchSeg/post_processing/post_processing.py`. The parameters for local test experiment are in the same file but a different function

```python
    post_processing_local_test(image_path = './test_images/',vis_path = './results/vis_crf',
                    dir_path = './dir_map_test/', prob_path = './epoch2800_test/',
                    out_path = './results/crf', with_prop=False, with_kernel=False)
```

### Process images
To post-process given images: 

1. Open `TorchSeg/post_processing/post_processing.py`
    
2. Call function the following function, set the pathes and desired component.
    
    ```python
    post_processing(image_path = './test_images/',vis_path = './results_eval/vis_crf_no_parking',
                    dir_path = './dir_map_test/', prob_path = './epoch3000_test/',
                    out_path = './results_eval/crf_no_parking', with_prop=False, with_kernel=False)
    ```
    
3. The tuned parameters are hardcoded in the function. If you'd like to set different parameters, change the code directly.

### Voting
Our final submission is based on a voting results of several base models and several crf parameters. To reproduce the voting result you have to set save all the model probabiities as well as the original images and the direction maps. Then run

```python
    voting_gen(image_path = './test_images/', dir_path = './dir_map_test/', 
               prob_base_path='./train_all/', out_path='./crf_all/')
```

in `TorchSeg/post_processing/post_processing.py` to generate all voting candidates.

Then run

```python
    voting(image_path = './test_images/', use_extra_param=True, 
           out_path="./crf_vote/", vis_path="./vis_vote/",thres = 0.2)
```
    
in `TorchSeg/post_processing/post_processing.py` to get the final result.

### Tuning the CRF parameters
Even if we argue that out proposed direction-based kernel can be inplemented in an end-to-end manner, with limited data, we found random search on a validation set a more generalizable way to tune the parameters.

To tune the parameters yourself, open `TorchSeg/post_processing/tuning_parameters_for_post_processing.py` and go to

```python
    cross_validation_random()
```

Change the pathes in 

```python
    score = compute_score_full_param(prop_num, window_size, sigma_prop_color, sigma_prop_pos,
                             crf_iter_num, compat_smooth, compat_appearance, compat_struct, 
                             w_smooth, w_appearance, w_struct,
                             sigma_smooth, sigma_app_color, sigma_app_pos,
                             sigma_struct_pos, sigma_struct_feat,
                             crf_sm_iter_num, crf_sm_w, crf_sm_sigma,
                             with_prop=with_prop, with_struct_kernel=with_struct_kernel, use_sm_crf=use_sm_crf, 
                             cv_label_path = './groundtruth/', cv_img_path = './images/', 
                             test_prob_path = "./epoch2800_eval/", dir_path = './dir_map_train/')
 ```

to tune the parameters with your favorate settings or images. Note that you have to generate the direction map first.

## Submission to Kaggle
For make submissions on the kaggle server, go to the `model` directory
```
python mask_to_submission.py -p <pred_save_path> -n <submission_fname>
kaggle competitions submit -c cil-road-segmentation-2020 -f <submission_fname> -m <message>
``` 
