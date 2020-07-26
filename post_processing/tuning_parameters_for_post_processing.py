
# coding: utf-8

# In[1]:


import numpy as np
import pydensecrf.densecrf as dcrf
import os
import cv2
import random
from tqdm import tqdm
from utils import *


# In[2]:


from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
#from osgeo import gdal
# %matplotlib inline

MAX = 0
SUM = 1
VEC = 0
MAT = 1


# In[10]:


def compute_score_full_param(prop_num=None, window_size=None, sigma_prop_color=None, sigma_prop_pos=None,
                             crf_iter_num=None, compat_smooth=None, compat_appearance=None, compat_struct=None, 
                             w_smooth=None, w_appearance=None, w_struct=None,
                             sigma_smooth=None, sigma_app_color=None, sigma_app_pos=None,
                             sigma_struct_pos=None, sigma_struct_feat=None,
                             crf_sm_iter_num=None, crf_sm_w=None, crf_sm_sigma=None,
                             with_prop=True, with_struct_kernel=True, use_sm_crf=False, 
                             cv_label_path = './groundtruth/', cv_img_path = './images/', 
                             test_prob_path = "./epoch3000_eval/", dir_path = './dir_map_train/'):
    """Compute patched accuracy for given model parameters
    
    """
    threshold_for_label = 0.25 # as suggested in the provided code
    
    label_list = []
    prob_list = []
    crf_list = []
    i=0
    for file in os.listdir(test_prob_path):
        label =cv2.imread(os.path.join(cv_label_path, file))
        label = label[:,:,0]
        label[label>0]=1
        label = prob_to_patch(label)
        label[label>threshold_for_label]=1
        label[label<=threshold_for_label]=0
        label_list.append(label)
        
        img=cv2.imread(os.path.join(cv_img_path, file))
        prob=cv2.imread(os.path.join(test_prob_path, file))
        y=prob/255
        y=y[:,:,0]
        prob_list.append(prob_to_patch(y))
        if with_prop:
            prop_img = propagate_max_vec(img,y,prop_num=prop_num, prop_size=window_size, 
                                         sigma_color=sigma_prop_color, sigma_pos=sigma_prop_pos)
        else:
            prop_img = y
        if with_struct_kernel:
            dir_color_map = cv2.imread(os.path.join(dir_path, file))
            dir_feature = dir_to_features(dir_color_map)
            crf_img = crf_with_dir_kernel(img, dir_feature, prop_img,
                                         iter_num=crf_iter_num, compat_smooth=compat_smooth,
                                         compat_appearance=compat_appearance,
                                         compat_struct=compat_struct,
                                         w_smooth=w_smooth, w_appearance=w_appearance,
                                         w_struct=w_struct, sigma_smooth=sigma_smooth,
                                         sigma_app_color=sigma_app_color, sigma_app_pos=sigma_app_pos,
                                         sigma_struct_pos=sigma_struct_pos, sigma_struct_feat=sigma_struct_feat)
        else:
            crf_img = crf(img, prop_img, iter_num=crf_iter_num, compat_smooth=compat_smooth,
                                         compat_appearance=compat_appearance,
                                         w_smooth=w_smooth, w_appearance=w_appearance,
                                         sigma_smooth=sigma_smooth,
                                         sigma_app_color=sigma_app_color, sigma_app_pos=sigma_app_pos)
        if use_sm_crf:
            crf_img = crf_smooth(img, crf_img, iter_num=crf_sm_iter_num, w=crf_sm_w, sigma_sm=crf_sm_sigma)
        crf_list.append(prob_to_patch(crf_img))
    best_prob_score = 0
    best_prob_thres = 0
    best_prop_score = 0
    best_prop_thres = 0
    for threshold in [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        new_prob_score = 0
        new_prop_score = 0
        img_num = len(label_list)
        for i in range(img_num):
            label = label_list[i].flatten()
            prob = prob_list[i].flatten()
            prop = crf_list[i].flatten()
            prop = prop>threshold
            prob = prob>threshold
            new_prob_score = new_prob_score + accuracy_score(label, prob)/img_num
            new_prop_score = new_prop_score + accuracy_score(label, prop)/img_num
        if new_prob_score>best_prob_score:
            best_prob_score = new_prob_score
            best_prob_thres = threshold
        if new_prop_score>best_prop_score:
            best_prop_score = new_prop_score
            best_prop_thres = threshold
    print("best original pspnet score={}, threshold={}, best prop score={}, threshold={}".format(best_prob_score, best_prob_thres, best_prop_score, best_prop_thres))
    return best_prop_score


# In[11]:


def cross_validation_random():
    prop_num_and_window_sizes = [[16, 5], [32, 5], [16, 11],[32, 11],[16, 21], [24, 21]]
    sigma_2s = [10,20,40,80]
    crf_iter_nums = [1,2,4,8]
    sample_num = 500
    crf_sm_iter_nums = [0,1,2,4]
    with_prop=True
    with_struct_kernel = True
    use_sm_crf=False
    best_score = 0.5
    best_i = 0
    if with_prop:
        print("with prop")
    if with_struct_kernel:
        print("with kernel")
    for i, delta in enumerate([-0.2,-0.1,0,0.1,0.2]):
        p_w = random.sample(prop_num_and_window_sizes,1)[0]
        """
        prop_num = p_w[0]
        winow_size = p_w[1]
        sigma_prop_color = 0.5+ 1.5*np.random.rand() # sigma_1 \in [0.5, 2]
        sigma_prop_pos = random.sample(sigma_2s,1)[0]
        crf_iter_num = random.sample(crf_iter_nums,1)[0]
        compat_smooth = np.array([[-0.5+2*np.random.rand(), 0.5+np.random.rand()],[0.5+np.random.rand(), -0.5+np.random.rand()]])
        compat_appearance = np.array([[-0.5+np.random.rand(), 0.5+np.random.rand()],[0.5+np.random.rand(), -0.5+np.random.rand()]])
        compat_struct = np.array([[-0.5+np.random.rand(), np.random.rand()],[np.random.rand(), -10+10*np.random.rand()]])
        w_smooth = 1 + 3*np.random.rand()
        w_appearance = 1 + np.random.rand()
        w_struct = 1 + 5*np.random.rand()
        sigma_smooth= 1 + 9*np.random.rand()
        sigma_app_color = 1+1.5*np.random.rand()
        sigma_app_pos = random.sample(sigma_2s,1)[0]
        sigma_struct_pos = 100 +400*np.random.rand()
        sigma_struct_feat = 10 + 40*np.random.rand()
        """
        prop_num=16
        window_size=5
        sigma_prop_color=0.713060162472891+delta
        sigma_prop_pos=20
        crf_iter_num=8
        compat_smooth=np.array([[1.26958173,  0.8307862],[0.71450311, -0.35393216]])
        compat_appearance=np.array([[-0.05790994,  1.3304376],[1.32379982, -0.00975676]])
        compat_struct=np.array([[-0.40640465,  0.79798697],[0.99422088, -2.98206395]])
        w_smooth=1.2092231919137424
        w_appearance=1.77501181382847
        w_struct=1.4917858524385004
        sigma_smooth=2.115722641303978
        sigma_app_color=2.043231728337113
        sigma_app_pos=40
        sigma_struct_pos=173.1556405794435
        sigma_struct_feat=17.572299710793928
        
        crf_sm_iter_num = random.sample(crf_sm_iter_nums,1)[0]
        crf_sm_w = 1+4*np.random.rand()
        crf_sm_sigma = 1+9*np.random.rand()
        
        print("Iter {} Parameters:".format(i))
        if with_prop:
            print("prop_num={}, prop_size={}, sigma_prop_color={}, sigma_prop_pos={}".format(prop_num, window_size, sigma_prop_color, sigma_prop_pos))
        print("crf_iter_num={},compat_smooth={},compat_appearance={} w_smooth={}, w_appearance={}".format(crf_iter_num, compat_smooth.flatten(), compat_appearance.flatten(), w_smooth, w_appearance))
        print("sigma_smooth={}, sigma_app_color={}, sigma_app_pos={}".format(sigma_smooth,sigma_app_color,sigma_app_pos))
        if with_struct_kernel:
            print("compat_struct={},w_struct={},sigma_struct_pos={}, sigma_struct_feat={}".format(compat_struct.flatten(),
                                                                                                                      w_struct, sigma_struct_pos,
                                                                                                                      sigma_struct_feat))
        if use_sm_crf:
            print("sm iter={}, w={}, sigma={}".format(crf_sm_iter_num, crf_sm_w, crf_sm_sigma))
        score = compute_score_full_param(prop_num, window_size, sigma_prop_color, sigma_prop_pos,
                             crf_iter_num, compat_smooth, compat_appearance, compat_struct, 
                             w_smooth, w_appearance, w_struct,
                             sigma_smooth, sigma_app_color, sigma_app_pos,
                             sigma_struct_pos, sigma_struct_feat,
                             crf_sm_iter_num, crf_sm_w, crf_sm_sigma,
                             with_prop=with_prop, with_struct_kernel=with_struct_kernel, use_sm_crf=use_sm_crf, 
                             cv_label_path = './groundtruth/', cv_img_path = './images/', 
                             test_prob_path = "./epoch2800_eval/", dir_path = './dir_map_train/')
        if score>best_score:
            best_score=score
            best_i=i
        print("best score: {} at iter {}".format(best_score, best_i))
        print()


# In[29]:


def compute_score_for_tuned_parameters(with_prop=False, with_struct_kernel=False):
    if with_prop:
        if with_struct_kernel:
            compute_score_full_param(prop_num=16, window_size=5, sigma_prop_color=0.713060162472891, sigma_prop_pos=20,
                                     crf_iter_num=4, compat_smooth=np.array([[1.46958173,  0.8307862],[0.71450311, -0.35393216]]), 
                                     compat_appearance=np.array([[-0.05790994,  1.3304376],[1.32379982, -0.00975676]]), 
                                     compat_struct=np.array([[-0.40640465,  0.79798697],[0.99422088, -3.48206395]]), 
                                     w_smooth=1.2092231919137424, w_appearance=1.77501181382847, w_struct=1.2417858524385004,
                                     sigma_smooth=2.315722641303978, sigma_app_color=2.043231728337113, sigma_app_pos=40,
                                     sigma_struct_pos=173.1556405794435, sigma_struct_feat=2.572299710793928,
                                     with_prop=with_prop, with_struct_kernel=with_struct_kernel, use_sm_crf=False, 
                                     cv_label_path = './cv_labels/', cv_img_path = './cv_images/', 
                                     test_prob_path = "./cv_probs/", dir_path = './dir_map_test/')
        else:
            compute_score_full_param(prop_num=16, window_size=5, sigma_prop_color=0.6650886795908088, sigma_prop_pos=10,
                                     crf_iter_num=8, compat_smooth=np.array([[0.24658912 , 1.03629577],[0.69663063, -0.31590588]]), 
                                     compat_appearance=np.array([[-0.31513215,  0.97190996],[1.04959312, -0.47501959]]), 
                                     w_smooth=1.828328302038134, w_appearance=1.795302766064866,
                                     sigma_smooth=1.435892752213356, sigma_app_color=1.78496352847059, sigma_app_pos=80,
                                     with_prop=with_prop, with_struct_kernel=with_struct_kernel, use_sm_crf=False, 
                                     cv_label_path = './cv_labels/', cv_img_path = './cv_images/', 
                                     test_prob_path = "./cv_probs/", dir_path = './dir_map_test/')
    else:
        if with_struct_kernel:
            compute_score_full_param(crf_iter_num=8, compat_smooth=np.array([[0.2704138 ,  1.09232546],[0.80253412, -0.18487427]]), 
                                     compat_appearance=np.array([[-0.37001789,  0.85249872],[1.29555175, -0.2937206]]), 
                                     compat_struct=np.array([[0.1788232 ,  0.61148446],[0.1116445,  -4.44564896]]), 
                                     w_smooth=1.6814390659156584, w_appearance=1.83980578425931, w_struct=1.3154124820494024,
                                     sigma_smooth=5.692475296551731, sigma_app_color=1.5828168297951695, sigma_app_pos=40,
                                     sigma_struct_pos=264.5010753324061, sigma_struct_feat=29.132062312611474,
                                     with_prop=with_prop, with_struct_kernel=with_struct_kernel, use_sm_crf=False, 
                                     cv_label_path = './cv_labels/', cv_img_path = './cv_images/', 
                                     test_prob_path = "./cv_probs/", dir_path = './dir_map_test/')
        else:
            compute_score_full_param(crf_iter_num=8, compat_smooth=np.array([[0.88830681,  0.69689981],[0.54353049, -0.1542836]]), 
                                         compat_appearance=np.array([[-0.49690445,  1.15925799],[1.22089288, -0.34833315]]), 
                                         w_smooth=2.2810288259551967, w_appearance=1.90286829269048,
                                         sigma_smooth=8.053041617053246, sigma_app_color=1.6955329962509278, sigma_app_pos=80,
                                         with_prop=with_prop, with_struct_kernel=with_struct_kernel, use_sm_crf=False, 
                                         cv_label_path = './cv_labels/', cv_img_path = './cv_images/', 
                                         test_prob_path = "./cv_probs/", dir_path = './dir_map_test/')


# In[30]:


# cross_validation_random()
compute_score_for_tuned_parameters(with_prop=True, with_struct_kernel=True)

