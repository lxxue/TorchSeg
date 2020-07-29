
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


# In[3]:


def post_processing(image_path = './test_images/',vis_path = './results/vis_crf',
                    dir_path = './dir_map_test/', prob_path = './epoch3000_test/',
                    out_path = './results/crf', with_prop=False, with_kernel=False):
    if with_prop:
        vis_path = vis_path + "_prop"
        out_path = out_path + "_prop"
    if with_kernel:
        vis_path = vis_path + "_kernel"
        out_path = out_path + "_kernel"
    for path in [vis_path, out_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    for file in tqdm(os.listdir(image_path)):
        img = cv2.imread(os.path.join(image_path, file))
        prob = cv2.imread(os.path.join(prob_path, file))
        prob=prob/255
        prob=prob[:,:,0]
        dir_color_map = cv2.imread(os.path.join(dir_path, file))
        dir_feature = dir_to_features(dir_color_map)
        if with_prop:
            if with_kernel:
                """
                # 3000
                prop_img = propagate_max_vec(img,prob,prop_num=16, prop_size=5, 
                                             sigma_color=1.6935536588645517, sigma_pos=20)
                crf_img = crf_with_dir_kernel(img, dir_feature, prop_img, iter_num=12, 
                                         compat_smooth=np.array([[0.57842984, 0.95404739],[0.52004501, 0.40422152]]),
                                         compat_appearance=np.array([[-0.31741694,  1.18494974],[1.12442186,  0.24456809]]),
                                         compat_struct=np.array([[-0.30595129,  0.7728254],[0.12691528, -0.85317293]]),
                                         w_smooth=1.6516782004079098, w_appearance=1.529408010866331,
                                         w_struct=4.938469494095774, sigma_smooth=4.883141908483301, 
                                         sigma_app_color=1.5137269619243123, sigma_app_pos=40,
                                         sigma_struct_pos=475.9932426424997, sigma_struct_feat=40.37248405378539)
                """
                # 2800
                prop_img = propagate_max_vec(img,prob,prop_num=32, prop_size=11, 
                                             sigma_color=0.9558387244727098, sigma_pos=10)
                crf_img = crf_with_dir_kernel(img, dir_feature, prop_img, iter_num=8, 
                                         compat_smooth=np.array([[0.63928644, 0.7558237],[1.24894913, 0.12096477]]),
                                         compat_appearance=np.array([[-0.21364454 , 0.86426312],[0.81418789 , 0.16731218]]),
                                         compat_struct=np.array([[-0.43469763 , 0.9034787],[0.55590722 ,-0.09176918]]),
                                         w_smooth=1.0807095786024483, w_appearance=1.1901483892457647,
                                         w_struct=2.942751798533171, sigma_smooth=5.899771615934947,
                                         sigma_app_color=2.4522940567843348, sigma_app_pos=80,
                                         sigma_struct_pos=187.99058882436577, sigma_struct_feat=14.469581394524806)


                
            else:
                prop_img = propagate_max_vec(img,prob,prop_num=32, prop_size=5, 
                                             sigma_color=0.8648530839870829, sigma_pos=10)
                crf_img = crf(img, prop_img, iter_num=8, compat_smooth=np.array([[-0.27312429,  1.44344053],[0.94756049, -0.05987938]]),
                                         compat_appearance=np.array([[0.30314967, 0.93767007],[1.37851458, 0.14069547]]),
                                         w_smooth=2.622061363416207, w_appearance=1.5669198261263837,
                                         sigma_smooth=7.2716251558091365,
                                         sigma_app_color=2.0988905139201224, sigma_app_pos=10)
        else:
            if with_kernel:
                crf_img = crf_with_dir_kernel(img, dir_feature, prob, iter_num=8, 
                                         compat_smooth=np.array([[1.42628086, 0.57972011],[1.36363985, 0.21731918]]),
                                         compat_appearance=np.array([[-0.43596408,  1.04737195],[1.39793517,  0.07055685]]),
                                         compat_struct=np.array([[-0.32739809,  0.45224483],[0.7109288,  -0.35132453]]),
                                         w_smooth=2.1670176734442625, w_appearance=1.699234108202612,
                                         w_struct=4.818667137623257, sigma_smooth=4.835584315772047,
                                         sigma_app_color=2.0611781853345224, sigma_app_pos=10,
                                         sigma_struct_pos=174.76741408433867, sigma_struct_feat=48.092921007782984)
            else:

                """
                # 2800
                                crf_img = crf(img, prob, iter_num=8, compat_smooth=np.array([[-0.43588055,  1.49743748],[1.08083823 , 0.48265268]]),
                              compat_appearance=np.array([[-0.30344363 , 0.62247936],[0.9058573,   0.26622347]]),
                              w_smooth=3.8508739815607615, w_appearance=1.8400845974535944,
                              sigma_smooth=4.980895096538283,
                              sigma_app_color=1.150807401052397, sigma_app_pos=80)
                """
                crf_img = crf(img, prob, iter_num=12, compat_smooth = np.array([[-0.4946432,  1.27117338],[0.59452892, 0.23182234]]), 
                            compat_appearance = np.array([[-0.30571318,  0.83015124],[1.3217825,  -0.13046645]]), 
                            w_smooth=3.7946478055761963, w_appearance=1.8458537690881878,
                            sigma_smooth=8.575103751642672, sigma_app_color=2.0738539891571977, sigma_app_pos=20)
                
        cv2.imwrite(os.path.join(out_path,file), gray2rgb(np.floor(crf_img*255)))
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.imshow(img); plt.title('Original image'); plt.axis('off');
        plt.subplot(1,3,2); plt.imshow(gray2rgb(prob)); plt.title('PSP prob'); plt.axis('off');
        plt.subplot(1,3,3); plt.imshow(gray2rgb(crf_img)); plt.title('CRF prob'); plt.axis('off');
        plt.savefig(os.path.join(vis_path, file),dpi=300)
        plt.close()


# In[4]:


def post_processing_local_test(image_path = './test_images/',vis_path = './results/vis_crf',
                    dir_path = './dir_map_test/', prob_path = './epoch2800_test/',
                    out_path = './results/crf', with_prop=False, with_kernel=False):
    if with_prop:
        vis_path = vis_path + "_prop"
        out_path = out_path + "_prop"
    if with_kernel:
        vis_path = vis_path + "_kernel"
        out_path = out_path + "_kernel"
    for path in [vis_path, out_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    for file in tqdm(os.listdir(image_path)):
        img = cv2.imread(os.path.join(image_path, file))
        prob = cv2.imread(os.path.join(prob_path, file))
        prob=prob/255
        prob=prob[:,:,0]
        dir_color_map = cv2.imread(os.path.join(dir_path, file))
        dir_feature = dir_to_features(dir_color_map)
        if with_prop:
            if with_kernel:
                prop_img = propagate_max_vec(img,prob,prop_num=16, prop_size=5, 
                                             sigma_color=0.713060162472891, sigma_pos=20)
                crf_img = crf_with_dir_kernel(img, dir_feature, prop_img, iter_num=4, 
                                         compat_smooth=np.array([[1.46958173,  0.8307862],[0.71450311, -0.35393216]]),
                                         compat_appearance=np.array([[-0.05790994,  1.3304376],[1.32379982, -0.00975676]]),
                                         compat_struct=np.array([[-0.40640465,  0.79798697],[0.99422088, -3.48206395]]),
                                         w_smooth=1.2092231919137424, w_appearance=1.77501181382847,
                                         w_struct=1.2417858524385004, sigma_smooth=2.315722641303978, 
                                         sigma_app_color=2.043231728337113, sigma_app_pos=40,
                                         sigma_struct_pos=173.1556405794435, sigma_struct_feat=2.572299710793928)
                
            else:
                prop_img = propagate_max_vec(img,prob,prop_num=16, prop_size=5, 
                                             sigma_color=0.6650886795908088, sigma_pos=10)
                crf_img = crf(img, prop_img, iter_num=8, compat_smooth=np.array([[0.24658912 , 1.03629577],[0.69663063,-0.31590588]]),
                                         compat_appearance=np.array([[-0.31513215,  0.97190996],[1.04959312, -0.47501959]]),
                                         w_smooth=1.828328302038134, w_appearance=1.795302766064866,
                                         sigma_smooth=1.435892752213356, sigma_app_color=1.78496352847059, sigma_app_pos=80)
        else:
            if with_kernel:
                crf_img = crf_with_dir_kernel(img, dir_feature, prob, iter_num=8, 
                                         compat_smooth=np.array([[0.2704138,   1.09232546],[0.80253412, -0.18487427]]),
                                         compat_appearance=np.array([[-0.37001789,  0.852498725],[1.29555175, -0.2937206]]),
                                         compat_struct=np.array([[0.1788232,   0.61148446],[0.1116445,  -4.44564896]]),
                                         w_smooth=1.6814390659156584, w_appearance=1.83980578425931,
                                         w_struct=1.3154124820494024, sigma_smooth=5.692475296551731, 
                                         sigma_app_color=1.5828168297951695, sigma_app_pos=40,
                                         sigma_struct_pos=264.5010753324061, sigma_struct_feat=9.132062312611474)
            else:
                crf_img = crf(img, prob, iter_num=8, compat_smooth=np.array([[0.88830681,  0.69689981],[0.54353049, -0.1542836]]),
                              compat_appearance=np.array([[-0.49690445,  1.15925799],[1.22089288, -0.34833315]]),
                              w_smooth=2.2810288259551967, w_appearance=1.90286829269048,
                              sigma_smooth=8.053041617053246, sigma_app_color=1.6955329962509278, sigma_app_pos=80)
                
        cv2.imwrite(os.path.join(out_path,file), gray2rgb(np.floor(crf_img*255)))
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.imshow(img); plt.title('Original image'); plt.axis('off');
        plt.subplot(1,3,2); plt.imshow(gray2rgb(prob)); plt.title('PSP prob'); plt.axis('off');
        plt.subplot(1,3,3); plt.imshow(gray2rgb(crf_img)); plt.title('CRF prob'); plt.axis('off');
        plt.savefig(os.path.join(vis_path, file),dpi=300)
        plt.close()


# In[5]:


def voting_gen(image_path = './test_images/', dir_path = './dir_map_test/', prob_base_path='./train_all/', out_path='./crf_all/'):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for model_name in os.listdir(prob_base_path):
        print("Processing {}".format(model_name))
        model_out_path = os.path.join(out_path, model_name)
        model_prob_path = os.path.join(prob_base_path, model_name)
        if not os.path.exists(model_out_path):
            os.mkdir(model_out_path)
        out_path_3000 = os.path.join(model_out_path, "param_3000")
        out_path_2800 = os.path.join(model_out_path, "param_2800")
        for path in [out_path_2800, out_path_3000]:
            if not os.path.exists(path):
                os.mkdir(path)
        for file in tqdm(os.listdir(image_path)):
            img = cv2.imread(os.path.join(image_path, file))
            prob = cv2.imread(os.path.join(model_prob_path, file))
            prob=prob/255
            prob=prob[:,:,0]
            dir_color_map = cv2.imread(os.path.join(dir_path, file))
            dir_feature = dir_to_features(dir_color_map)
            # 3000 parameters
            prop_img = propagate_max_vec(img,prob,prop_num=16, prop_size=5, 
                                             sigma_color=1.6935536588645517, sigma_pos=20)
            crf_img = crf_with_dir_kernel(img, dir_feature, prop_img, iter_num=12, 
                                         compat_smooth=np.array([[0.57842984, 0.95404739],[0.52004501, 0.40422152]]),
                                         compat_appearance=np.array([[-0.31741694,  1.18494974],[1.12442186,  0.24456809]]),
                                         compat_struct=np.array([[-0.30595129,  0.7728254],[0.12691528, -0.85317293]]),
                                         w_smooth=1.6516782004079098, w_appearance=1.529408010866331,
                                         w_struct=4.938469494095774, sigma_smooth=4.883141908483301, 
                                         sigma_app_color=1.5137269619243123, sigma_app_pos=40,
                                         sigma_struct_pos=475.9932426424997, sigma_struct_feat=40.37248405378539)
                                        # change sigma_struct_feat = 5.37248405378539 and we get the third param
            cv2.imwrite(os.path.join(out_path_3000,file), gray2rgb(np.floor(crf_img*255)))
            
            # 2800 parameters
            prop_img = propagate_max_vec(img,prob,prop_num=32, prop_size=11, 
                                             sigma_color=0.9558387244727098, sigma_pos=10)
            crf_img = crf_with_dir_kernel(img, dir_feature, prop_img, iter_num=8, 
                                         compat_smooth=np.array([[0.63928644, 0.7558237],[1.24894913, 0.12096477]]),
                                         compat_appearance=np.array([[-0.21364454 , 0.86426312],[0.81418789 , 0.16731218]]),
                                         compat_struct=np.array([[-0.43469763 , 0.9034787],[0.55590722 ,-0.09176918]]),
                                         w_smooth=1.0807095786024483, w_appearance=1.1901483892457647,
                                         w_struct=2.942751798533171, sigma_smooth=5.899771615934947,
                                         sigma_app_color=2.4522940567843348, sigma_app_pos=80,
                                         sigma_struct_pos=187.99058882436577, sigma_struct_feat=14.469581394524806)
                                        # change sigma_struct_feat = 4.469581394524806 and we get the fourth param
            cv2.imwrite(os.path.join(out_path_2800,file), gray2rgb(np.floor(crf_img*255)))
                


# In[40]:


def voting(image_path = './test_images/', use_extra_param=True, out_path="./crf_vote/", vis_path="./vis_vote/",thres = 0.2):
    out_thres_path = "./crf_vote_thres_{}".format(thres)
    for path in [out_thres_path, vis_path, out_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    prob_path_list = []
    for path in os.listdir("./crf_all/"):
        for sub_path in os.listdir(os.path.join("./crf_all/", path)):
            prob_path_list.append(os.path.join("./crf_all/", path, sub_path))
    if use_extra_param:
        for path in os.listdir("./crf_all_2/"):
            for sub_path in os.listdir(os.path.join("./crf_all_2/", path)):
                prob_path_list.append(os.path.join("./crf_all_2/", path, sub_path))
    vote_len = len(prob_path_list)
    print("Voting with {} models".format(vote_len))
    for file in tqdm(os.listdir(image_path)):
        prob_vote = np.zeros((608,608))
        img = cv2.imread(os.path.join(image_path, file))
        for prob_path in prob_path_list:
            prob = cv2.imread(os.path.join(prob_path, file))
            prob = prob[:,:,0]
            # prob_vote= np.maximum(prob_vote,prob)
            prob_vote=prob_vote+prob
        prob_vote=prob_vote/vote_len
        cv2.imwrite(os.path.join(out_path,file), gray2rgb(np.floor(prob_vote)))
        prob_vote[prob_vote<thres*255]=0
        prob_vote[prob_vote>=thres*255]=255
        cv2.imwrite(os.path.join(out_thres_path,file), gray2rgb(np.floor(prob_vote)))
        """
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1); plt.imshow(img); plt.title('Original image'); plt.axis('off');
        plt.subplot(1,2,2); plt.imshow(gray2rgb(prob_vote/255)); plt.title('CRF prob'); plt.axis('off');
        plt.savefig(os.path.join(vis_path, file),dpi=300)
        plt.close()
        """


# In[41]:


#post_processing(image_path = './test_images/',vis_path = './results_sanqian_eval/vis_crf_no_parking',
#                dir_path = './dir_map_test/', prob_path = './epoch3000_test/',
#                out_path = './results_sanqian_eval/crf_no_parking', with_prop=False, with_kernel=False)
# voting_gen()
voting()

