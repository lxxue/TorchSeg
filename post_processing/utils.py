
# coding: utf-8

# In[20]:


import numpy as np
import pydensecrf.densecrf as dcrf
import os
import cv2
import random
from tqdm import tqdm


# In[21]:


from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
#from osgeo import gdal
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


# Color maps for direction map
COLOR_LR = [0,128,128]
COLOR_UD = [128,0,128]
COLOR_DIAG = [255,215,0]
COLOR_ADIAG = [1,255,255]
INF = 10000


# In[23]:


MAX = 0
SUM = 1
VEC = 0
MAT = 1


# In[24]:


def dir_to_features(dir_map):
    """Converts direction color map to feature used for crf kernel. The
    feature is obtained by computing the intersections of the x, y axis and the
    line determined by the position of one point and its direction. (More details in
    the report)
    
    Parameters
    ____________
    dir_map: numpy.array
        Direction map that maps each pixel to a direction in 
        [left_right, up_down, diagonal, anti-diagonal], each direction
        is represented by a color.
    """
    (h, w, c) = dir_map.shape
    feature_map = np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            dir_color = dir_map[i,j]
            if dir_color[0] == COLOR_LR[0]: # dir = lr
                feature_map[i,j] = np.array([INF,i])
            if dir_color[0] == COLOR_UP[0]: # dir = ud
                feature_map[i,j] = np.array([j,INF])
            if dir_color[1] == COLOR_DIAG[0]: # dir = diag
                feature_map[i,j] = np.array([j-i,i-j])
            if dir_color[1] == COLOR_ADIAG[0]: # dir = adiag
                feature_map[i,j] = np.array([i+j, i+j])
    return feature_map


# In[25]:


def gen_dir_map(img):
    """Generate direction map from a rgb img
    
    Parameters
    ____________
    img: numpy.array
        Rgb img with width = height
    """
    window_size = 101
    half_size = int((window_size-1)/2)
    sigma_1 = 2
    sigma_2 = 40
    (h, w, c) = img.shape
    assert h==w, "h and w are not equal"
    dir_map = np.zeros((h,w))
    pos_mat = np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            pos_mat[i,j,0]=i
            pos_mat[i,j,1]=j
            
    padded_pos = np.pad(pos_mat, ((half_size, half_size), (half_size, half_size), (0,0)))
    padded_img = np.pad(img, ((half_size, half_size), (half_size, half_size), (0,0)))
    
    index_mask_lr = np.zeros((window_size, window_size)).astype("bool")
    index_mask_lr[half_size,:]=True
    index_mask_ud = np.zeros((window_size, window_size)).astype("bool")
    index_mask_ud[:,half_size]=True
    index_mask_diag = np.identity(window_size).astype("bool")
    index_mask_adiag = np.fliplr(np.identity(window_size)).astype("bool")
    mask_list = [index_mask_lr, index_mask_ud, index_mask_diag, index_mask_adiag]
    for i in range(h):
        for j in range(w):
            img_nbr = padded_img[i:i+window_size,j:j+window_size]
            pos_nbr = padded_pos[i:i+window_size,j:j+window_size]
            img_nbr = img_nbr - img[i,j,:]
            pos_nbr = pos_nbr - np.array([i,j])
            dir_intensity = np.zeros(4)
            for dir_index, index_mask in enumerate(mask_list):
                img_nbr_dir = img_nbr[index_mask]
                pos_nbr_dir = pos_nbr[index_mask]
                img_nbr_dir = np.sum(img_nbr_dir**2, axis=1)/(2*sigma_1**2)
                pos_nbr_dir = np.sum(pos_nbr_dir**2, axis=1)/(2*sigma_2**2)
                k = np.exp(-img_nbr_dir-pos_nbr_dir)
                dir_intensity[dir_index]=np.sum(k)
            dir_map[i,j]=np.argmax(dir_intensity)+1
    return dir_map


# In[26]:


def visualize_dir_map(img, dir_map, save_file=False, 
                      filename=None, vis_path=None, dir_path=None):
    """Visualize a direction map
    
    Parameters
    ____________
    img: numpy.array
        Rgb img
    dir_map: numpy.array
        Correspongding direction map
    ...
    """
    h = img.shape[0]
    w = img.shape[1]
    vis_dir = np.zeros(img.shape)
    vis_dir[dir_map==1] = np.array(COLOR_LR)
    vis_dir[dir_map==2] = np.array(COLOR_UD)
    vis_dir[dir_map==3] = np.array(COLOR_DIAG)
    vis_dir[dir_map==4] = np.array(COLOR_ADIAG)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(img); plt.title('Original Image (blurred)'); plt.axis('off');
    plt.subplot(1,2,2); plt.imshow(dir_map); plt.title('Direction map'); plt.axis('off');
    if save_file:
        plt.savefig(os.path.join(vis_path, filename),dpi=300)
        plt.close()
        cv2.imwrite(os.path.join(dir_path, filename), vis_dir)


# In[27]:


def gen_dir_map_and_visualize(image_path= './images/',
                              vis_path='./vis_dir_blur_/',
                              dir_path='./dir_map_/',
                              process_all=True):
    """Generate direction color map for images in image_path
    
    Parameters
    ____________
    image_path: string
        Image path
    vis_path: string
        Path to save visualization results
    dir_path: string
        Path to save direction map
    process_all: Bool
        False to generate a single visualization result without save. True to 
        generate and save visualizaiton results for all images.
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    if process_all:
        for file in tqdm(os.listdir(image_path)):
            img = cv2.imread(os.path.join(image_path, file))
            img = cv2.GaussianBlur(img,(5,5),0)
            dir_map = gen_dir_map(img)
            visualize_dir_map(img, dir_map, filename=file, save_file=True, 
                              vis_path=vis_path, dir_path=dir_path)
    else:
        img = cv2.imread('./images/satImage_001.png')
        img = cv2.GaussianBlur(img,(5,5),0)
        dir_map = gen_dir_map(img)
        visualize_dir_map(img, dir_map, save_file=False)


# In[28]:


def crf_with_dir_kernel(original_img, dir_feature, prob, 
                        iter_num, compat_smooth, compat_appearance, compat_struct,
                        w_smooth, w_appearance, w_struct,
                        sigma_smooth, sigma_app_color, sigma_app_pos,
                        sigma_struct_pos, sigma_struct_feat):
    """CRF with a Gaussian smoothing kernel, an appearance kernel and a structural kernel
    
    """
    (h,w) = prob.shape
    y = np.zeros((h,w,2))
    y[:,:,1] = prob
    y[:,:,0] = 1-y[:,:,1]
    annotated_image=y.transpose((2, 0, 1))
    #Gives no of class labels in the annotated image
    n_labels = 2

    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_img.shape[1], original_img.shape[0], n_labels)
        
    # get unary potentials (neg log probability)
        
    U = unary_from_softmax(annotated_image)
    unary = np.ascontiguousarray(U)
    d.setUnaryEnergy(unary)
    
    compat_smooth = compat_smooth * w_smooth
    compat_appearance = compat_appearance * w_appearance
    compat_struct = compat_struct * w_struct
    
    # Smooth kernel
    d.addPairwiseGaussian(sxy=(sigma_smooth, sigma_smooth), compat=compat_smooth.astype(np.float32), 
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Appearance kernel
    d.addPairwiseBilateral(sxy=(sigma_app_pos, sigma_app_pos), 
                           srgb=(sigma_app_color, sigma_app_color, sigma_app_color), 
                           rgbim=original_image,
                           compat=compat_appearance.astype(np.float32),
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Structural kernel
    pairwise_energy = create_pairwise_bilateral(sdims=(sigma_struct_pos,sigma_struct_pos), 
                                                schan=(sigma_struct_feat,sigma_struct_feat), 
                                                img=dir_feature, chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=compat_struct.astype(np.float32))
    
    Q = d.inference(iter_num)
    proba = np.array(Q)
    return proba[1].reshape((dir_feature.shape[0], dir_feature.shape[1]))


# In[29]:


def crf(original_image, prob,
        iter_num=4, compat_smooth = np.array([[-0.4946432,  1.27117338],[0.59452892, 0.23182234]]), 
        compat_appearance = np.array([[-0.30571318,  0.83015124],[1.3217825,  -0.13046645]]), 
        w_smooth=3.7946478055761963, w_appearance=1.8458537690881878,
        sigma_smooth=8.575103751642672, sigma_color=2.0738539891571977, sigma_color_pos=20):
    """Basic CRF with a Gaussian smoothing kernel and an appearance kernel
    
    """
    (h,w) = prob.shape
    y = np.zeros((h,w,2))
    y[:,:,1] = prob
    y[:,:,0] = 1-y[:,:,1]
    annotated_image=y.transpose((2, 0, 1))
    #Gives no of class labels in the annotated image
    n_labels = 2
    
    #print("No of labels in the Image are ")
    #print(n_labels)
        
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
        
    # get unary potentials (neg log probability)
        
    U = unary_from_softmax(annotated_image)
    unary = np.ascontiguousarray(U)
    d.setUnaryEnergy(unary)
    compat_smooth=compat_smooth*w_smooth
    compat_appearance=compat_appearance*w_appearance
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(sigma_smooth, sigma_smooth), compat=compat_smooth.astype(np.float32), kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(sigma_color_pos, sigma_color_pos), srgb=(sigma_color, sigma_color, sigma_color), rgbim=original_image,
                           compat=compat_appearance.astype(np.float32),
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(iter_num)
    proba = np.array(Q)
    return proba[1].reshape((original_image.shape[0], original_image.shape[1]))


# In[30]:


def crf_smooth(original_image, prob, use_2d = True, iter_num=1, w=4.921522279119057, sigma_sm=4.325251720130304):
    """CRF with only a smoothing kernel
    
    """
    (h,w) = prob.shape
    y = np.zeros((h,w,2))
    y[:,:,1] = prob
    y[:,:,0] = 1-y[:,:,1]
    annotated_image=y.transpose((2, 0, 1))
    #Gives no of class labels in the annotated image
    n_labels = 2
    
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
        
        # get unary potentials (neg log probability)
        
        U = unary_from_softmax(annotated_image)
        unary = np.ascontiguousarray(U)
        d.setUnaryEnergy(unary)
        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(sigma_sm, sigma_sm), compat=w, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(iter_num)
    proba = np.array(Q)
    return proba[1].reshape((original_image.shape[0], original_image.shape[1]))


# In[31]:


def propagate_max_mat(img, prob):
    """Probability propagation (max) in 4 directions via matrix multiplication
    """
    prob_out = prob.copy()
    prop_size = 51
    half_size = int((prop_size-1)/2)
    prop_num = 3
    sigma_1 = 5
    sigma_2 = 42
    (h, w) = prob.shape
    
    pos_mat = np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            pos_mat[i,j,0]=i
            pos_mat[i,j,1]=j
            
    padded_pos = np.pad(pos_mat, ((half_size, half_size), (half_size, half_size), (0,0)))
    padded_img = np.pad(img, ((half_size, half_size), (half_size, half_size), (0,0)))
    
    index_mask = np.zeros((prop_size, prop_size)).astype("bool")
    for i in range(prop_size):
        index_mask[i,half_size]=1
        index_mask[half_size,i]=1
        index_mask[i,i]=1
        index_mask[prop_size-1-i,i]=1
        
    for iteration in range(prop_num):
        padded_prob = np.pad(prob_out, ((half_size, half_size), (half_size, half_size)))
        # propagate prob (maximum)
        for i in range(h):
            for j in range(w):
                if prob_out[i,j]<0.01:
                    continue
                img_nbr = padded_img[i:i+prop_size,j:j+prop_size]
                pos_nbr = padded_pos[i:i+prop_size,j:j+prop_size]
                img_nbr = img_nbr - img[i,j,:]
                pos_nbr = pos_nbr - np.array([i,j])
                img_nbr[~index_mask]=0
                pos_nbr[~index_mask]=0
                img_nbr = np.sum(img_nbr**2, axis=2)/(2*sigma_1**2)
                pos_nbr = np.sum(pos_nbr**2, axis=2)/(2*sigma_2**2)
                k = np.exp(-img_nbr-pos_nbr)*prob_out[i,j]
                k = k*index_mask
                padded_prob[i:i+prop_size,j:j+prop_size] = np.maximum(padded_prob[i:i+prop_size,j:j+prop_size], k)
        prob_out = padded_prob[half_size:h+half_size,half_size:w+half_size]
        
    return prob_out


# In[32]:


def propagate_max_vec(img, prob, prop_size=11, 
                      prop_num=16, sigma_1=1.039316347691348, sigma_2=40):
    """
    vec means only do propagation along x and y axis
    max means propagate using max function
    Args:
        prop_size: neighborhood size
        prop_num: number of iteration/propagation
        sigma_1: variance of color
        sigma_2: variance of distance
    """
    prob_out = prob.copy()
    half_size = int((prop_size-1)/2)
    (h, w, c) = img.shape
    
    pos_mat = np.zeros((h,w,2))        # position matrix
    for i in range(h):
        for j in range(w):
            pos_mat[i,j,0]=i
            pos_mat[i,j,1]=j
            
    padded_pos = np.pad(pos_mat, ((half_size, half_size), (half_size, half_size), (0,0)))
    padded_img = np.pad(img, ((half_size, half_size), (half_size, half_size), (0,0)))
        
    for iteration in range(prop_num):
        padded_prob = np.pad(prob_out, ((half_size, half_size), (half_size, half_size)))
        padded_prob_fix = padded_prob.copy()
        # propagate prob (maximum)
        assert h==w, "h and w are not equal"
        for i in range(h):
            # prop along y for row i
            img_nbr = padded_img[i:i+prop_size,:]
            pos_nbr = padded_pos[i:i+prop_size,:]
            img_nbr = img_nbr - padded_img[i+half_size,:,:]
            pos_nbr = pos_nbr - padded_pos[i+half_size,:,:]
            img_nbr = np.sum(img_nbr**2, axis=2)/(2*sigma_1**2)
            pos_nbr = np.sum(pos_nbr**2, axis=2)/(2*sigma_2**2)
            k = np.exp(-img_nbr-pos_nbr)*padded_prob_fix[i+half_size,:]
            padded_prob[i:i+prop_size,:] = np.maximum(padded_prob[i:i+prop_size,:], k)
            
            # prop along x for col i
            img_nbr = padded_img[:,i:i+prop_size]
            pos_nbr = padded_pos[:,i:i+prop_size]
            img_nbr = img_nbr - padded_img[:,i+half_size,:].reshape((padded_img.shape[0],1,c))
            pos_nbr = pos_nbr - padded_pos[:,i+half_size,:].reshape((padded_img.shape[0],1,2))
            img_nbr = np.sum(img_nbr**2, axis=2)/(2*sigma_1**2)
            pos_nbr = np.sum(pos_nbr**2, axis=2)/(2*sigma_2**2)
            k = np.exp(-img_nbr-pos_nbr)*padded_prob_fix[:,i+half_size].reshape((-1,1))
            padded_prob[:,i:i+prop_size] = np.maximum(padded_prob[:,i:i+prop_size], k)
            
        prob_out = padded_prob[half_size:h+half_size,half_size:w+half_size]
        
    return prob_out


# In[33]:


def propagate_sum_vec(img, prob, prop_size=11, prop_num=1, sigma_1=1.5319569104856783, sigma_2=80):
    """
    vec means only do propagation along x and y axis
    sum means propagate in a additive schema (with total probability fixed)
    Args:
        prop_size: neighborhood size
        prop_num: number of iteration/propagation
        sigma_1: variance of color
        sigma_2: variance of distance
    """
    # print(np.sum(prob))
    prob_out = prob.copy()
    half_size = int((prop_size-1)/2)
    (h, w, c) = img.shape
    
    pos_mat = np.zeros((h,w,2))        # position matrix
    for i in range(h):
        for j in range(w):
            pos_mat[i,j,0]=i
            pos_mat[i,j,1]=j
            
    padded_pos = np.pad(pos_mat, ((half_size, half_size), (half_size, half_size), (0,0)))
    padded_img = np.pad(img, ((half_size, half_size), (half_size, half_size), (0,0)))
    padded_prob = np.pad(prob, ((half_size, half_size), (half_size, half_size)))
    for iteration in range(prop_num):
        padded_prob_fix = padded_prob.copy()
        padded_prob = np.pad(np.zeros((h,w)), ((half_size, half_size), (half_size, half_size)))
        # propagate prob (sum)
        assert h==w, "h and w are not equal"
        # compute the degree mat
        deg_mat = np.zeros((h+2*half_size,w+2*half_size))
        for i in range(h):
            # prop along y for row i
            img_nbr = padded_img[i:i+prop_size,:]
            pos_nbr = padded_pos[i:i+prop_size,:]
            img_nbr = img_nbr - padded_img[i+half_size,:,:]
            pos_nbr = pos_nbr - padded_pos[i+half_size,:,:]
            img_nbr = np.sum(img_nbr**2, axis=2)/(2*sigma_1**2)
            pos_nbr = np.sum(pos_nbr**2, axis=2)/(2*sigma_2**2)
            k = np.exp(-img_nbr-pos_nbr)
            deg_mat[i+half_size,:] = deg_mat[i+half_size,:]+np.sum(k,axis=0)
            
            # prop along x for col i
            img_nbr = padded_img[:,i:i+prop_size]
            pos_nbr = padded_pos[:,i:i+prop_size]
            img_nbr = img_nbr - padded_img[:,i+half_size,:].reshape((padded_img.shape[0],1,c))
            pos_nbr = pos_nbr - padded_pos[:,i+half_size,:].reshape((padded_img.shape[0],1,2))
            img_nbr = np.sum(img_nbr**2, axis=2)/(2*sigma_1**2)
            pos_nbr = np.sum(pos_nbr**2, axis=2)/(2*sigma_2**2)
            k = np.exp(-img_nbr-pos_nbr)
            deg_mat[:,i+half_size] = deg_mat[:,i+half_size]+np.sum(k,axis=1)

        for i in range(h):
            # prop along y for row i
            img_nbr = padded_img[i:i+prop_size,:]
            pos_nbr = padded_pos[i:i+prop_size,:]
            img_nbr = img_nbr - padded_img[i+half_size,:,:]
            pos_nbr = pos_nbr - padded_pos[i+half_size,:,:]
            img_nbr = np.sum(img_nbr**2, axis=2)/(2*sigma_1**2)
            pos_nbr = np.sum(pos_nbr**2, axis=2)/(2*sigma_2**2)
            k = np.exp(-img_nbr-pos_nbr) # similarity matrix
            k = k/deg_mat[i+half_size,:] #devided by degree
            prop_prob = k * padded_prob_fix[i+half_size,:]
            padded_prob[i:i+prop_size,:] = padded_prob[i:i+prop_size,:] + prop_prob

            # prop along x for col i
            img_nbr = padded_img[:,i:i+prop_size]
            pos_nbr = padded_pos[:,i:i+prop_size]
            img_nbr = img_nbr - padded_img[:,i+half_size,:].reshape((padded_img.shape[0],1,c))
            pos_nbr = pos_nbr - padded_pos[:,i+half_size,:].reshape((padded_img.shape[0],1,2))
            img_nbr = np.sum(img_nbr**2, axis=2)/(2*sigma_1**2)
            pos_nbr = np.sum(pos_nbr**2, axis=2)/(2*sigma_2**2)
            k = np.exp(-img_nbr-pos_nbr) # similarity matrix
            k = k/deg_mat[:,i+half_size].reshape((-1,1)) #devided by degree
            
            prop_prob = k * padded_prob_fix[:,i+half_size].reshape((-1,1))
            padded_prob[:,i:i+prop_size] =  padded_prob[:,i:i+prop_size]+ prop_prob
        # padded_prob = padded_prob + 0.5 * padded_prob_fix # lazy propagation  
        prob_out = padded_prob[half_size:h+half_size,half_size:w+half_size]
        # print(np.sum(prob_out))
    prob_out[prob_out>1]=1
    return prob_out


# In[34]:


def prob_to_patch(im):
    """Convert pixel level probability prediction to patch version
    """
    patch_list = []
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            df = np.mean(patch)
            patch_list.append(df)
    return np.array(patch_list)

