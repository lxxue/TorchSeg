import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from sklearn import linear_model
from tqdm import tqdm
import cv2

patch_size = 16 # each patch is 16*16 pixels
train_dir = '../cil/training/images/'
gt_dir = '../cil/training/groundtruth/'
test_dir = '../cil/test_images/'

# Helper functions
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X


def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
    

def dataloader(img_dir, gt=False):
    # Loaded a set of images
    files = os.listdir(img_dir)
    n = len(files)
    print("Loading " + str(n) + " images")
    imgs = [load_image(img_dir + files[i]) for i in range(n)]

    # Extract patches from input images
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

    # Compute features for each image patch
    if gt:
        X = np.asarray([value_to_class(np.mean(img_patches[i])) for i in range(len(img_patches))])
    else:
        X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])

    return X



if __name__ == '__main__':
    X = dataloader(train_dir)
    print(X.shape)
    Y = dataloader(gt_dir, True)
    print(Y.shape)

    # Print feature statistics
    print('Computed ' + str(X.shape[0]) + ' features')
    print('Feature dimension = ' + str(X.shape[1]))
    print('Number of classes = ' + str(np.max(Y)+1))

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print('Class 0: ' + str(len(Y0)) + ' samples')
    print('Class 1: ' + str(len(Y1)) + ' samples')

    # train a logistic regression classifier
    # we create an instance of the classifier and fit the data
    logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
    logreg.fit(X, Y)

    # Predict on the test set
    T = dataloader(test_dir)
    print(T.shape)
    Z = logreg.predict(T)

    try:
        os.mkdir('pred_linear')
    except:
        pass
    test_files = os.listdir(test_dir)
    for img in tqdm(test_files):
        Xi = extract_img_features(test_dir + img)
        Zi = logreg.predict(Xi)

        # Display prediction as an image
        w = 608
        h = 608
        predicted_im = label_to_img(w, h, patch_size, patch_size, Zi)
        # plt.imsave("pred/"+img, predicted_im, cmap='Greys_r')
        heatmap = (predicted_im * 255).astype(np.uint8)
        cv2.imwrite("pred_linear/"+img, heatmap)


