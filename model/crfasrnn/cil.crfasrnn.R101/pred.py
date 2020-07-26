import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from tools.benchmark import compute_speed, stat
from datasets import CIL
from network import CrfRnnNet
from utils.img_utils import normalize

logger = get_logger()

# add pixel-wise RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

_CPU = torch.device("cpu")
_GPU = torch.device("cuda")

def img_to_black(img, threshold=50):
    """Binary filter on greyscale image."""
    img = img.astype(np.int64)
    idx = img[:,:] > threshold
    idx_0 = img[:,:] <= threshold
    img[idx] = 1
    img[idx_0] = 0
    return img

def img_to_uint8(img, threshold=0.50, patch_size = 16):
    img = img_to_black(img)
    """Reads a single image and outputs the strings that should go into the submission file"""
    for j in range(0, img.shape[1], patch_size):
        for i in range(0, img.shape[0], patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            if np.mean(patch) > threshold:
                img[i:i + patch_size, j:j + patch_size] = np.ones_like(patch)
            else:
                img[i:i + patch_size, j:j + patch_size] = np.zeros_like(patch)
    return img

class TestPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt):

        img = normalize(img, self.img_mean, self.img_std)

        img = img.transpose(2, 0, 1)

        extra_dict = None

        return img, gt, extra_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--input_size', type=str, default='1x3x400x400',
                        help='Input size. '
                             'channels x height x width (default: 1x3x224x224)')
    parser.add_argument('-speed', '--speed_test', action='store_true')
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('-summary', '--summary', action='store_true')
    parser.add_argument('-m', '--mode', default='test', type=str)
    parser.add_argument('-ob', '--output_binary', default=False, action='store_true')

    args = parser.parse_args()
    # dev = torch.device("cuda:0")

    network = CrfRnnNet(config.num_classes, n_iter=config.eval_num_iter)
    weights = torch.load(os.path.join("log", "snapshot", "epoch-{}.pth".format(args.epochs)))['model']
    network.load_state_dict(weights)
    # network.to(dev)
    network.to(_GPU)
    network.crfrnn.to(_CPU)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'test_source': config.test_source}
    dataset = CIL(data_setting, args.mode, preprocess=TestPre(config.image_mean, config.image_std))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    if args.save_path is not None:
        try:
            os.makedirs(args.save_path)
        except:
            pass

    with torch.no_grad():
        network.eval()
        for data in tqdm(dataloader):
            img = data['data']
            name = data['fn'][0]

            img = torch.from_numpy(np.ascontiguousarray(img)).float().to(_GPU)

            fmap = network(img)

            score = F.softmax(fmap, dim=1)

            # if network.crfrnn.num_iterations==0:
            #     score = F.softmax(fmap, dim=1)
            # else:
            #     score = fmap

            if args.save_path is not None:
                fn = name + '.png'
                if(args.output_binary):
                    heatmap = (np.argmax(score[0].cpu().numpy(), axis=0) * 255).astype(np.uint8)
                else:
                    heatmap = (score[0, 1].cpu().numpy() * 255).astype(np.uint8)
                # print(heatmap)
                cv2.imwrite(os.path.join(args.save_path, fn), heatmap)