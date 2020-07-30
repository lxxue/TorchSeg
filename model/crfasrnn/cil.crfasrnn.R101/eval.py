import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import json

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

def img_to_black(img, threshold=50):
    """Binary filter on greyscale image."""
    img = img.astype(np.int64)
    idx = img[:,:] > threshold
    idx_0 = img[:,:] <= threshold
    img[idx] = 1
    img[idx_0] = 0
    return img

def img_to_uint8(img, threshold=0.50, patch_size = 16):
    """Reads a single image and outputs the strings that should go into the submission file"""
    for j in range(0, img.shape[1], patch_size):
        for i in range(0, img.shape[0], patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            if np.mean(patch) > threshold:
                img[i:i + patch_size, j:j + patch_size] = np.ones_like(patch)
            else:
                img[i:i + patch_size, j:j + patch_size] = np.zeros_like(patch)
    return img

class EvalPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt):
        gt = img_to_black(gt)

        img = normalize(img, self.img_mean, self.img_std)

        img = img.transpose(2, 0, 1)

        extra_dict = None

        return img, gt, extra_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default='', type=str)
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-n', '--num_iter', default=-1, type=int)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--input_size', type=str, default='1x3x400x400',
                        help='Input size. '
                             'channels x height x width (default: 1x3x224x224)')

    args = parser.parse_args()
    dev = torch.device("cuda:0")

    network = CrfRnnNet(config.num_classes, criterion=None, n_iter=args.num_iter)
    weights = torch.load(os.path.join("log", "snapshot_{}".format(args.learning_rate), "epoch-{}.pth".format(args.epochs)))['model']
    network.load_state_dict(weights)
    network.to(dev)
    network.crfrnn.to(torch.device('cpu'))
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'test_source': config.test_source}
    dataset = CIL(data_setting, 'val', preprocess=EvalPre(config.image_mean, config.image_std))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = np.zeros((len(dataloader),))

    pred_patch1 = []
    gt_patch1 = []
    pred_patch16 = []
    gt_patch16 = []


    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    with torch.no_grad():
        network.eval()
        for i, data in enumerate(tqdm(dataloader)):
            img = data['data']
            gt = data['label']
            name = data['fn'][0]

            img = torch.from_numpy(np.ascontiguousarray(img)).float().to(dev)
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long().to(dev)

            fmap = network(img)

            score = F.softmax(fmap, dim=1)
            # print(fmap)

            loss[i] = criterion(fmap, gt).item()
            # print(loss.item())
            heatmap = (score[0, 1].cpu().numpy() * 255).astype(np.uint8)

            if args.save_path is not None:
                fn = name + '.png'
                # print(heatmap)
                cv2.imwrite(os.path.join(args.save_path, fn), heatmap)

            pred_patch1.append(img_to_uint8(heatmap, patch_size=1).reshape((-1,)))
            gt_patch1.append(img_to_uint8(gt[0].cpu().numpy()*255, patch_size=1).reshape((-1,)))
            pred_patch16.append(img_to_uint8(heatmap, patch_size=16).reshape((-1,)))
            gt_patch16.append(img_to_uint8(gt[0].cpu().numpy()*255, patch_size=16).reshape((-1,)))

    pred_patch1 = np.stack(pred_patch1, axis=0)
    gt_patch1 = np.stack(gt_patch1, axis=0)
    pred_patch16 = np.stack(pred_patch16, axis=0)
    gt_patch16 = np.stack(gt_patch16, axis=0)
    # print(pred_patch16)
    # print(gt_patch16)

    stats = {}
    stats['mean_loss'] = np.mean(loss)
    stats['acc_1'] = np.mean(pred_patch1 == gt_patch1)
    stats['f1_micro_1'] = metrics.f1_score(gt_patch1, pred_patch1, average='micro')
    stats['f1_macro_1'] = metrics.f1_score(gt_patch1, pred_patch1, average='macro')
    stats['f1_samples_1'] = metrics.f1_score(gt_patch1, pred_patch1, average='samples')
    stats['acc_16'] = np.mean(pred_patch16 == gt_patch16)
    stats['f1_micro_16'] = metrics.f1_score(gt_patch16, pred_patch16, average='micro')
    stats['f1_macro_16'] = metrics.f1_score(gt_patch16, pred_patch16, average='macro')
    stats['f1_samples_16'] = metrics.f1_score(gt_patch16, pred_patch16, average='samples')

    print(json.dumps(stats, indent=4))
    if args.save_path is not None:
        with open(os.path.join(args.save_path, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)

    #print(np.mean(loss))
    # print(metrics.f1_score(gt_patch1, pred_patch1, average=None).shape)
    # print(metrics.f1_score(gt_patch1, pred_patch1, average='binary'))
    #print(np.mean(pred_patch1 == gt_patch1))
    #print(metrics.f1_score(gt_patch1, pred_patch1, average='micro'))
    #print(metrics.f1_score(gt_patch1, pred_patch1, average='macro'))
    #print(metrics.f1_score(gt_patch1, pred_patch1, average='samples'))
    # print(metrics.f1_score(gt_patch16, pred_patch16, average=None))
    # print(metrics.f1_score(gt_patch16, pred_patch16, average='binary'))
    #print(np.mean(pred_patch16 == gt_patch16))
    #print(metrics.f1_score(gt_patch16, pred_patch16, average='micro'))
    #print(metrics.f1_score(gt_patch16, pred_patch16, average='macro'))
    #print(metrics.f1_score(gt_patch16, pred_patch16, average='samples'))
