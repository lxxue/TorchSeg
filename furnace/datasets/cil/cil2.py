import os
import time
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

import torch.utils.data as data

class CIL2(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None):
        super(CIL2, self).__init__()
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._test_source = setting['test_source']
        self._file_names = self._get_file_names(split_name)
        # self._file_length = file_length
        self._file_length = len(self._file_names)
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        img_path = os.path.join(self._img_path, names[0])
        gt_path = os.path.join(self._gt_path, names[1])
        item_name = names[1].split("/")[-1].split(".")[0]

        img, gt = self._fetch_data(img_path, gt_path)
        img = img[:, :, ::-1]

        if self._split_name == 'train' or self._split_name == 'val':
            edge_path = os.path.join(self._img_path, names[2])
            midline_path = os.path.join(self._img_path, names[3])
            edge = np.array(cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE), dtype=None)
            midline = np.array(cv2.imread(midline_path, cv2.IMREAD_GRAYSCALE), dtype=None)
        else:
            edge = None
            midline = None
        if self.preprocess is not None:
            img, gt, edge, midline, extra_dict = self.preprocess(img, gt, edge, midline)

        if self._split_name == 'train' or self._split_name == 'val':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            edge = torch.from_numpy(np.ascontiguousarray(edge)).long()
            midline = torch.from_numpy(np.ascontiguousarray(midline)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

            output_dict = dict(data=img, label=gt, fn=str(item_name),
                               n=len(self._file_names), edge=edge, midline=midline)
        else:
            output_dict = dict(data=img, label=gt, fn=str(item_name),
                               n=len(self._file_names))

        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)

        return img, gt

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        if split_name == 'train':
            source = self._train_source
        elif split_name == "val":
            source = self._eval_source
        else:
            source = self._test_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            img_name, gt_name, edge_name, midline_name = self._process_item_names(item)
            file_names.append([img_name, gt_name, edge_name, midline_name])

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def _process_item_names(self, item):
        item = item.strip()
        item = item.split('\t')
        img_name = item[0]
        gt_name = item[1]
        if self._split_name == 'train' or self._split_name == 'val':
            edge_name = item[2]
            midline_name = item[3]

            return img_name, gt_name, edge_name, midline_name
        else:
            return img_name, gt_name, None, None

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)

        return img

    @classmethod
    def get_class_colors(*args):
        """color for visualization and saving images."""
        return [[255, 255, 255], [0, 0, 0]]

    @classmethod
    def get_class_names(*args):
        """Label names."""
        return ['road', 'non-road']