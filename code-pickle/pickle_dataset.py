import os.path as osp
import math
import json
from PIL import Image
import pickle
import random
import os

import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon


class PickleDataset(Dataset):
    def __init__(self, datadir, to_tensor=True):
        self.datadir = datadir
        self.to_tensor = to_tensor
        self.datalist = [f for f in os.listdir(datadir) if f.endswith('.pkl')]
 
    def __getitem__(self, idx):
        with open(file=osp.join(self.datadir, f"1024_1024_{idx}.pkl"), mode="rb") as f:
            data = pickle.load(f)
            
        image, score_map, geo_map, roi_mask = data
        if self.to_tensor:
            image = torch.Tensor(image)
            score_map = torch.Tensor(score_map)
            geo_map = torch.Tensor(geo_map)
            roi_mask = torch.Tensor(roi_mask)

        return image, score_map, geo_map, roi_mask

    def __len__(self):
        return len(self.datalist)