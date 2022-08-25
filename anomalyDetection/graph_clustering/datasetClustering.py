import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import cv2
import random

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    #def __init__(self, args, is_normal=True, transform=None, test_mode=False, only_anomaly=False):
    def __init__(self, T, STRIDE, training_folder, test = False):
            pass