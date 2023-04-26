# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Peiwen Lin, Xiangtai Li
# Email: linpeiwen@sensetime.com, lixiangtai@sensetime.com
import os
from PIL import Image
import time
import logging
from argparse import ArgumentParser
import numpy as np
import yaml
import sys
import math
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

#from segment_anything 
#import SamAutomaticMaskGenerator, sam_model_registry, predictor_registry

from segment_anything_1 import sam_model_registry
#from sam_folder import predictor_registry


import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from torchvision import transforms

# Setup Parser
def get_parser():
    parser = ArgumentParser(description='PyTorch Evaluation')
    parser.add_argument(
        '--base_size', type=int,
        default=2048, help='based size for scaling')
    parser.add_argument(
        '--scales', type=float,
        default=[1.0], nargs='+', help='evaluation scales')
    parser.add_argument(
        "--config", type=str, default="config.yaml")
    parser.add_argument(
        '--model_path', type=str,
        default='checkpoints/psp_best.pth', help='evaluation model path')
    parser.add_argument(
        '--save_folder', type=str,
        default='checkpoints/results/', help='results save folder')
    parser.add_argument(
        '--names_path', type=str,
        default='../../vis_meta/cityscapes/cityscapesnames.mat',
        help='path of dataset category names')
    parser.add_argument(
        '--crop', action="store_true", default=False, help="whether use sliding-window evaluation"
    )
    parser.add_argument(
        '--flip', type=bool,
        default=False, help='whether flip')
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_list(h):
    if h % 1024 == 0:
        h_list = np.arange(0, h, 1024)
        return h_list
    elif h % 1024 >= 768:
        h_list = np.arange(0, h, 1024)
        return h_list
    elif h % 1024 <= 256:
        h_list = np.arange(0, h, 1024)[: -1]
        return h_list
    else:
        num = int(h / 1024) + 1
        interval = math.ceil(h / num)
        h_list = [int(i*interval) for i in range(num)]
        return h_list


def main():

    sam = sam_model_registry['vit_h']('/public2/home/lanyun/pretrain/sam_vit_h_4b8939.pth')
    
    sam = sam.cuda()
    sam.eval()
    data_list = []
    
    #d_list = os.path.join(cfg['val']['data_root'], cfg['val']['data_list'])
    trans = transforms.Compose([
                transforms.ToTensor()])
    epoch_minus = []
    
    
    inputs = Image.open("./test_image/6.jpg").convert('RGB')
    inputs = np.array(inputs)
    inputs = torch.Tensor(inputs).float()
    inputs = inputs.permute(2, 0, 1)
    #inputs = trans(inputs)
    inputs = inputs.cuda()
    inputs = inputs.unsqueeze(0)
    b, c, h, w = inputs.shape
    inputs = F.interpolate(inputs, size=(1024, 1024), mode='bilinear', align_corners=True)
    inputs = inputs.squeeze(0)
    with torch.set_grad_enabled(False):
        pre_count = 0.0
        batch = {"image": inputs,
                    "points": None,
                    "targets": None,
                    "st_sizes": None,
                    "gd_count": None,
                    "original_size": (h, w),
                    }
        outputs = sam([batch], multimask_output=False)
        masks = outputs[0]['masks']
        masks = torch.sigmoid(masks)
        masks = masks.squeeze(0).squeeze(0).cpu().numpy()
        print(torch.max(masks))
        out = np.ones(masks.shape) * (masks > 0.5)
        out = np.uint8(out * 255)
        print(masks.shape)
        #print(torch.max(masks))
        out = cv2.imwrite('./3.jpg', out)
        
        

if __name__ == '__main__':
    main()
    
    
    