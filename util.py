import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import cv2

def visual_explanation(heatmap, size):
    heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min()+0.000000001)
    heatmap = heatmap.detach().cpu().numpy()
    return cv2.resize(np.transpose(heatmap, (1, 2, 0)), size), heatmap, np.transpose(heatmap, (1, 2, 0)).squeeze()
    # return np.squeeze(np.transpose(heatmap, (1, 2, 0)))

def visual_explanation_hse(heatmap, size):
    heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min()+0.000000001)
    heatmap = heatmap.detach().cpu().numpy()
    # return cv2.resize(np.transpose(heatmap, (1, 2, 0)), size), heatmap, np.transpose(heatmap, (1, 2, 0)).squeeze()
    return np.squeeze(np.transpose(heatmap, (1, 2, 0)))