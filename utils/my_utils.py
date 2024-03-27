import cv2
import torch 
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from kornia.morphology import erosion

def split_batches(x: Tensor):
    """ Split a 2*B batch of images into two B images per batch,
    in order to adapt to MMSegmentation """

    assert x.ndim == 4, f'expect to have 4 dimensions, but got {x.ndim}'
    batch_size = x.shape[0] // 2
    x1 = x[0:batch_size, ...]
    x2 = x[batch_size:, ...]
    return x1, x2

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def mask2edge(mask):
    mask = mask.float()
    kernel = torch.ones((5, 5), device=mask.device)
    ero = erosion(mask, kernel)
    return mask - ero

# def pred_to_image(pred):
#     ''' pred->torch, ground truth->torch'''
#     pred = pred.sigmoid().data.cpu().numpy() 
#     pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
#     mean, var = pred.mean(), pred.var()
#     threshold = mean + 3 * np.sqrt(var)
#     pred = (pred > threshold).astype(np.uint8) * 255
#     print(pred.shape)
#     assert False
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     return cv2.erode(pred, kernel, iterations=3)

def remove_border(pred, top = 0, bottom = 0, left = 0, right = 0):
    pred[:top, :] = 0
    pred[bottom:, :] = 0
    pred[:, :left] = 0
    pred[:, right:] = 0
    return pred

def borders_capture(gt,pred,dksize=15):
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img=gt.copy()
    img[:]=0
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    kernel = np.ones((dksize, dksize), np.uint8)
    img_dilate = cv2.dilate(img, kernel)

    res = cv2.bitwise_and(img_dilate, gt)
    b, g, r = cv2.split(res)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    merge = cv2.merge((b, g, r, alpha))

    resp = cv2.bitwise_and(img_dilate, pred)
    b, g, r = cv2.split(resp)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    mergep = cv2.merge((b, g, r, alpha))

    merge = cv2.cvtColor(merge, cv2.COLOR_RGB2GRAY)
    mergep = cv2.cvtColor(mergep, cv2.COLOR_RGB2GRAY)
    return merge,mergep,np.sum(img_dilate)/255