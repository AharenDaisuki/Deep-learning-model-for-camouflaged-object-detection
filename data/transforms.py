# Data augumentation strategies borrowed from https://github.com/XuelianCheng/SLT-Net/blob/master/dataloaders/video_list.py
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

def cv_random_flip(img1, img2, flow, mask):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        flow = flow.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img1, img2, flow, mask

def randomRotation(img1, img2, flow, mask):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img1 = img1.rotate(random_angle, mode)
        img2 = img2.rotate(random_angle, mode)
        flow = flow.rotate(random_angle, mode)
        mask = mask.rotate(random_angle, mode)
    return img1, img2, flow, mask

def colorEnhance(img1, img2):
    # brightness
    bright_intensity = random.randint(5, 15) / 10.0
    img1 = ImageEnhance.Brightness(img1).enhance(bright_intensity)
    img2 = ImageEnhance.Brightness(img2).enhance(bright_intensity)
    # contrast
    contrast_intensity = random.randint(5, 15) / 10.0
    img1 = ImageEnhance.Contrast(img1).enhance(contrast_intensity)
    img2 = ImageEnhance.Contrast(img2).enhance(contrast_intensity)
    # color intensity
    color_intensity = random.randint(0, 20) / 10.0
    img1 = ImageEnhance.Color(img1).enhance(color_intensity)
    img2 = ImageEnhance.Color(img2).enhance(color_intensity)
    # sharpness
    sharp_intensity = random.randint(0, 30) / 10.0
    img1 = ImageEnhance.Sharpness(img1).enhance(sharp_intensity)
    img2 = ImageEnhance.Sharpness(img2).enhance(sharp_intensity)
    return img1, img2

def randomPeper(img):
    img = np.array(img)
    # TODO: noise number
    noiseNum = int(0.0001 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255       
    return Image.fromarray(img)

def create_transform(newsize):
    # img transform
    img_transform = transforms.Compose(
        [
            transforms.Resize([newsize, newsize]),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    # flow transform
    flow_transform = transforms.Compose(
        [
            transforms.Resize([newsize, newsize]), # NOTE: {0, 255} => {0, ..., 255}
            transforms.ToTensor()
        ]
    )
    # mask transform
    mask_transform = transforms.Compose(
        [
            transforms.Resize([newsize, newsize]), # NOTE: {0, 255} => {0, ..., 255}
            transforms.ToTensor()
        ]
    )
    return img_transform, flow_transform, mask_transform

class BinaryMapping:
    '''
    data: mask (Mask will be messed up by downsampling. Map [0, 255] to [0, 1])
    '''
    def __call__(self, mask):
        bin_tensor = mask > 0.0
        
        return bin_tensor.long()