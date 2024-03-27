import os 
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

from model.model import Net_V3
from data.dataset import create_dataset
from utils.my_utils import resize, remove_border

def pred_to_image(pred):
    ''' pred->torch, ground truth->torch'''
    pred = pred.sigmoid().data.cpu().numpy() 
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mean, var = pred.mean(), pred.var()
    threshold = mean + 2 * np.sqrt(var)
    pred = (pred > threshold).astype(np.uint8) * 255
    pred = remove_border(pred, 120, 680, 40, 1240)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.erode(pred, kernel, iterations=3)

def infer_model(args):
    # dataset
    train_dataset, val_dataset, train_loader, val_loader = create_dataset(
        data_path  = args.dataset,
        input_size = args.inputsize, 
        batch_size = args.batchsize, 
        test       = True
    )
    # model
    model = Net_V3().cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['checkpoint'])

    step_n = len(val_loader)
    with torch.no_grad():
        with tqdm(total=step_n) as pbar:
            for idx, (image1, image2, flow, mask, meta_info) in enumerate(val_loader):
                image1 = image1.to(0) # [B, C, H, W]
                image2 = image2.to(0) # [B, C, H, W]
                flow = flow.to(0) # [B, C, H, W]
                mask = mask.to(0) # [B, 1, H, W]
                # edge = mask2edge(mask) # [b, 1, H, W] 
                mask = torch.squeeze(mask, 1) # [B, H, W]
                
                preds1, edges1 = model(image1, flow)
                preds1 = resize(preds1, size=args.outputsize, mode='bilinear', align_corners=False)
                N = preds1.shape[0]
                for i in range(N): 
                    pred = preds1[i][1]
                    filename = '{}_{:0>5d}.png'.format(meta_info['category'][i], meta_info['index'][i])
                    # img = pred_to_image(pred, threshold=args.threshold)
                    img = pred_to_image(pred)
                    cv2.imwrite(os.path.join(args.savepath, filename), np.round(img))
                
                pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='.pth path')
    parser.add_argument('--dataset', type=str, help='dataset path')
    parser.add_argument('--savepath', type=str, help='prediction path')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--inputsize', type=int, default=352, help='input size')
    parser.add_argument('--outputsize', type=tuple, default=(720, 1280), help='output size')
    # parser.add_argument('--threshold', type=float, default=0.9, help='inference threshold')
    args = parser.parse_args()
    infer_model(args)