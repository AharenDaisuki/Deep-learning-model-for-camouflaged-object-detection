import os 
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

from thop import profile

from model.model import Net_V3, Net_V3_S
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

def infer_model(args, model):
    # dataset
    train_dataset, val_dataset, train_loader, val_loader = create_dataset(
        data_path  = args.dataset,
        input_size = args.inputsize, 
        batch_size = args.batchsize, 
        test       = True
    )

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

def infer_time(args, model):
    # load dataset
    train_dataset, val_dataset, train_loader, val_loader = create_dataset(
        data_path  = args.dataset,
        input_size = args.inputsize, 
        batch_size = args.batchsize, 
        test       = True
    )
    # load model
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['checkpoint'])
    # flops & params
    dummy_input = torch.randn(args.batchsize, 3, 352, 352).to(0)
    flops, params = profile(model, inputs=(dummy_input, dummy_input, ))
    print('flops: %.2f G, MACs: %.2f G, params: %.2f M' % (flops / (args.batchsize * 1e9), 2 * flops / (args.batchsize * 1e9), params / 1e6))
    # warm-up for 200 iterations
    iter_n = 1000
    print('warm-up ...')
    with tqdm(total=iter_n) as pbar:
        for i in range(iter_n):
            random_image = torch.randn(args.batchsize, 3, 352, 352).to(0)
            random_flow  = torch.randn(args.batchsize, 3, 352, 352).to(0) 
            _, _ = model(random_image, random_flow)
            pbar.update()

    step_n = len(val_loader)
    times  = torch.zeros(step_n)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with tqdm(total=step_n) as pbar:
        for i in range(step_n):
            random_image = torch.randn(args.batchsize, 3, 352, 352).to(0)
            random_flow  = torch.randn(args.batchsize, 3, 352, 352).to(0) 
            
            starter.record()
            _, _ = model(random_image, random_flow)
            ender.record()
            torch.cuda.synchronize() # synchronize GPU time
            curr_time = starter.elapsed_time(ender) # 计算时间
            times[i] = curr_time
            pbar.update()

    avg_time = times.mean().item() # in ms
    print("Inference time: {:.6f} ms, FPS: {} ".format((avg_time / args.batchsize), int(1000 * args.batchsize / avg_time)))
    log = f'cons-edge\nflops: {flops / (args.batchsize * 1e9)} G, MACs: {2 * flops / (args.batchsize * 1e9)} G, params: {params / 1e6} M\nInference time: {avg_time / args.batchsize} ms, FPS: {int(1000 * args.batchsize / avg_time)}\n'
    f = open('inference_speed.txt', 'a')
    f.write(log)
    f.write("\n")
    f.close() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='Net_V3', help='model variant')
    parser.add_argument('--checkpoint', type=str, help='.pth path')
    parser.add_argument('--dataset', type=str, help='dataset path')
    parser.add_argument('--savepath', type=str, help='prediction path')
    parser.add_argument('--batchsize', type=int, default=2, help='batch size')
    parser.add_argument('--inputsize', type=int, default=352, help='input size')
    parser.add_argument('--outputsize', type=tuple, default=(720, 1280), help='output size')
    # parser.add_argument('--threshold', type=float, default=0.9, help='inference threshold')
    args = parser.parse_args()
    model = Net_V3().cuda() # NOTE: change different models here
    infer_model(args, model)
    # infer_time(args, model)