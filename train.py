import math
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model.model import Net_V3, Net_V3_S, Net_V1, Net_V2
from data.dataset import create_dataset
from utils.my_utils import mask2edge, resize
from loss import hybrid_loss

def train(args, model):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    optimizer    = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    device       = torch.device(0)
    train_losses = []
    eval_maes    = []

    if args.resume:
        checkpoint   = torch.load(args.resume)
        train_losses = checkpoint['train_losses']
        eval_maes    = checkpoint['eval_maes']
        model.load_state_dict(checkpoint['checkpoint'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    train_dataset, val_dataset, train_loader, val_loader = create_dataset(
        data_path  = args.dataset,
        input_size = args.inputsize, 
        batch_size = args.batchsize
    )

    start_epoch = len(train_losses) + 1
    max_epoch   = args.epoch
    best_mae    = 1e3
    for epoch in range(start_epoch, max_epoch + 1):
        # train
        train_n = len(train_loader)
        tmp_losses = []
        with tqdm(total=train_n) as pbar:
            for idx, (image1, image2, flow, mask, meta_info) in enumerate(train_loader):
                image1 = image1.to(device) # [B, C, H, W]
                image2 = image2.to(device) # [B, C, H, W]
                flow = flow.to(device) # [B, C, H, W]
                mask = mask.to(device) # [B, 1, H, W]
                edge = mask2edge(mask) # [B, 1, H, W] 
                # mask = torch.squeeze(mask, 1) # [B, H, W]
                # clean
                optimizer.zero_grad()
                # forward
                if args.loss == 'cons+edge' or args.loss == 'edge':
                    pred1, edges1 = model(image1, flow)
                    pred2, edges2 = model(image2, flow)
                else:
                    # TODO: no edge
                    pred1 = model(image1, flow)
                    pred2 = model(image2, flow)
                
                pred1 = resize(pred1, size=mask.shape[2:], mode='bilinear', align_corners=False) # [B, 2, H, W]
                pred2 = resize(pred2, size=mask.shape[2:], mode='bilinear', align_corners=False)
                loss1 = hybrid_loss(pred1[:,:1,:,:], mask.float())
                loss2 = hybrid_loss(pred2[:,:1,:,:], mask.float())

                if args.loss == 'cons+edge' or args.loss == 'edge':
                    loss_edge1, loss_edge2 = 0.0, 0.0 
                    for i in edges1:
                        i = resize(i, size=edge.shape[2:], mode='bilinear', align_corners=False)
                        loss_edge1 += hybrid_loss(i, edge)
                    for i in edges2:
                        i = resize(i, size=edge.shape[2:], mode='bilinear', align_corners=False) 
                        loss_edge2 += hybrid_loss(i, edge)
                
                if args.loss == 'cons+edge':
                    loss = 0.495 * loss1 + 0.495 * loss2 + 0.005 * loss_edge1 + 0.005 * loss_edge2
                elif args.loss == 'cons':
                    loss = 0.5 * loss1 + 0.5 * loss2
                elif args.loss == 'edge':
                    loss = 0.99 * loss1 + 0.01 * loss_edge1
                else:
                    loss = loss1

                # backpropagate
                loss.backward()
                optimizer.step()
                # statistics
                assert not math.isnan(loss.item())
                tmp_losses.append(loss.item())
                pbar.set_description(f'epoch ({epoch}/{max_epoch})')
                if args.loss == 'cons+edge' or args.loss == 'edge':
                    pbar.set_postfix({
                        'train loss': '{:.5f}'.format(loss.item()),
                        'mask loss1': '{:.5f}'.format(loss1.item()), 
                        'mask loss2': '{:.5f}'.format(loss2.item()), 
                        'edge loss1': '{:.5f}'.format(loss_edge1.item()), 
                        'edge loss2': '{:.5f}'.format(loss_edge2.item()),
                    })
                else:
                    pbar.set_postfix({
                        'train loss': '{:.5f}'.format(loss.item()),
                        'mask loss1': '{:.5f}'.format(loss1.item()), 
                        'mask loss2': '{:.5f}'.format(loss2.item())
                    })
                pbar.update()
        # set 2
        step_n = len(val_loader)
        with tqdm(total=step_n) as pbar:
            for idx, (image1, image2, flow, mask, meta_info) in enumerate(val_loader):
                image1 = image1.to(device) # [B, C, H, W]
                image2 = image2.to(device) # [B, C, H, W]
                flow = flow.to(device) # [B, C, H, W]
                mask = mask.to(device) # [B, 1, H, W]
                edge = mask2edge(mask) # [b, 1, H, W] 
                # mask = torch.squeeze(mask, 1) # [B, H, W]
                # clean
                optimizer.zero_grad()
                # forward
                if args.loss == 'cons+edge' or args.loss == 'edge':
                    pred1, edges1 = model(image1, flow)
                    pred2, edges2 = model(image2, flow)
                else:
                    # TODO: no edge
                    pred1 = model(image1, flow)
                    pred2 = model(image2, flow)
                
                pred1 = resize(pred1, size=mask.shape[2:], mode='bilinear', align_corners=False)
                pred2 = resize(pred2, size=mask.shape[2:], mode='bilinear', align_corners=False)
                loss1 = hybrid_loss(pred1[:,:1,:,:], mask.float())
                loss2 = hybrid_loss(pred2[:,:1,:,:], mask.float())

                if args.loss == 'cons+edge' or args.loss == 'edge':
                    loss_edge1, loss_edge2 = 0.0, 0.0 
                    for i in edges1:
                        i = resize(i, size=edge.shape[2:], mode='bilinear', align_corners=False)
                        loss_edge1 += hybrid_loss(i, edge)
                    for i in edges2:
                        i = resize(i, size=edge.shape[2:], mode='bilinear', align_corners=False) 
                        loss_edge2 += hybrid_loss(i, edge)
                
                if args.loss == 'cons+edge':
                    loss = 0.495 * loss1 + 0.495 * loss2 + 0.005 * loss_edge1 + 0.005 * loss_edge2
                elif args.loss == 'cons':
                    loss = 0.5 * loss1 + 0.5 * loss2
                elif args.loss == 'edge':
                    loss = 0.99 * loss1 + 0.01 * loss_edge1
                else:
                    loss = loss1
                # backpropagate
                loss.backward()
                optimizer.step()
                # statistics
                assert not math.isnan(loss.item())
                tmp_losses.append(loss.item())
                pbar.set_description(f'epoch ({epoch}/{max_epoch})')
                if args.loss == 'cons+edge' or args.loss == 'edge':
                    pbar.set_postfix({
                        'train loss': '{:.5f}'.format(loss.item()),
                        'mask loss1': '{:.5f}'.format(loss1.item()), 
                        'mask loss2': '{:.5f}'.format(loss2.item()), 
                        'edge loss1': '{:.5f}'.format(loss_edge1.item()), 
                        'edge loss2': '{:.5f}'.format(loss_edge2.item()),
                    })
                else:
                    pbar.set_postfix({
                        'train loss': '{:.5f}'.format(loss.item()),
                        'mask loss1': '{:.5f}'.format(loss1.item()), 
                        'mask loss2': '{:.5f}'.format(loss2.item())
                    })
                pbar.update()

        train_loss = np.mean(tmp_losses)
        train_losses.append(train_loss)
        scheduler.step()

        # eval
        eval_n = len(val_loader)
        tmp_maes = []
        with torch.no_grad():
            model.eval()
            with tqdm(total=eval_n) as pbar:
                for idx, (image1, image2, flow, mask, meta_info) in enumerate(val_loader):
                    image1 = image1.to(device) # [B, C, H, W]
                    image2 = image2.to(device) # [B, C, H, W]
                    flow = flow.to(device) # [B, C, H, W]
                    mask = mask.to(device) # [B, 1, H, W]
                    edge = mask2edge(mask) # [B, 1, H, W] 
                    # mask = torch.squeeze(mask, 1) # [B, H, W]
                    # edge = torch.squeeze(edge, 1) # [B, H, W]
                    # clean
                    # optimizer.zero_grad()
                    # forward
                    mask_arr = np.asarray(mask.cpu(), np.float32)
                    mask_arr /= (mask_arr.max() + 1e-8)
                    if args.loss == 'cons+edge' or args.loss == 'edge':
                        pred1, edges1 = model(image1, flow)
                    else:
                        # TODO: no edge
                        pred1 = model(image1, flow)
                    # pred1, edges1 = model(image1, flow)
                    # pred2, edges2 = model(image2, flow)

                    pred1 = resize(pred1, size=mask.shape[2:], mode='bilinear', align_corners=False)
                    batch_size = pred1.shape[0]
                    batch_mae = 0.0

                    pred1_arr = pred1[:,:1,:,:].sigmoid().data.cpu().numpy()
                    pred1_arr = (pred1_arr - pred1_arr.min()) / (pred1_arr.max() - pred1_arr.min() + 1e-8)
                    batch_mae += np.sum(np.abs(pred1_arr - mask_arr)) * 1.0 / (mask.shape[2] * mask.shape[3])
                    batch_mae /= batch_size
                    
                    tmp_maes.append(batch_mae)
                    pbar.set_description(f'epoch ({epoch}/{max_epoch})')
                    pbar.set_postfix({
                        'MAE': '{:.5f}'.format(batch_mae),
                    })
                    pbar.update()
        eval_mae = np.mean(tmp_maes)
        eval_maes.append(eval_mae)
        if eval_mae < best_mae:
            best_mae = eval_mae
            model_save_path = f'{args.name}({epoch}_{max_epoch})loss={train_loss}_mae={eval_mae}.pth'
            print(f'save model {model_save_path} ...')
            torch.save({
                'checkpoint': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'eval_maes': eval_maes,
            }, model_save_path)
    return train_losses, eval_maes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='demo', help='model name')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--epoch', type=int, default=100, help='ending epoch')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path')
    parser.add_argument('--batchsize', type=int, default=2, help='batch size')
    parser.add_argument('--inputsize', type=int, default=352, help='input size')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weightdecay', type=float, default=1e-2, help='AdamW weight decay')
    parser.add_argument('--stepsize', type=int, default=25, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='scheduler gamma')

    parser.add_argument('--dataset', type=str, help='dataset path')
    parser.add_argument('--loss', type=str, default='cons+edge', help='loss type')
    args = parser.parse_args()
    model = Net_V2().cuda() # TODO: change model [Net_V3, Net_V3_S, Net_V1, Net_V2]
    train(args, model)