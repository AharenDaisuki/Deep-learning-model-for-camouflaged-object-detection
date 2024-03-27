import argparse
from metrics.vcod_metrics import evaluation
from data.dataset import create_dataset

def eval(args):
    # dataset for evaluation
    train_dataset, val_dataset, train_loader, val_loader = create_dataset(
        data_path  = args.dataset,
        input_size = args.inputsize, 
        batch_size = args.batchsize, 
        test       = True
    )
    # evaluation
    evaluation(
        gt_root    = args.gt, 
        pred_root  = args.pred, 
        dataloader = val_loader, 
        save_txt   = args.savetxt, 
        save_png  = args.savepng
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--inputsize', type=int, default=352, help='input size')
    parser.add_argument('--dataset', type=str, help='dataset path')
    parser.add_argument('--pred', type=str, default='prediction path')
    parser.add_argument('--gt', type=str, default='ground truth path')
    parser.add_argument('--savetxt', type=str, default='save txt path')
    parser.add_argument('--savepng', type=str, default='save png path')
    args = parser.parse_args()
    eval(args)