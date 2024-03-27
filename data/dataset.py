import os
import torch
import pandas as pd

from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from data.transforms import BinaryMapping
from data.transforms import colorEnhance, cv_random_flip, randomRotation, randomPeper, create_transform

class VCODDataset(Dataset):
    def __init__(self, img_data, img_transform=None, flow_transform=None, mask_transform=None, test=False):
        # transformation
        self.img_transform = img_transform
        self.flow_transform = flow_transform
        self.mask_transform = mask_transform
        self.bin_mapping = BinaryMapping()
        self.img_data = img_data
        self.test = test
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        rgb_path1 = self.img_data.loc[index, 'frames_left']
        rgb_path2 = self.img_data.loc[index, 'frames_right']
        flow_path = self.img_data.loc[index, 'flows']
        mask_path = self.img_data.loc[index, 'masks']
        category = self.img_data.loc[index, 'categories']
        index = self.img_data.loc[index, 'indices']
        
        assert os.path.exists(rgb_path1)
        assert os.path.exists(rgb_path2)
        assert os.path.exists(flow_path)
        assert os.path.exists(mask_path)
        
        rgb1 = self.rgb_loader(rgb_path1)
        rgb2 = self.rgb_loader(rgb_path2)
        flow = self.rgb_loader(flow_path)
        mask = self.binary_loader(mask_path)
        rgb1, rgb2 = colorEnhance(rgb1, rgb2)
        # TODO: if testing => no transformations 
        if not self.test:
            rgb1, rgb2, flow, mask = cv_random_flip(rgb1, rgb2, flow, mask)
            rgb1, rgb2, flow, mask = randomRotation(rgb1, rgb2, flow, mask)
            mask = randomPeper(mask)

        # resize [1280, 720]
        if self.img_transform is not None:
            rgb1 = self.img_transform(rgb1)            
            rgb2 = self.img_transform(rgb2)
        if self.flow_transform is not None: 
            flow = self.flow_transform(flow)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        # assert type(rgb1) == torch.Tensor
        # assert rgb1.shape == torch.Size([3, INPUT_SIZE, INPUT_SIZE])
        # assert mask.shape == torch.Size([1, INPUT_SIZE, INPUT_SIZE])
        # binary mapping
        mask = self.bin_mapping(mask)
        # assert mask.dtype == torch.long
        # assert isinstance(category, str)
        # assert isinstance(index, np.int64)
        img_meta_info = {'category': category, 'index': index}
        
        return rgb1, rgb2, flow, mask, img_meta_info
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

def create_dataset(data_path, input_size=352, batch_size=8, test=False):
    root = data_path
    splits = ['train', 'val']
    types = ['flow', 'frame', 'mask']
    train_frames_left = []    
    train_frames_right = []
    train_flows = []
    train_masks = []
    train_categories = []
    train_indices = []
    val_frames_left = []    
    val_frames_right = []
    val_flows = []
    val_masks = []
    val_categories = []
    val_indices = []
    
    
    for split in splits:
        for t in types:
            for data in sorted(glob(os.path.join(root, t, split, '*.jpg'))):
                filename = os.path.basename(data)
                category = filename[:-10]
                index = filename[-9:-4]
                next_frame = os.path.join(root, t, split, '{}_{:0>5d}.jpg'.format(category, int(index)+1))
                if os.path.exists(next_frame):
                    if t == 'flow':
                        # TODO: optical flow
                        if split == 'train':
                            train_flows.append(data)
                            train_categories.append(category)
                            train_indices.append(int(index))
                        elif split == 'val':
                            val_flows.append(data)
                            val_categories.append(category)
                            val_indices.append(int(index))
                    elif t == 'frame':
                        if split == 'train':
                            train_frames_left.append(data)
                            train_frames_right.append(next_frame)
                        elif split == 'val':
                            val_frames_left.append(data)
                            val_frames_right.append(next_frame)
                    else:
                        assert False
                    
            for data in sorted(glob(os.path.join(root, t, split, '*.png'))):
                filename = os.path.basename(data)
                category = filename[:-10]
                index = filename[-9:-4]
                next_frame = os.path.join(root, 'frame', split, '{}_{:0>5d}.jpg'.format(category, int(index)+1))
                if os.path.exists(next_frame):
                    if t == 'mask':
                        if split == 'train':
                            train_masks.append(data)
                        elif split == 'val':
                            val_masks.append(data)
                    else:
                        assert False
                        
    train_data = {
        'frames_left': train_frames_left, 
        'frames_right':train_frames_right, 
        'flows': train_flows, 
        'masks': train_masks,
        'categories': train_categories,
        'indices': train_indices
    }
    val_data = {
        'frames_left':val_frames_left, 
        'frames_right':val_frames_right, 
        'flows': val_flows, 
        'masks': val_masks,
        'categories': val_categories,
        'indices': val_indices
    }
    train_data = pd.DataFrame(train_data)
    val_data = pd.DataFrame(val_data)
    img_transform, flow_transform, mask_transform = create_transform(input_size) # [352, 352]
    
    print(train_data.head())
    print(val_data.head())
    print(f'# of training samples: {len(train_data)}')
    print(f'# of validating samples: {len(val_data)}')
    
    train_dataset = VCODDataset(train_data, img_transform=img_transform, flow_transform=flow_transform, mask_transform=mask_transform, test=test)
    val_dataset = VCODDataset(val_data, img_transform=img_transform, flow_transform=flow_transform, mask_transform=mask_transform, test=test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_dataset, val_dataset, train_loader, val_loader