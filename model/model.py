import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mit import mit_b1, mit_b2
from mmseg.models.utils import SelfAttentionBlock
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.cnn.bricks import ConvModule

import sys
sys.path.append("..")
from utils.my_utils import resize, split_batches
from utils.vis_utils import vis_feature_map


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Backbone, self).__init__()
        # TODO: pretrained path
        self.backbone = mit_b2(pretrained=pretrained)

    def forward(self, x):
        x = self.backbone(x)
        return x
    
class Concatenate(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, img, flow):
        return img + flow

class AttentionFusion(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AttentionFusion, self).__init__()
        inter_channels = int(channels // r)
        
        # local attention
        self.local_attention = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # global attention
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, img, flow):
        x = img + flow
        x_channel = self.local_attention(x)
        x_pixel = self.global_attention(x)
        w = x_channel + x_pixel
        weight = self.sigmoid(w)

        x_output = img * weight + flow * (1 - weight)
        return x_output
    
    # def demo(self):
    #     img = torch.randn((8, 64, 720, 720))
    #     flow = torch.randn((8, 64, 720, 720))  
    #     output = self.forward(img, flow)
    #     print(output.shape)

class IterativeAttentionFusion(nn.Module):
    def __init__(self, channels=64, r=4):
        super(IterativeAttentionFusion, self).__init__()
        inter_channels = int(channels // r)

        # local attention
        self.local_attention = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # global attention
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # local attention2
        self.local_attention2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # global attention2
        self.global_attention2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, img, flow):
        x = img + flow
        x_channel = self.local_attention(x)
        x_pixel = self.global_attention(x)
        w = x_channel + x_pixel
        weight = self.sigmoid(w)
        x_ = img * weight + flow * (1 - weight)

        x_channel2 = self.local_attention2(x_)
        x_pixel2 = self.global_attention2(x_)
        w2 = x_channel2 + x_pixel2
        weight2 = self.sigmoid(w2)
        x_output = img * weight2 + flow * (1 - weight2)
        return x_output
    
    # def demo(self):
    #     img = torch.randn((8, 64, 720, 720))
    #     flow = torch.randn((8, 64, 720, 720))  
    #     output = self.forward(img, flow)
    #     print(output.shape)

class SelfAttention(SelfAttentionBlock):
    def __init__(self, channels, inter_channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(SelfAttention, self).__init__(
            key_in_channels=channels,
            query_in_channels=channels,
            channels=inter_channels,
            out_channels=channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            channels * 2,
            channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(SelfAttention, self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)
        return output
    
# (CSS) Category-Specific Semantic, please refer to https://github.com/NUST-Machine-Intelligence-Laboratory/HFAN/blob/main/mmseg/models/hfan/hfan_vos.py
class CSS(nn.Module):
    def __init__(self, scale):
        super(CSS, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        # b, c, h, w = feats.size()
        # print(batch_size, num_classes, height, width)
        # print(b, c, h, w)
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context
    
class Fusion(nn.Module):
    def __init__(self, channels=64, r=4, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), fusion_method='iaff'):
        super(Fusion, self).__init__()
        self.fusion_method = fusion_method

        if fusion_method == 'aff':
            self.fusion = AttentionFusion(channels, r)
        elif fusion_method == 'iaff':
            self.fusion = IterativeAttentionFusion(channels, r)
        elif fusion_method == 'cat':
            self.fusion = Concatenate()
        else:
            assert False, 'Fusion method not implemented!'
        assert self.fusion is not None
        
        inter_channels = int(channels // r)
        self.sa = SelfAttention(channels=channels, inter_channels=inter_channels, scale=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.css = CSS(scale=1)
        self.pred = ConvModule(channels, 2, kernel_size=1)
        
    def forward(self, img, flow):
        img_pred = self.pred(img)
        img_context = self.css(img, img_pred)
        i = self.sa(img, img_context)
        f = self.sa(flow, img_context)
        output = self.fusion(i, f)
        return output

class ReverseEdgeAttention(nn.Module):
    def __init__(self, channels):
        super(ReverseEdgeAttention, self).__init__()        
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, feature_input, feature_fuse):
        feature_avg = torch.mean(feature_input, dim=1, keepdim=True)
        feature_max, _ = torch.max(feature_input, dim=1, keepdim=True)
        feature = torch.cat([feature_avg, feature_max], dim=1)
        feature = self.conv1(feature).sigmoid()
        feature = 1 - feature
        res = feature_fuse * feature
        return self.conv2(res)
    
class IntraFusion(nn.Module):
    def __init__(self, channels_in, channels_out):
        ''' [LR, ..., HR] '''
        super(IntraFusion, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, channels_out), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, channels_out)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, channels_out), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, channels_out)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, channels_out), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, channels_out)
        )
        self.conv4   = nn.Sequential(
            nn.Conv2d(channels_out, channels_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(32, channels_in),
            nn.ReLU(inplace=True)
        )
        self.rea1 = ReverseEdgeAttention(channels_out)
        self.rea2 = ReverseEdgeAttention(channels_out)
        self.rea3 = ReverseEdgeAttention(channels_out)
        
    def forward(self, features):
        edges = []
        f1, f2, f3, f4 = features
        
        x1 = self.conv1_1(f1)
        x2 = self.conv1_2(f2)
        x = x1 + x2
        x = F.relu(x, inplace=True) 
        edge1 = self.rea1(x, x2)
        edges.append(edge1)
        
        x3 = self.conv2_1(x)
        x4 = self.conv2_2(f3)
        x = x3 + x4
        x = F.relu(x, inplace=True)
        edge2 = self.rea2(x, x4)
        edges.append(edge2)
        
        x5 = self.conv2_1(x)
        x6 = self.conv2_2(f4)
        x = x5 + x6
        x = F.relu(x, inplace=True)
        edge3 = self.rea3(x, x6)
        edges.append(edge3)
        
        mask = self.conv4(x)
        return mask, edges
    
class LinearEmbedding(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super(LinearEmbedding, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class Decoder_V2(BaseDecodeHead):
    def __init__(self, feature_strides, embedding_dim, visualization=False, method='intra', **kwargs):
        super(Decoder_V2, self).__init__(input_transform='multiple_select', **kwargs) # TODO: input transform
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.visualization = visualization
        self.method = method

        in_channels1, in_channels2, in_channels3, in_channels4 = self.in_channels # [64, 128, 320, 512]
        
        self.linear_embed4 = LinearEmbedding(input_dim=in_channels4, embed_dim=embedding_dim) # 256
        self.linear_embed3 = LinearEmbedding(input_dim=in_channels3, embed_dim=embedding_dim)
        self.linear_embed2 = LinearEmbedding(input_dim=in_channels2, embed_dim=embedding_dim)
        self.linear_embed1 = LinearEmbedding(input_dim=in_channels1, embed_dim=embedding_dim)

        # self.linear_fuse = ConvModule(
        #     in_channels=embedding_dim * 4,
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='BN', requires_grad=True)
        # )
        self.intra_fuse = IntraFusion(channels_in = embedding_dim, channels_out = embedding_dim // 2)

        self.linear = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.stage1 = Fusion(channels=in_channels1, r=in_channels1 // 16)
        self.stage2 = Fusion(channels=in_channels2, r=in_channels2 // 16)
        self.stage3 = Fusion(channels=in_channels3, r=in_channels3 // 16)
        self.stage4 = Fusion(channels=in_channels4, r=in_channels4 // 16)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # 1/4, 1/8, 1/16, 1/32
        x1, x2, x3, x4 = x
        im1, fw1 = split_batches(x1)
        im2, fw2 = split_batches(x2)
        im3, fw3 = split_batches(x3)
        im4, fw4 = split_batches(x4)

        n, _, h, w = im1.shape
        
        f1 = self.stage1(im1, fw1)
        f2 = self.stage2(im2, fw2)
        f3 = self.stage3(im3, fw3)
        f4 = self.stage4(im4, fw4)
        
        f1_ = self.linear_embed1(f1).permute(0, 2, 1).reshape(n, -1, f1.shape[2], f1.shape[3])
        f2_ = self.linear_embed2(f2).permute(0, 2, 1).reshape(n, -1, f2.shape[2], f2.shape[3])
        f3_ = self.linear_embed3(f3).permute(0, 2, 1).reshape(n, -1, f3.shape[2], f3.shape[3])
        f4_ = self.linear_embed4(f4).permute(0, 2, 1).reshape(n, -1, f4.shape[2], f4.shape[3])
        
        # TODO: visualization
        if self.visualization: 
            vis_feature_map(f1_.cpu(), title=f'af f1')
            vis_feature_map(f2_.cpu(), title=f'af f2')
            vis_feature_map(f3_.cpu(), title=f'af f3')
            vis_feature_map(f4_.cpu(), title=f'af f4')

        f_, edges = self.intra_fuse([f4_, f3_, f2_, f1_]) 
    
        if self.visualization: 
            for idx, edge in enumerate(edges):
                vis_feature_map(edge.cpu(), title=f'edge {idx}', policy='first')
            vis_feature_map(f_.cpu(), title=f'af f')
    
        f = self.dropout(f_)
        f = self.linear(f)
        return f, edges
        
        # if self.method == 'linear':
        #     f_ = self.linear_fuse(torch.cat([f4_, f3_, f2_, f1_], dim=1)) 

        #     if self.visualization: 
        #         vis_feature_map(f_.cpu(), title=f'af f')
            
        #     f = self.dropout(f_)
        #     f = self.linear(f)
        #     return f
        # elif self.method == 'intra':
        #     f_, edges = self.intra_fuse([f4_, f3_, f2_, f1_]) 
        
        #     if self.visualization: 
        #         for idx, edge in enumerate(edges):
        #             vis_feature_map(edge.cpu(), title=f'edge {idx}', policy='first')
        #         vis_feature_map(f_.cpu(), title=f'af f')
        
        #     f = self.dropout(f_)
        #     f = self.linear(f)
        #     return f, edges
        # else:
        #     assert False, 'Fusion method not implemented!'

class Net_V3(nn.Module):
    def __init__(self, visualization=False, edge=True):
        super(Net_V3, self).__init__()
        self.visualization = visualization
        self.edge = edge
        # encoder
        self.encoder = Backbone(pretrained=True)
        # decoder TODO: modify parameters
        self.decoder = Decoder_V2(
            embedding_dim = 256, 
            visualization = self.visualization, 
            method = 'intra' if edge else 'linear', 
            feature_strides = [4, 8, 16, 32], 
            in_index=[0, 1, 2, 3],
            in_channels = [64, 128, 320, 512],
            channels = 128, 
            dropout_ratio = 0.2, # dropout
            num_classes = 2,  
            out_channels = 1,
            threshold = 0.3, 
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True)
        ) 

        
    def forward(self, img, flow):
        '''
        img1: [B, C, H, W]
        img2: [B, C, H, W]
        flow: [B, C, H, W]
        '''
        rgb_fw = torch.cat([img, flow], dim=0)
        # encode
        features = self.encoder(rgb_fw)
        
        if self.visualization:      
            for idx, f in enumerate(features):
                vis_feature_map(f.cpu(), title=f'bf {idx}')
        
        # decode
        if self.edge:
            x, edges = self.decoder(features)
            return x, edges
        else:
            x = self.decoder(features)
            return x
