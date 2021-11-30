# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from .modules import PositionEmbeddingLearned

class TransformerDecoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU(inplace=True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm2(tgt)
        memory = self.norm_mem(memory)
        tgt2, mask = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.d_model=d_model
        self.decoder = TransformerDecoderLayerPreNorm(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward) 
        self.self_posembed = PositionEmbeddingLearned(3, d_model)
        self.cross_posembed = PositionEmbeddingLearned(3, d_model)

    def forward(self, point, point_feature, group_point, group_feature):
        point_embed = self.self_posembed(point)
        query = point_feature + point_embed
        B, C, np, ns = group_point.shape
        group_embed = self.cross_posembed(group_point.permute(0, 2, 3, 1).reshape(B*np, ns, C)).permute(2, 0, 1)
        key = group_feature.permute(0, 2, 3, 1).reshape(-1, ns, self.d_model).transpose(0, 1) + group_embed
        features = self.decoder(query.transpose(1, 2).reshape(-1, self.d_model).unsqueeze(0), key)
        features = features.reshape(-1, B, np, self.d_model).transpose(2, 3).squeeze(0)
        return features
 

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0, width=1, depth=2):
        super().__init__()
        self.depth = depth
        self.width = width
        nhead = 8

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlp=[input_feature_dim] + [64 * width for i in range(depth)] + [128 * width],
            use_xyz=True,
            normalize_xyz=True
        )
        self.dec1 = DecoderBlock(
            128, nhead=nhead, dim_feedforward=128*2, 
        ) 

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )
        self.dec2 = DecoderBlock(
            256, nhead=nhead, dim_feedforward=256*2, 
        ) 

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )
        self.dec3 = DecoderBlock(
            256, nhead=nhead, dim_feedforward=256*2, 
        ) 

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )
        self.dec4 = DecoderBlock(
            256, nhead=nhead, dim_feedforward=256*2, 
        ) 

        self.fp1 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, 256 * width])
        self.fp2 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, 288])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, grouped_xyz, grouped_features, fps_inds = self.sa1(xyz, features)    # grouped_features : B, C, np, ns
        features = self.dec1(xyz, features, grouped_xyz, grouped_features).contiguous()
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, grouped_xyz, grouped_features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        features = self.dec2(xyz, features, grouped_xyz, grouped_features).contiguous()
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, grouped_xyz, grouped_features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        features = self.dec4(xyz, features, grouped_xyz, grouped_features).contiguous()
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, grouped_xyz, grouped_features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        features = self.dec4(xyz, features, grouped_xyz, grouped_features).contiguous()
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
                            end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds

        return end_points


if __name__ == '__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16, 20000, 6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
