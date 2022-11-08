import sys
import os
root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

joint_id_to_name = {
  0: 'Head',        8: 'Torso',
  1: 'Neck',        9: 'R Hip',
  2: 'R Shoulder',  10: 'L Hip',
  3: 'L Shoulder',  11: 'R Knee',
  4: 'R Elbow',     12: 'L Knee',
  5: 'L Elbow',     13: 'R Foot',
  6: 'R Hand',      14: 'L Foot',
  7: 'L Hand',
}

anchors = torch.Tensor([[16*x+4*i+2, 16*y+4*j+2] for j in range(4) for i in range(4) for y in range(16) for x in range(16)])
anchors = anchors.permute(1,0).contiguous().view(1,1,2,-1).cuda().float()
input_size = 256

class post_process(nn.Module):
    
    def __init__(self, is_3D=False):
        super(post_process, self).__init__()
        self.is_3D = is_3D
        
    def forward(self, heads, mean, box):
        
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
  
        reg = anchors + regressions
        reg_weight = F.softmax(classifications, dim=-1)

        # anchor_map = reg_weight[0].view(-1,4,4,16,16).cpu().permute(0,3,1,4,2)
        # anchor_map = anchor_map.contiguous().view(-1,64,64)
        # depth_map = depthregressions[0].view(-1,4,4,16,16).cpu().permute(0,3,1,4,2)
        # depth_map = depth_map.contiguous().view(-1,64,64)
        # for i in range(15):
        #     plt.imshow(anchor_map[i,:,:], cmap='hot')
        #     plt.savefig(os.path.join(root, 'res/') + str(joint_id_to_name[i]) + '.png')
        #     plt.imshow(depth_map[i,:,:], cmap='hot')
        #     plt.savefig(os.path.join(root, 'res/') + str(joint_id_to_name[i]) + '_depth.png')


        if self.is_3D:
            Pred = (reg_weight * torch.cat([reg, depthregressions + mean.view(-1, 1, 1, 1)], 2)).sum(dim=-1)
        else:
            Pred = (reg_weight * reg).sum(dim=-1)
            
        Pred[:,:,0] = Pred[:,:,0] * (box[:,2:3] - box[:,0:1]) / input_size + box[:,0:1]
        Pred[:,:,1] = Pred[:,:,1] * (box[:,3:4] - box[:,1:2]) / input_size + box[:,1:2]

        return Pred
        
class A2J_loss(nn.Module):
    
    def __init__(self, is_3D=False):
        super(A2J_loss, self).__init__()
        self.is_3D = is_3D

    def forward(self, heads, label, box):

        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads

        reg = anchors + regressions
        reg_weight = F.softmax(classifications, dim=-1)
        if self.is_3D:
            Pred = (reg_weight * torch.cat([reg, depthregressions], 2)).sum(dim=-1)
        else:
            Pred = (reg_weight * reg).sum(dim=-1)
            label = label[:,:,:2]

        regression_loss = torch.abs(label - Pred)
        if self.is_3D:
            regression_loss[:,:,2] *= 64
        regression_loss = regression_loss ** 2
        return regression_loss.sum()