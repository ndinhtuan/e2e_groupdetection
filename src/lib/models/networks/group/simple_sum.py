from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class SimpleSum(torch.nn.Module):

    def __init__(self, opt):
        super(SimpleSum, self).__init__()
        self.embed_dim = opt.group_embed_dim

        self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 1),
        )

        self.fc.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, embed1, embed2, *args):
        
        cat_embed = torch.add(embed1, embed2)
        cat_embed = self.fc(cat_embed)
        cat_embed = torch.squeeze(cat_embed)

        return cat_embed

def get_group_simple_sum(opt):
    return SimpleSum(opt)
