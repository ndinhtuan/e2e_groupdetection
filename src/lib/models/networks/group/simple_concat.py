from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class SimpleConcat(torch.nn.Module):
    def __init__(self, embed_dim):
        super(SimpleConcat, self).__init__()
        self.embed_dim = embed_dim

        self.fc = torch.nn.Sequential(
                torch.nn.Linear(2*self.embed_dim, 100),
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

    def forward(self, embed1, embed2, additional_embed=None):
        if additional_embed is None:
            cat_embed = torch.cat((embed1, embed2), 1)
        else:   
            cat_embed = torch.cat((embed1, embed2, additional_embed), 1)

        cat_embed = self.fc(cat_embed)
        cat_embed = torch.squeeze(cat_embed)

        return cat_embed


def get_group_simple_concat(embed_dim):
    return SimpleConcat(embed_dim)