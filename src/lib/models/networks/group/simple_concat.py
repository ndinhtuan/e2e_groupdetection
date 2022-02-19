from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import einops

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

    def forward(self, embed1, embed2, *args):
        cat_embed = torch.cat((embed1, embed2), 1)

        cat_embed = self.fc(cat_embed)

        cat_embed = torch.squeeze(cat_embed, 1)

        return cat_embed
    
# # dot product self attention
# class GroupDotSelfAttention(torch.nn.Module):
#     def __init__(self, embed_dim, num_heads=1, *args):
#         super(GroupDotSelfAttention, self).__init__()
#         self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#         self.cls = SimpleConcat(embed_dim)
    
#     def forward(self, embeds1, embeds2, id_head, **args):
#         query, key, value = None, None, None
#         query1 = embeds1.unsqueeze(0)
#         query2 = embeds2.unsqueeze(0)
#         key = id_head.unsqueeze(0)
#         value = key
        
#         attn_output1, _ = self.attn(query1, key, value) 
#         attn_output2, _ = self.attn(query2, key, value) 
        
#         # print("Query1", query1.shape)
#         # print("Query2", query2.shape)
#         # print("Key", key.shape)
#         # print("Value", value.shape)
#         # print("attn_output1", attn_output1.shape)
#         # print("attn_output2", attn_output2.shape)
        
#         output = self.cls(attn_output1.squeeze(0), attn_output2.squeeze(0))
#         return output
        

def get_group_simple_concat(embed_dim):
    return SimpleConcat(embed_dim)

# def get_group_attention_concat(embed_dim):
#     return GroupDotSelfAttention(embed_dim)