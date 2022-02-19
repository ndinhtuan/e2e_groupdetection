import torch
import einops
from .simple_concat import SimpleConcat

# dot product self attention
class GroupDotSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads=1, *args):
        super(GroupDotSelfAttention, self).__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cls = SimpleConcat(embed_dim)
    
    def forward(self, embeds1, embeds2, id_head, **args):
        query, key, value = None, None, None
        query1 = embeds1.unsqueeze(0)
        query2 = embeds2.unsqueeze(0)
        key = id_head.unsqueeze(0)
        value = key
        
        attn_output1, _ = self.attn(query1, key, value) 
        attn_output2, _ = self.attn(query2, key, value) 
        
        output = self.cls(attn_output1.squeeze(0), attn_output2.squeeze(0))
        return output
        

def get_group_attention_concat(embed_dim):
    return GroupDotSelfAttention(embed_dim)