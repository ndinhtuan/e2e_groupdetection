import torch
from .simple_concat import SimpleConcat

# dot product self attention
class GroupDotSelfAttention(torch.nn.Module):
    def __init__(self, opt, num_heads=1, *args):
        super(GroupDotSelfAttention, self).__init__()
        self.attn = torch.nn.MultiheadAttention(opt.group_embed_dim, num_heads, batch_first=True)
        self.cls = SimpleConcat(opt)
    
    def forward(self, embeds1, embeds2, id_head, **args):
        query1 = embeds1.unsqueeze(1)
        query2 = embeds2.unsqueeze(1)
        kv = torch.repeat_interleave(id_head.unsqueeze(0), query2.shape[0], dim=0) # repeat to add batch dimension as query 

        print("Query1", query1.shape)
        print("Query2", query2.shape)
        
        try:
            attn_output1, _ = self.attn(query1, kv, kv) 
            attn_output2, _ = self.attn(query2, kv, kv) 

            print("Attn1", attn_output1.shape)
            print("Attn2", attn_output2.shape)
        
            output = self.cls(attn_output1.squeeze(1), attn_output2.squeeze(1))
        except:
            import IPython
            IPython.embed()

        return output
        

def get_group_attention_concat(opt):
    return GroupDotSelfAttention(opt)