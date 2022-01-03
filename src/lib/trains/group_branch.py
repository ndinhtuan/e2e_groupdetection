from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class SimpleConcat(torch.nn.Module):

    def __init__(self, opt):
        super(SimpleConcat, self).__init__()
        self.embed_dim = 128#opt.embed_dim

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

    def forward(self, embed1, embed2):
        
        cat_embed = torch.cat((embed1, embed2), 1)
        cat_embed = self.fc(cat_embed)
        cat_embed = torch.squeeze(cat_embed)

        return cat_embed

def pair_sampling(group_embeds, ids, number_samples, positive=True):

    r"""

    Args:
        group_embeds : Tensor for embedding of group batch_size x max_number of person x embed_dim
        ids: Tensor for id of batch - batch_size x max_number of person. People in same group have 
            same id, people in group with size 1 have id = -1
        number_samples(int): number of sample for sampling for each data in batch
        positive (boolean): sampling positive or sampling negative
    Returns:
        embeds1 : Tensor number_samples x embed_dim
        embeds2 : Tensor number_samples x embed_dim
    """

    return _pair_sampling(group_embeds, ids, number_samples, positive)

def _pair_sampling(group_embeds, ids, number_samples, positive=True):
    
    batch_size = group_embeds.shape[0]
    embed_dim = group_embdes.shape[2]
    embeds1 = torch.zeros((batch_size*number_samples, embed_dim))
    embeds2 = torch.zeros((batch_size*number_samples, embed_dim))

    for i in range(number_samples):
        
        id_ = ids[i]
        group_embed = group_embeds[i]

        id_samples = None

        if positive:
            id_samples = _positive_id_pair_generator(id_, number_samples)
        else:
            id_samples = _negative_id_pair_generator(id_, number_samples)
        
        id_samples_1 = id_samples[:, 0].numpy()
        id_samples_2 = id_samples[:, 1].numpy()
        
        embeds1[i*number_samples:(i+1)*number_samples,:] = group_embed[id_samples_1]
        embeds2[i*number_samples:(i+1)*number_samples,:] = group_embed[id_samples_2]

    return embeds1, embeds2

def _positive_id_pair_generator(id_, number_samples):

    r"""
    This function generates id positive pair for one image

    Args:
        id_: Tensor n x 1, saving group label for person
        number_samples: number of sampling we want to take
    Return:
        Tensor number_samples x 2: positive pairs in one image
    """
    unique_group_id = torch.unique(id_).numpy()
    combinations = None

    for group_id in unique_group_id:

        if group_id > 0: # Remove alone people and no object
            list_id = (id_==group_id).nonzero(as_tuple=True)[0]
            _combinations = torch.combinations(list_id)

            if combinations is None:
                combinations = _combinations
            else:
                combinations = torch.cat([combinations, _combinations])
    
    combinations = combinations[torch.randperm(combinations.size()[0])]
    return combinations[:number_samples,:]

def _negative_id_pair_generator(id_, number_samples):
    
    r"""
    This function generates id negative pair for one image

    Args:
        id_: Tensor n x 1, saving group label for person
        number_samples: number of sampling we want to take
    Return:
        Tensor number_samples x 2: negative pairs in one image
    """

    unique_group_id = torch.unique(id_).numpy()
    combinations = None

    num_group = len(unique_group_id)

    for i in range(num_group):

        group_id_1 = unique_group_id[i]

        for j in range(i+1, num_group):
            
            group_id_2 = unique_group_id[j]

            if group_id_1 > 0 and group_id_2 > 0:

                list_id_1 = (id_==group_id_1).nonzero(as_tuple=True)[0]
                list_id_2 = (id_==group_id_2).nonzero(as_tuple=True)[0]
                
                for i1 in list_id_1.numpy():
                    for i2 in list_id_2.numpy():

                        _combinations = torch.Tensor([id1, id2], dtype=torch.int32)
                        if combinations is None:
                            combinations = _combinations
                        else:
                            combinations = torch.cat([combinations, _combinations])
    
    combinations = combinations[torch.randperm(combinations.size()[0])]
    return combinations[:number_samples,:]
