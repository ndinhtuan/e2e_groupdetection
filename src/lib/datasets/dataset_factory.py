from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.fformation import JointDataset


def get_dataset(dataset, task):
  if task == 'group':
    return JointDataset
  else:
    return None
  
