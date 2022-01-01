from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .groupdet import GroupDetTrainer


train_factory = {
  'group': GroupDetTrainer,
}
