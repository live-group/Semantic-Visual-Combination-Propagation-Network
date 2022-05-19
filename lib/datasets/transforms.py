import torch
from torch.nn import functional as F


class Compose(object):

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, tensors):
    for t in self.transforms:
      tensors = t(tensors)
    return tensors

  def __repr__(self):
    format_string = self.__class__.__name__ + '('
    for t in self.transforms:
      format_string += '\n'
      format_string += '  {0}'.format(t)
    format_string += '\n)'
    return format_string


class Noramlize(object):

  def __call__(self, feature):
    assert feature.dim() == 1, 'invalid shape : {:}'.format(feature.shape)
    return F.normalize(feature, dim=0)

  def __repr__(self):
    return self.__class__.__name__ + '()'


class AddNoise(object):

  def __init__(self, mean, std):
    self.mean = mean
    self.std  = std

  def __call__(self, feature):
    noise = torch.randn_like(feature)
    noise.normal_(self.mean, self.std)
    return feature + noise
