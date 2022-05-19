import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class LinearModule(nn.Module):

  def __init__(self, field_center, out_dim):
    super(LinearModule, self).__init__()
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.register_buffer('field_center', field_center.clone())
    self.fc = nn.Linear(field_center.numel(), out_dim)

  def forward(self, semantic_vec):
    input_offsets = semantic_vec - self.field_center
    response = F.relu(self.fc(input_offsets))
    return response


class LinearEnsemble(nn.Module):

  def __init__(self, field_centers, out_dim):
    super(LinearEnsemble, self).__init__()
    self.individuals = nn.ModuleList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    self.require_adj = False
    for i in range(field_centers.shape[0]):
      layer = LinearModule(field_centers[i], out_dim)
      self.individuals.append( layer )

  def forward(self, semantic_vec):
    responses = [indiv(semantic_vec) for indiv in self.individuals]
    feature_anchor = sum(responses)
    return feature_anchor
    
    
    
class LinearModule1(nn.Module):

  def __init__(self, field_center, out_dim, num):
    super(LinearModule1, self).__init__()
    
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.register_buffer('field_center', field_center.clone())
    self.fc = nn.Linear(field_center.numel(), out_dim)
    self.w     = nn.Parameter(torch.Tensor(field_center.numel(), out_dim))
    self.b     = nn.Parameter(torch.Tensor(num, out_dim))
    nn.init.kaiming_uniform_(self.b.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.w.data, a=math.sqrt(5))

  def forward(self, semantic_vec):
    input_offsets = semantic_vec - self.field_center
    #input_offsets = F.relu(self.fc(input_offsets))
    #input_offsets = F.relu(torch.mm(input_offsets, self.w)+ self.b)
    input_offsets=F.relu(input_offsets+self.b)
    return input_offsets


class LinearEnsemble1(nn.Module):

  def __init__(self, field_centers, out_dim, num):
    super(LinearEnsemble1, self).__init__()
    self.individuals = nn.ModuleList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    self.require_adj = False
    for i in range(field_centers.shape[0]):
      layer = LinearModule1(field_centers[i], out_dim, num)
      self.individuals.append( layer )

  def forward(self, semantic_vec):
    responses = [indiv(semantic_vec) for indiv in self.individuals]
    feature_anchor = sum(responses)
    return feature_anchor