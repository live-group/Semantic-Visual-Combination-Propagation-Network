import torch.nn as nn
import torch.nn.functional as F
import math, torch
from models import distance_func
from networks import LinearEnsemble 


class DPPN_gg(nn.Module):
  """docstring for PPN"""
  def __init__(self, att_dim, image_dim, att_hC, T, degree_a, degree_v, att_centroids):
    super(DPPN_gg, self).__init__()
    self.att_g_a     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    #self.att_h_a     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    self.att_g_v     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    #self.att_h_v     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    nn.init.kaiming_uniform_(self.att_g_a, a=math.sqrt(5))
    #nn.init.kaiming_uniform_(self.att_h_a, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.att_g_v, a=math.sqrt(5))
    #nn.init.kaiming_uniform_(self.att_h_v, a=math.sqrt(5))
    self.T         = T
    assert degree_a >= 0 and degree_a < 100, 'invalid degree : {:}'.format(degree_a)
    assert degree_v >= 0 and degree_v < 100, 'invalid degree : {:}'.format(degree_v)
    self.degree_a    = degree_a
    self.degree_v    = degree_v
    self.thresh_a    = math.cos(math.pi*degree_a/180)
    self.thresh_v    = math.cos(math.pi*degree_v/180)
    self.att_centroids = att_centroids 
    # assume img dim the same with the attri transform dim

  def extra_repr(self):
    g_a_shape = 'att_g_a-shape: {:}'.format(list(self.att_g_a.shape))
    #h_a_shape = 'att_h_a-shape: {:}'.format(list(self.att_h_a.shape))
    g_v_shape = 'att_g_v-shape: {:}'.format(list(self.att_g_v.shape))
    #h_v_shape = 'att_h_v-shape: {:}'.format(list(self.att_h_v.shape))
    return ('{name}(degree_a={degree_a:}, degree_v={degree_v:}, thresh_a={thresh_a:.3f}, thresh_v={thresh_v:.3f}, temperature={T:}, '.format(name=self.__class__.__name__, **self.__dict__) + g_a_shape + g_v_shape + ')')

  def get_attention(self, attributes):
    attributes, _ = self._encode_attr(attributes)
    att_prop_g = torch.mm(attributes, self.att_g_v)
    att_prop_h = torch.mm(attributes, self.att_g_v)
    distances  = distance_func(att_prop_g, att_prop_h, 'cosine')
    zero_vec   = -9e15 * torch.ones_like(distances)
    raw_attss  = torch.where(distances > self.thresh_v, distances, zero_vec) * self.T
    attention  = F.softmax(raw_attss, dim=1)
    #attention  = F.softmax(raw_attss * self.T, dim=1)
    return raw_attss, attention

  def _encode_attr(self, attributes):
    att_prop_g = torch.mm(attributes, self.att_g_a)
    att_prop_h = torch.mm(attributes, self.att_g_a)
    distances  = distance_func(att_prop_g, att_prop_h, 'cosine')
    zero_vec   = -9e15 * torch.ones_like(distances)
    raw_attss  = torch.where(distances > self.thresh_a, distances, zero_vec)
    attention  = F.softmax(raw_attss * self.T, dim=1)
    att_outs   = torch.mm(attention, attributes)
    return att_outs, distances > self.thresh_a

  def forward(self, image_feats, attributes, labels, target_proto_labels):
    # img_avg propagation, no extra transform, just attention
    _, attention = self.get_attention(attributes)
    protos = []
    for cls in target_proto_labels:
      proto = image_feats[labels==cls].mean(dim=0)
      protos.append( proto )
    protos = torch.stack(protos)
    att_protos = torch.mm(attention, protos)
    return att_protos
