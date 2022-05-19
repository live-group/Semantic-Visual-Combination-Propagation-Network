import torch.nn as nn
import torch.nn.functional as F
import math, torch
from models import distance_func
from networks import LinearEnsemble 


class PPNSemEnc(nn.Module):
  """docstring for PPN"""
  def __init__(self, att_dim, image_dim, att_hC, T, degree, att_centroids):
    super(PPNSemEnc, self).__init__()
    self.att_g     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    #self.att_h     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    nn.init.kaiming_uniform_(self.att_g, a=math.sqrt(5))
    #nn.init.kaiming_uniform_(self.att_h, a=math.sqrt(5))
    self.T         = T
    assert degree >= 0 and degree < 100, 'invalid degree : {:}'.format(degree)
    self.degree    = degree
    self.thresh    = math.cos(math.pi*degree/180)
    self.att_centroids = att_centroids 
    # assume img dim the same with the attri transform dim
    self.expert    = LinearEnsemble(att_centroids, att_dim)

  def extra_repr(self):
    xshape = 'att-shape: {:}'.format(list(self.att_g.shape))
    return ('{name}(degree={degree:}, thresh={thresh:.3f}, temperature={T:}, '.format(name=self.__class__.__name__, **self.__dict__) + xshape + ')')

  def get_attention(self, attributes):
    att_prop_g = torch.mm(attributes, self.att_g)
    att_prop_h = torch.mm(attributes, self.att_g)
    distances  = distance_func(att_prop_g, att_prop_h, 'cosine')
    zero_vec   = -9e15 * torch.ones_like(distances)
    raw_attss  = torch.where(distances > self.thresh, distances, zero_vec) * self.T
    attention  = F.softmax(raw_attss, dim=1)
    #attention  = F.softmax(raw_attss * self.T, dim=1)
    return raw_attss, attention

  def encode_attr(self, attributes):
    return self.expert(attributes)

  def forward(self, image_feats, attributes, labels, target_proto_labels):
    # attribute propgation
    cls_num, at_dim = attributes.shape
    attributes = self.encode_attr(attributes)
    _, attention = self.get_attention(attributes)

    protos = []
    for cls in target_proto_labels:
      proto = image_feats[labels==cls].mean(dim=0)
      protos.append( proto )
    protos = torch.stack(protos)
    att_protos = torch.mm(attention, protos)
    return att_protos
