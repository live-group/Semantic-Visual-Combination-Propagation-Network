import torch.nn as nn
import torch.nn.functional as F
import math, torch
from distance_utils import distance_func



class PPNRelationNet(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, att_dim, image_dim, att_hC, hidden_C, T, degree):
    super(PPNRelationNet, self).__init__()
    print(att_dim,image_dim,att_hC,hidden_C)
    self.att_g     = nn.Parameter(torch.Tensor(att_dim, att_hC))  #1024*256
    
    
    self.att_v     = nn.Parameter(torch.Tensor(hidden_C, att_hC))
    nn.init.kaiming_uniform_(self.att_v, a=math.sqrt(5))
    
    nn.init.kaiming_uniform_(self.att_g, a=math.sqrt(5))
    self.T         = T
    assert degree >= 0 and degree < 100, 'invalid degree : {:}'.format(degree)
    self.degree    = degree
    self.thresh    = math.cos(math.pi*degree/180)
    #self.att_h     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    #init.kaiming_uniform_(self.att_h, a=math.sqrt(5))
    #nn.init.xavier_uniform_(self.att_g.data, gain=1.414)
    #nn.init.xavier_uniform_(self.att_h.data, gain=1.414)
    self.img_w     = nn.Parameter(torch.Tensor(image_dim, hidden_C))
    self.sem_w     = nn.Parameter(torch.Tensor(image_dim, hidden_C))
    self.sem_b     = nn.Parameter(torch.Tensor(1, hidden_C))
    nn.init.kaiming_uniform_(self.img_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.sem_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.sem_b.data, a=math.sqrt(5))
    fan_in, _      = nn.init._calculate_fan_in_and_fan_out(self.img_w)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(self.sem_b, -bound, bound)
    #nn.init.xavier_uniform_(self.img_w.data, gain=1.414)
    #nn.init.xavier_uniform_(self.sem_w.data, gain=1.414)
    #nn.init.xavier_uniform_(self.sem_b.data, gain=1.414)
    self.fc        = nn.Linear(hidden_C, 1)
     
    self.fc1 = nn.Linear(att_dim, hidden_C)
    
  def extra_repr(self):
    xshape = 'att-shape: {:}'.format(list(self.att_g.shape)) + ' img-w-shape : {:}'.format(list(self.img_w.shape)) + ' sem-w-shape : {:}'.format(list(self.sem_w.shape))
    return ('{name}(degree={degree:}, thresh={thresh:.3f}, temperature={T:}, '.format(name=self.__class__.__name__, **self.__dict__) + xshape + ')')

  def get_new_attribute(self, attributes):
    att_prop_g = torch.mm(attributes, self.att_g)
    att_prop_h = torch.mm(attributes, self.att_g)
    distances  = distance_func(att_prop_g, att_prop_h, 'cosine')
    zero_vec   = -9e15 * torch.ones_like(distances)
    raw_attss  = torch.where(distances > self.thresh, distances, zero_vec)
    attention  = F.softmax(raw_attss * self.T, dim=1)
    att_outs   = torch.mm(attention, attributes)
    return att_outs, distances > self.thresh
  

  def forward(self, image_feats, attributes, _):
    # attribute propgation
    cls_num, at_dim = attributes.shape
    att_outs, _ = self.get_new_attribute(attributes)
    
    batch, feat_dim = image_feats.shape
    #att_outs  = F.relu(self.fc1(att_outs))

    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    att_feats_ext   = att_outs.view(1, cls_num, -1).expand(batch, cls_num, feat_dim)
    hidden_feats    = F.relu( torch.matmul(image_feats_ext, self.img_w) + torch.matmul(att_feats_ext, self.sem_w) + self.sem_b )
    #hidden_feats    = F.sigmoid( torch.matmul(image_feats_ext, self.img_w) + torch.matmul(att_feats_ext, self.sem_w) + self.sem_b )
    outputs         = self.fc(hidden_feats)
    return outputs.view(batch, cls_num),att_outs