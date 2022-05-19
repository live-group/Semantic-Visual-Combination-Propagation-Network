import os, torch
from collections import OrderedDict
from .distance_utils import distance_func
from .backbone import SimpleBlock 

def obtain_model(model_name, att_dim, image_dim, centroids=None):
  name = model_name.split('-')[0]
  if name == 'PPNSimple':
    from .PPNSimpleNet import PPNSimple
    _, att_hC, T, degree = model_name.split('-')
    return PPNSimple(att_dim, image_dim, int(att_hC), int(T), int(degree))
  elif name == 'PPNSemEnc':
    from .PPNSemEnc import PPNSemEnc
    _, att_hC, T, degree = model_name.split('-')
    return PPNSemEnc(att_dim, image_dim, int(att_hC), int(T), int(degree), centroids)
  elif name == 'DXYNet':
    from .DXYNet import DXYNet
    _, att_hC, T, degree = model_name.split('-')
    return DXYNet(att_dim, image_dim, int(att_hC), int(T), int(degree))
  elif name == 'DPPN':
    from .DPPN import DPPN
    _, att_hC, T, degree_a, degree_v = model_name.split('-')
    return DPPN(att_dim, image_dim, int(att_hC), int(T), int(degree_a), int(degree_v), centroids)
  elif name == 'DPPN':
    from .DPPN import DPPN
    _, att_hC, T, degree_a, degree_v = model_name.split('-')
    return DPPN(att_dim, image_dim, int(att_hC), int(T), int(degree_a), int(degree_v), centroids)
  elif name == 'DPPN_gg':
    from .DPPN_gg import DPPN_gg
    _, att_hC, T, degree_a, degree_v = model_name.split('-')
    return DPPN_gg(att_dim, image_dim, int(att_hC), int(T), int(degree_a), int(degree_v), centroids)
  elif name == 'DPPN_gg_trans':
    from .DPPN_gg_trans import DPPN_gg_trans
    _, att_hC, T, degree_a, degree_v = model_name.split('-')
    return DPPN_gg_trans(att_dim, image_dim, int(att_hC), int(T), int(degree_a), int(degree_v), centroids)
  elif name == 'DPPN_gg_endecoder':
    from .DPPN_gg_endecoder import DPPN_gg_endecoder
    _, att_hC, T, degree_a, degree_v = model_name.split('-')
    return DPPN_gg_endecoder(att_dim, image_dim, int(att_hC), int(T), int(degree_a), int(degree_v), centroids)
  else:
    raise ValueError('Invalid model-name : {:} vs. {:}'.format(name, model_name))


def obtain_backbone(cnn_name, config={'pretrained':True, 'pretrain_path': None}):
  from .backbone import resnet18, resnet50, resnet101 
  if cnn_name.lower() == 'resnet18':
    model = resnet18(pretrained=config['pretrained'])
  elif cnn_name.lower() == 'resnet50':
    model = resnet50(pretrained=config['pretrained'])
  elif cnn_name.lower() == 'resnet101':
    model = resnet101(pretrained=config['pretrained'])
  else:
    raise ValueError('invalid name : {:}'.format(name))
  if 'pretrain_path' in config and config['pretrain_path'] is not None:
    data = torch.load(config['pretrain_path'], map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in data['net-state-dict'].items():
      name = k[7:] # remove `module.`
      new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
  return model
