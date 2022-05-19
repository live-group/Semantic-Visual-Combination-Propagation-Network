import os, sys, time, torch, random, argparse, math, json
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from copy    import deepcopy
from pathlib import Path

from datasets        import AwA2_Simple


def plt_protos(data_root, n_hop, new_protos, save_path):
  from sklearn.manifold import TSNE
  from matplotlib import pyplot as plt
  plt.switch_backend('agg')
  from matplotlib import cm
  from collections import OrderedDict
  # your main function
  # print some necessary informations
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  assert torch.cuda.is_available(), 'You must have at least one GPU'

  # set random seed
  torch.backends.cudnn.benchmark = True

  awa2_graph_info = torch.load(Path(data_root) / 'awa2-hop-{:}-graph.pth'.format(n_hop), map_location='cpu')
  wnids           = awa2_graph_info['wnids']
  adj_noself      = awa2_graph_info['all_matrix']
  adj_self        = adj_noself + torch.eye(len(wnids))
  graph_embedding = awa2_graph_info['vectors'].clone()      # is actually glove features

  awa2_all_infos  = torch.load(Path(data_root) / 'awa2-hop-{:}-infos.pth'.format(n_hop), map_location='cpu')

  # All labels return original value between 0-49
  train_dataset       = AwA2_Simple(awa2_all_infos, 'train', wnids, graph_embedding)
  train_loader        = torch.utils.data.DataLoader(train_dataset      , batch_size=128, shuffle=True , num_workers=4)
  test_seen_dataset   = AwA2_Simple(awa2_all_infos, 'test-seen', wnids, graph_embedding)
  test_seen_loader    = torch.utils.data.DataLoader(test_seen_dataset  , batch_size=128, shuffle=False, num_workers=4)
  test_unseen_dataset = AwA2_Simple(awa2_all_infos, 'test-unseen', wnids, graph_embedding)
  test_unseen_loader  = torch.utils.data.DataLoader(test_unseen_dataset, batch_size=128, shuffle=False, num_workers=4)

  wnid2image_feature_avg, all_cls_img_feat_avg = awa2_all_infos['wnid2feature_avg'], awa2_all_infos['all_cls_feature_avg']
  allclasses, attributes, know_wnid_ids = awa2_all_infos['allclasses'], awa2_all_infos['attributes'], []
  graph_img_features = []
  graph_att_features = []
  for idx, wnid in enumerate(wnids):
    if wnid in wnid2image_feature_avg:
      img_feature = wnid2image_feature_avg[wnid]
      class_label = allclasses.index( awa2_graph_info['wnid2name'][wnid] )
      att_feature = attributes[class_label]
      know_wnid_ids.append( idx )
    else:
      img_feature = all_cls_img_feat_avg
      att_feature = attributes.mean(dim=0)
    graph_img_features.append( img_feature.float() )
    graph_att_features.append( att_feature )

  #graph_img_features = torch.stack( graph_img_features ).float()
  #graph_att_features = torch.stack( graph_att_features ).float()

  #model_path = 'logs/DXYS-FEAT-V2/Hop_2-R_0-awa2_train-CasShareV1-128-stage-0-1-euclidean-pow-LR0.0001-B512-WD0.00005/ckp-last-86596.pth'
  #model_path = 'logs/baseline-HOP_2-R-w_0.6-LRD0-knn-All-vae/ckp-last-11111.pth'
  # 605 * 2048
  new_protos = new_protos.cpu()
  old_protos = graph_img_features
  
  train_img_features = train_loader.dataset.features
  sam_idx = random.sample(list(range(len(train_img_features))), int(len(train_img_features)/5))
  train_img_features = train_img_features[sam_idx]
  test_unseen_img_features = test_unseen_loader.dataset.features
  test_seen_img_features = test_seen_loader.dataset.features
  num_train_img_features = train_img_features.shape[0]
  num_test_unseen_img_features = test_unseen_img_features.shape[0]
  num_test_seen_img_features = test_seen_img_features.shape[0]

  seen_wnids = train_loader.dataset.current_wnids
  unseen_wnids = test_unseen_loader.dataset.current_wnids
  assert test_seen_loader.dataset.current_wnids == seen_wnids

  awa2_new_protos_seen, awa2_old_protos_seen = [], []
  awa2_new_protos_unseen, awa2_old_protos_unseen = [], []
  

  # train img features; seen img features; unseen img features
  # [old and new] train protos; seen ---; unseen ---  
  for wnid, new_proto, old_proto in zip(wnids, new_protos, old_protos):
    if wnid in seen_wnids:
      awa2_new_protos_seen.append(new_proto)
      awa2_old_protos_seen.append(old_proto)  
    elif wnid in unseen_wnids:
      awa2_new_protos_unseen.append(new_proto) 
      awa2_old_protos_unseen.append(old_proto) 
    else: print("class {} not in awa2".format(wnid))
    
  awa2_new_protos_seen   = torch.stack(awa2_new_protos_seen)
  awa2_old_protos_seen   = torch.stack(awa2_old_protos_seen)
  awa2_new_protos_unseen = torch.stack(awa2_new_protos_unseen)  
  awa2_old_protos_unseen = torch.stack(awa2_old_protos_unseen)
  num_protos_seen   = len(awa2_new_protos_seen)
  num_protos_unseen = len(awa2_new_protos_unseen)

  features = torch.cat([train_img_features, test_unseen_img_features, test_seen_img_features, awa2_new_protos_seen, awa2_old_protos_seen, awa2_new_protos_unseen, awa2_old_protos_unseen], dim=0)

   
  pth_path  = "{:}/tsne.pth".format(save_path)
  if os.path.exists(pth_path):
    features_2d = torch.load(pth_path)
  else: 
    tsne = TSNE()
    features    = np.array(features)
    features_2d = tsne.fit_transform(features)
    torch.save(features_2d, pth_path)
    print("save tsne-feature into {}".format(pth_path))

  plt.figure(figsize=(15, 15)) 
  
  def jet(m):
    cm_subsection = np.linspace(0, 1, m)
    colors = [ cm.jet(x) for x in cm_subsection ]
    J = np.array(colors)
    J = J[:, :3]
    return J

  idx_0 = num_train_img_features
  idx_1 = idx_0+num_test_unseen_img_features
  idx_2 = idx_1+num_test_seen_img_features
  idx_3 = idx_2+num_protos_seen
  idx_4 = idx_3+num_protos_seen
  idx_5 = idx_4+num_protos_unseen
  idx_6 = idx_5+num_protos_unseen
  assert idx_6 == len(features_2d)

  point_size = 60
  color_lst = jet(7) 
  for idx, fea in enumerate(features):
    if 0<= idx < idx_0:
      plt.scatter(features_2d[idx,0], features_2d[idx,1], marker='.', s=point_size, color=color_lst[0], label="train_img_features")
    elif idx_0 <= idx < idx_1: 
      plt.scatter(features_2d[idx,0], features_2d[idx,1], marker='.', s=point_size, color=color_lst[1], label="test_unseen_img_features")
    elif idx_1 <= idx < idx_2: 
      plt.scatter(features_2d[idx,0], features_2d[idx,1], marker='.', s=point_size, color=color_lst[2], label="test_seen_img_features")
    elif idx_2 <= idx < idx_3: 
      plt.scatter(features_2d[idx,0], features_2d[idx,1], marker='+', s=point_size*10, color=color_lst[3], label="new_protos_seen")
    elif idx_3 <= idx < idx_4: 
      plt.scatter(features_2d[idx,0], features_2d[idx,1], marker='+', s=point_size*10, color=color_lst[4], label="old_protos_seen")
    elif idx_4 <= idx < idx_5: 
      plt.scatter(features_2d[idx,0], features_2d[idx,1], marker='+', s=point_size*10, color=color_lst[5], label="new_protos_unseen")
    elif idx_5 <= idx < idx_6: 
      plt.scatter(features_2d[idx,0], features_2d[idx,1], marker='+', s=point_size*10, color=color_lst[6], label="old_protos_unseen")
    else: raise ValueError("invalid idx {}".format(idx))

  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

  save_tsne_path = '{}/t-sne-img-proto.pdf'.format(save_path)
  plt.savefig(save_tsne_path, transparent=True, bbox_inches='tight', pad_inches=0.001)
  print("save into {}".format(save_tsne_path))
