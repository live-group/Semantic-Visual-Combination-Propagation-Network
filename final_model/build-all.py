import os, json, time, random, argparse, torch
from pathlib import Path
import scipy.io as sio
from collections import defaultdict
import numpy as np


def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string


dataset_root   = Path(os.environ["HOME"]) / 'data' 
save_dir       = Path(os.environ["HOME"]) / 'data' / 'info-files'
save_dir.mkdir(parents=True, exist_ok=True)


def preprocess_data(dataset_name, save_file):
  print('{:} process {:}.'.format(time_string(), dataset_name))
  # split dataset
  split_dir      = dataset_root / "xlsa17" / "data" / dataset_name
  matcontent     = sio.loadmat(str(split_dir / "res101.mat"))
  img_feature    = matcontent['features'].T
  img_label      = matcontent['labels'].astype(int).squeeze() - 1
  image_files    = matcontent['image_files']
  image_files    = [x[0][0] for x in image_files]
  # load info
  matcontent     = sio.loadmat(str(split_dir / "att_splits.mat"))
  allclasses     = [x[0][0] for x in matcontent['allclasses_names']]

  # get specific information
  trainval_loc        = matcontent['trainval_loc'].squeeze() - 1
  trainval_feature    = img_feature[trainval_loc]
  trainval_label      = img_label[trainval_loc]
  trainval_classes    = set(trainval_label.tolist())

  test_seen_loc       = matcontent['test_seen_loc'].squeeze() - 1
  test_seen_feature   = img_feature[test_seen_loc]
  test_seen_label     = img_label[test_seen_loc]

  test_unseen_loc     = matcontent['test_unseen_loc'].squeeze() - 1
  test_unseen_feature = img_feature[test_unseen_loc]
  test_unseen_label   = img_label[test_unseen_loc]

  attributes          = torch.from_numpy(matcontent['att'].T)
  if dataset_name in ['APY', 'SUN']:
    ori_attributes    = torch.from_numpy(matcontent['original_att'].T) * 100
  elif dataset_name in ['AWA1', 'AWA2', 'CUB']:
    ori_attributes    = torch.from_numpy(matcontent['original_att'].T)
  else:
    raise ValueError('invalid dataset-name: {:}'.format(dataset_name))

  train_classes  = sorted( list(set(trainval_label.tolist())) )
  unseen_classes = sorted( list(set(test_unseen_label.tolist())) )

  all_info = {'allclasses'         : allclasses,                         # the list of all class names
              'train_classes'      : train_classes,                      # the list of train classes
              'unseen_classes'     : unseen_classes,                     # the list of unseen classes
              'trainval_feature'   : torch.from_numpy(trainval_feature), # the PyTorch tensor, 23527 * 2048
              'trainval_label'     : trainval_label.tolist(),            # a list of 23527 labels
              'test_seen_feature'  : torch.from_numpy(test_seen_feature),# the PyTorch tensor, 5882 * 2048
              'test_seen_label'    : test_seen_label.tolist(),           # a list of 5882 labels
              'test_unseen_feature': torch.from_numpy(test_unseen_feature), # the PyTorch tensor, 7913 * 2048
              'test_unseen_label'  : test_unseen_label.tolist(),         # a list of 7913 labels
              'attributes'         : attributes,                         # a 50 * 85 PyTorch tensor
              'ori_attributes'     : ori_attributes                      # a 50 * 85 PyTorch tensor
           }
  torch.save(all_info, save_file)
  print('Save all-info into {:}, file size : {:.2f} GB'.format(save_file, os.path.getsize(save_file)/1e9))


if __name__ == '__main__':
  names = ['APY', 'AWA1', 'AWA2', 'CUB', 'SUN']
  for name in names:
    save_file = save_dir / 'x-{:}-data.pth'.format(name)
    preprocess_data(name, save_file)
