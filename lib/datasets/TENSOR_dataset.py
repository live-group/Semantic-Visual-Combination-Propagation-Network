import os, sys, copy, torch
from pathlib import Path
import numpy as np
import torch.utils.data as data
from PIL import Image


class TENSOR_DATA(data.Dataset):

  def __init__(self, infos, mode):
    super(TENSOR_DATA, self).__init__()
    self.mode               = mode
    self.allclasses         = copy.deepcopy( infos['allclasses'] )
    if mode == 'train':
      images, labels = infos['trainval_files']   , infos['trainval_label']
    elif mode == 'test-seen':
      images, labels = infos['test_seen_files']  , infos['test_seen_label']
    elif mode == 'test-unseen':
      images, labels = infos['test_unseen_files'], infos['test_unseen_label']
    elif mode == 'test':
      images = infos['test_seen_files'] + infos['test_unseen_files']
      labels = infos['test_seen_label'] + infos['test_unseen_label']
    else: 'invalid mode = {:}'.format(mode)
    cfile2tensor         = infos['image2feat']
    self.xfiles          = [cfile2tensor[x] for x in images]
    self.labels          = copy.deepcopy(labels)
    self.current_classes = sorted( list( set(self.labels) ) )
    self.num_classes     = len(self.current_classes)
    self.oriCLS2newCLS   = dict()
    for i, cls in enumerate(self.current_classes):
      self.oriCLS2newCLS[cls] = i
    self.return_label_mode = 'original'

  def set_return_label_mode(self, mode):
    assert mode in ['original', 'new', 'combine']
    self.return_label_mode = mode

  def __getitem__(self, index):
    assert 0 <= index < len(self), 'invalid index = {:}'.format(index)
    ori_label = self.labels[index]
    if self.return_label_mode == 'original':
      return_label = ori_label
    elif self.return_label_mode == 'new':
      return_label = self.oriCLS2newCLS[ ori_label ]
    elif self.return_label_mode == 'combine':
      return_label = (self.oriCLS2newCLS[ ori_label ], ori_label)
    else: raise ValueError('invalid mode = {:}'.format(self.return_label_mode))
    xdata = torch.load(self.xfiles[index], map_location='cpu')
    feats_000 = xdata['feats-000']
    feats_090 = xdata['feats-090']
    feats_180 = xdata['feats-180']
    feats_270 = xdata['feats-270']
    return feats_000[0], return_label

  def __repr__(self):
    return ('{name}({length:5d} samples with {num_classes} classes [{mode:}])'.format(name=self.__class__.__name__, length=len(self.labels), num_classes=self.num_classes, mode=self.mode))

  def __len__(self):
    return len(self.labels)
