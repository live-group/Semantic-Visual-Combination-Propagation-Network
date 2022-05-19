import os, sys, copy, torch
from pathlib import Path
import numpy as np
import random
import torch.utils.data as data
from PIL import Image


class ZSHOT_DATA(data.Dataset):

  def __init__(self, infos, mode):
    super(ZSHOT_DATA, self).__init__()
    self.mode               = mode
    self.allclasses         = copy.deepcopy( infos['allclasses'] )
    if mode == 'train':
      features, labels = infos['trainval_feature']   , infos['trainval_label']
    elif mode == 'test-seen':
      features, labels = infos['test_seen_feature']  , infos['test_seen_label']
    elif mode == 'test-unseen':
      features, labels = infos['test_unseen_feature'], infos['test_unseen_label']
    elif mode == 'test':
      features = torch.cat([infos['test_seen_feature'], infos['test_unseen_feature']], dim=0)
      labels   = infos['test_seen_label'] + infos['test_unseen_label']
    else: 'invalid mode = {:}'.format(mode)
    self.features        = features.clone().float()
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
    return self.features[index].clone(), return_label

  def __repr__(self):
    return ('{name}({length:5d} samples with {num_classes} classes [{mode:}])'.format(name=self.__class__.__name__, length=len(self.labels), num_classes=self.num_classes, mode=self.mode))

  def __len__(self):
    return len(self.labels)


def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


class AwA2_IMG(data.Dataset):

  def __init__(self, infos, mode, transform):
    super(AwA2_IMG, self).__init__()
    self.mode               = mode
    self.allclasses         = copy.deepcopy( infos['allclasses'] )
    self.transform          = transform
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
    self.images          = images
    self.labels          = copy.deepcopy(labels)
    self.current_classes = sorted( list( set(self.labels) ) )
    self.num_classes     = len(self.current_classes)
    self.attributes      = infos['attributes'].clone().float()
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
    image = self.images[index]
    image = pil_loader(image)
    if self.transform is not None:
      image = self.transform( image )
    return image.clone(), return_label

  def __repr__(self):
    return ('{name}({length:5d} samples with {num_classes} classes [{mode:}], transform={transform:})'.format(name=self.__class__.__name__, length=len(self.labels), num_classes=self.num_classes, mode=self.mode, transform=self.transform))

  def __len__(self):
    return len(self.labels)


class AwA2_IMG_Rotate(data.Dataset):

  def __init__(self, infos, mode, transform):
    super(AwA2_IMG_Rotate, self).__init__()
    self.mode               = mode
    self.allclasses         = copy.deepcopy( infos['allclasses'] )
    self.transform          = transform
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
    self.images          = images
    self.labels          = copy.deepcopy(labels)
    self.current_classes = sorted( list( set(self.labels) ) )
    self.num_classes     = len(self.current_classes)
    self.attributes      = infos['attributes'].clone().float()
    self.oriCLS2newCLS   = dict()
    for i, cls in enumerate(self.current_classes):
      self.oriCLS2newCLS[cls] = i
    self.return_label_mode = 'original'

  def set_return_label_mode(self, mode):
    assert mode in ['original', 'new', 'combine']
    self.return_label_mode = mode

  def set_return_img_mode(self, mode):
    assert mode in ['original', 'rotate']
    self.return_img_mode = mode

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
    image = self.images[index]
    image = pil_loader(image)
    if self.return_img_mode == "original":
      return_img = self.transform(image)
    elif self.return_img_mode == "rotate":
      rotated_imgs = [
                    self.transform(image),
                    self.transform(image.rotate(90)),
                    self.transform(image.rotate(180)),
                    self.transform(image.rotate(270))
                ]
      return_img = torch.stack(rotated_imgs, dim=0)
    else: raise ValueError("Invalid return_img_mode: {}".format(self.return_img_mode))
    return return_img.clone(), return_label

  def __repr__(self):
    return ('{name}({length:5d} samples with {num_classes} classes [{mode:}], transform={transform:})'.format(name=self.__class__.__name__, length=len(self.labels), num_classes=self.num_classes, mode=self.mode, transform=self.transform))

  def __len__(self):
    return len(self.labels)

class AwA2_IMG_Rotate_Save(data.Dataset):

  def __init__(self, infos, mode):
    super(AwA2_IMG_Rotate_Save, self).__init__()
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
    self.image2feat      = infos['image2feat']
    #self.image2feat      = infos['image2featmap']
    self.images          = images
    self.labels          = copy.deepcopy(labels)
    self.current_classes = sorted( list( set(self.labels) ) )
    self.num_classes     = len(self.current_classes)
    self.attributes      = infos['attributes'].clone().float()
    self.oriCLS2newCLS   = dict()
    for i, cls in enumerate(self.current_classes):
      self.oriCLS2newCLS[cls] = i
    self.return_label_mode = 'original'

  def set_return_label_mode(self, mode):
    assert mode in ['original', 'new', 'combine']
    self.return_label_mode = mode

  def set_return_img_mode(self, mode):
    assert mode in ['original', 'rotate', 'original_augment']
    self.return_img_mode = mode

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
    image_path = self.images[index]
    feats_path = self.image2feat[image_path]
    feats      = torch.load(feats_path)
    if self.return_img_mode == "original":
      return_img   = feats['feats-000'][0] 
    elif self.return_img_mode == "original_augment":
      n_aug = len(feats['feats-000']) # include test transform
      return_img   = feats['feats-000'][random.randint(0,n_aug-1)]
    elif self.return_img_mode == "rotate":
      n_aug = len(feats['feats-000']) # include test transform
      rotated_imgs = [
                feats['feats-000'][random.randint(0,n_aug-1)], 
                feats['feats-090'][random.randint(0,n_aug-1)], 
                feats['feats-180'][random.randint(0,n_aug-1)], 
                feats['feats-270'][random.randint(0,n_aug-1)] 
                ]
      # only rotate different degrees
      #rotated_imgs = [feats['feats-000'][0], feats['feats-090'][0], feats['feats-180'][0], feats['feats-270'][0]]
      # one img, different augmentation
      #which_degree = random.randint(0,3)
      #rotated_imgs  = feats['feats-{:03d}'.format(which_degree * 90)]     
      #return_img = rotated_imgs

      return_img = torch.stack(rotated_imgs, dim=0)
    else: raise ValueError("Invalid return_img_mode: {}".format(self.return_img_mode))
    return return_img.clone(), return_label

  def __repr__(self):
    return '{name}({length:5d} samples with {num_classes} classes [{mode:})'.format(name=self.__class__.__name__, length=len(self.labels), num_classes=self.num_classes, mode=self.mode) 

  def __len__(self):
    return len(self.labels)
