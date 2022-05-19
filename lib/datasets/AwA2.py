import os, sys, copy, torch
import torch.utils.data as data


class AwA2_Simple(data.Dataset):

  def __init__(self, infos, mode, wnids, extra_embedding):
    super(AwA2_Simple, self).__init__()
    self.mode               = mode
    self.allclasses         = copy.deepcopy( infos['allclasses'] )
    self.wnids              = copy.deepcopy( wnids )
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
    self.attributes      = infos['attributes'].clone().float()
    extra_attrs          = []
    self.allclass2wnidx  = []
    for class_name in self.allclasses:
      wordnet_id = infos['class2wnid'][class_name]
      wordnet_idx= wnids.index(wordnet_id)
      extra_attrs.append( extra_embedding[wordnet_idx] )
      self.allclass2wnidx.append( wordnet_idx )
    self.extra_attrs     = torch.stack(extra_attrs).float()
    self.oriCLS2newCLS   = dict()
    self.oriCLS2wnidCLS  = dict()
    self.current_wnids   = []
    self.current_wnidxes = []

    for i, cls in enumerate(self.current_classes):
      self.oriCLS2newCLS[cls] = i
    for i, cls in enumerate(self.current_classes):
      wordnet_id = infos['class2wnid'][ self.allclasses[cls] ]
      self.current_wnids.append(wordnet_id)
      self.oriCLS2wnidCLS[cls] = wnids.index(wordnet_id)
      self.current_wnidxes.append( wnids.index(wordnet_id) )
    self.return_label_mode = 'original'

  def set_return_label_mode(self, mode):
    assert mode in ['original', 'new', 'wnid', 'combine']
    self.return_label_mode = mode

  def get_current_classes(self):
    returnLIST = []
    for x in self.current_classes:
      if self.return_label_mode == 'new':
        x = self.oriCLS2newCLS[x]
      elif self.return_label_mode == 'wnid':
        x = self.oriCLS2wnidCLS[x]
      returnLIST.append( x )
    return returnLIST

  def get_all_wnid_index(self):
    return copy.deepcopy( self.current_wnidxes ), copy.deepcopy( self.allclass2wnidx ), list(range(len(self.wnids)))

  def __getitem__(self, index):
    assert 0 <= index < len(self), 'invalid index = {:}'.format(index)
    ori_label = self.labels[index]
    if self.return_label_mode == 'original':
      return_label = ori_label
    elif self.return_label_mode == 'new':
      return_label = self.oriCLS2newCLS[ ori_label ]
    elif self.return_label_mode == 'wnid':
      return_label = self.oriCLS2wnidCLS[ ori_label ]
    elif self.return_label_mode == 'combine':
      return_label = (self.oriCLS2newCLS[ ori_label ], ori_label)
    else: raise ValueError('invalid mode = {:}'.format(self.return_label_mode))
    return self.features[index].clone(), self.attributes[ori_label].clone(), self.extra_attrs[ori_label].clone(), return_label

  def __repr__(self):
    return ('{name}({length:5d} samples with {num_classes} classes [{mode:}])'.format(name=self.__class__.__name__, length=len(self.labels), num_classes=self.num_classes, mode=self.mode))

  def __len__(self):
    return len(self.labels)

  # TODO: FIX             
  def loop_class_names(self):
    for i, (key, value) in enumerate(self.classIDX2name_wd.items()):
      assert i == key, 'invalid {:} vs. {:}'.format(i, key)
      yield value


class SubgraphSampler(object):
  def __init__(self, root, mode, classes_per_it, num_samples, iterations): 
    super(SubgraphSampler, self).__init__()
    self.root = Path(root)
    self.label_idx_fea = torch.load( self.root / "idx_fea_label.pth" ) ["{}_label_idx_fea".format(mode)]
    self.classes_per_it  = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations

  def __iter__(self):
    '''
    yield a batch of indexes
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      print("")  
