import os, re, sys, time, torch, random, argparse, math, json
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from PIL     import ImageFile
from copy    import deepcopy
from pathlib import Path
import torch.nn as nn
from semantic_networks import LinearEnsemble,LinearEnsemble1
from relation_networks import  PPNRelationNet
from eval_util import evaluate_all
import sklearn.linear_model as models

# This is used to make dirs in lib visiable to your python program
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: 
    sys.path.insert(0, str(lib_dir))

from sklearn.cluster import KMeans
from config_utils import Logger, time_string, convert_secs2time, AverageMeter
from config_utils import load_configure, count_parameters_in_MB
from datasets     import ZSHOT_DATA
from models       import distance_func
from datasets     import MetaSampler
from visuals      import encoder_template

def obtain_relation_models(name, att_dim, image_dim):
  model_name = name.split('-')[0]
     #att_name=2048
  _, att_C, hidden_C, degree, T = name.split('-')
    
  return PPNRelationNet(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))

def obtain_semantic_models(name, field_centers):
  model_name = name.split('-')[0]
  if model_name == 'Linear':
    _, out_dim = name.split('-')
    return LinearEnsemble(field_centers, int(out_dim))   #out_dim=2048

def obtain_semantic_models1(name, field_centers, num):
  model_name = name.split('-')[0]
  if model_name == 'Linear':
    _, out_dim = name.split('-')
    return LinearEnsemble1(field_centers, int(out_dim), num)



def train_model(xargs, loader, semantics,visual, test_class_att, test_adj_dis, adj_distances, network1,network2,relation_network,network3, optimizer, config, logger):
  
  batch_time, Xlosses, accs, end = AverageMeter(), AverageMeter(), AverageMeter(), time.time()
  labelMeter = AverageMeter()
  network1.train()
  network2.train()
  relation_network.train()
  network3.train()
  

  loader.dataset.set_return_label_mode('new')

  logger.print('[TRAIN---{:}], semantics-shape={:}, adj_distances-shape={:}, config={:}'.format(config.epoch_str, semantics.shape, adj_distances.shape, config))

  for batch_idx, (image_feats, targets) in enumerate(loader):
    
    batch_label_set = set(targets.tolist())   #训练类标签0-39
    batch_label_lst = list(batch_label_set)
    class_num       = len(batch_label_lst)
    batch, feat_dim = image_feats.shape
    batch_attrs     = semantics[batch_label_lst, :]  #训练类语义
    batch_adj_dis   = adj_distances[batch_label_lst,:][:,batch_label_lst]
    
    batch_vis_center= visual[batch_label_lst, :]
    
    
    
    
    re_att = network1(batch_attrs.cuda())
    re_vis = network2(batch_vis_center.cuda())
    relation1 ,att_gcn =  relation_network(image_feats.cuda(), re_att.cuda(),batch_adj_dis.cuda())
    relation2 ,vis_gcn =  relation_network(image_feats.cuda(), re_vis.cuda(),batch_adj_dis.cuda())

    
    new_target_idxs = [batch_label_lst.index(x) for x in targets.tolist()]
    new_target_idxs = torch.LongTensor(new_target_idxs)
    one_hot_labels  = torch.zeros(batch, class_num).scatter_(1, new_target_idxs.view(-1,1), 1).cuda()
    target__labels  = new_target_idxs.cuda()
    
          
    feature_loss = nn.MSELoss(reduction='sum').cuda()
    cat_loss     = nn.CrossEntropyLoss().cuda()
    re_semantic  = network3(att_gcn)
    re_semantic_loss             = feature_loss(re_semantic, batch_attrs.cuda()) / (batch )
    loss4  =  re_semantic_loss
    
    CE = nn.CrossEntropyLoss().cuda()
    
    if config.loss_type == 'sigmoid-mse':
      loss1 = CE(relation1 * xargs.cross_entropy_lambda, target__labels)
      loss2 = CE(relation2 * xargs.cross_entropy_lambda, target__labels)
      loss3 = F.mse_loss(torch.sigmoid(re_vis), torch.sigmoid(re_att), reduction='elementwise_mean')
      loss = loss1+xargs.lambda_1*loss2+xargs.lambda_2*loss3+xargs.lambda_3*loss4
    else:
      raise ValueError('invalid loss type : {:}'.format(config.loss_type))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    # analysis
    Xlosses.update(loss.item(), batch)
    predict_labels = torch.argmax(relation1, dim=1)
    with torch.no_grad():
      accuracy = (predict_labels.cpu() == new_target_idxs.cpu()).float().mean().item()
      accs.update(accuracy*100, batch)
      labelMeter.update(class_num, 1)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()  
  
    if batch_idx % config.log_interval == 0 or batch_idx + 1 == len(loader):
      Tstring = 'TIME[{batch_time.val:.2f} ({batch_time.avg:.2f})]'.format(batch_time=batch_time)
      Sstring = '{:} [{:}] [{:03d}/{:03d}]'.format(time_string(), config.epoch_str, batch_idx, len(loader))
      Astring = 'loss={:.7f} ({:.5f}), acc@1={:.1f} ({:.1f})'.format(Xlosses.val, Xlosses.avg, accs.val, accs.avg)
      logger.print('{:} {:} {:} B={:}, L={:} ({:.1f}) : {:}'.format(Sstring, Tstring, Astring, batch, class_num, labelMeter.avg, batch_label_lst[:3]))
  return Xlosses.avg, accs.avg


def main(xargs):
  # your main function
  # print some necessary informations
  # create logger
  if not os.path.exists(xargs.log_dir):
    os.makedirs(xargs.log_dir)
  logger = Logger(xargs.log_dir, xargs.manual_seed)
  logger.print ("args :\n{:}".format(xargs))

  assert torch.cuda.is_available(), 'You must have at least one GPU'

  # set random seed
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
  random.seed(xargs.manual_seed)
  np.random.seed(xargs.manual_seed)
  torch.manual_seed(xargs.manual_seed)
  torch.cuda.manual_seed(xargs.manual_seed)

  logger.print('Start Main with this file : {:}, and load {:}.'.format(__file__, xargs.data_root))
  graph_info = torch.load('xargs.data_root/x-AWA2-data.pth')
  
  # All labels return original value between 0-49
  train_dataset       = ZSHOT_DATA(graph_info, 'train')
  batch_size          = xargs.class_per_it * xargs.num_shot
  total_episode       = ((len(train_dataset) / batch_size) // 100 + 1) * 100
  train_sampler       = MetaSampler(train_dataset, total_episode, xargs.class_per_it, xargs.num_shot)
  train_loader        = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=xargs.num_workers)
  #train_loader        = torch.utils.data.DataLoader(train_dataset,       batch_size=batch_size, shuffle=True , num_workers=xargs.num_workers, drop_last=True)
  test_seen_dataset   = ZSHOT_DATA(graph_info, 'test-seen')
  test_seen_loader    = torch.utils.data.DataLoader(test_seen_dataset,   batch_size=batch_size, shuffle=False, num_workers=xargs.num_workers)
  test_unseen_dataset = ZSHOT_DATA(graph_info, 'test-unseen')
  test_unseen_loader  = torch.utils.data.DataLoader(test_unseen_dataset, batch_size=batch_size, shuffle=False, num_workers=xargs.num_workers)
  logger.print('train-dataset       : {:}'.format(train_dataset))
  #logger.print('train_sampler       : {:}'.format(train_sampler))
  logger.print('test-seen-dataset   : {:}'.format(test_seen_dataset))
  logger.print('test-unseen-dataset : {:}'.format(test_unseen_dataset))

  features       = graph_info['ori_attributes'].float().cuda()
  train_features = features[graph_info['train_classes'], :]
  test_class_att = features[graph_info['unseen_classes'], :]
  logger.print('feature-shape={:}, train-feature-shape={:}'.format(list(features.shape), list(train_features.shape)))

  kmeans = KMeans(n_clusters=xargs.clusters, random_state=1337).fit(train_features.cpu().numpy())
  #kmeans = KMeans(n_clusters=xargs.clusters, random_state=1337).fit(features.cpu().numpy())
  att_centers = torch.tensor(kmeans.cluster_centers_).float().cuda()
  #print(att_centers.shape)
  for cls in range(xargs.clusters):
    logger.print('[cluster : {:}] has {:} elements.'.format(cls, (kmeans.labels_ == cls).sum()))
  logger.print('Train-Feature-Shape={:}, use {:} clusters, shape={:}'.format(train_features.shape, xargs.clusters, att_centers.shape))

  # build adjacent matrix
  distances     = distance_func(graph_info['attributes'], graph_info['attributes'], 'euclidean-pow').float().cuda()
  xallx_adj_dis = distances.clone()
  train_adj_dis = distances[graph_info['train_classes'],:][:,graph_info['train_classes']]
  test_adj_dis  = distances[graph_info['unseen_classes'],:][:,graph_info['unseen_classes']]
  

  target_VC=[]
  train_center=[]
  is_first = True
  # if not os.path.exists('/media/cqu/D/SWL/zero-shot-propagation/prepare_data_AWA1_2/x-AWA1-CENTER.pth'):
    # print('aaaa')
  for x in graph_info['train_classes']:
  
    #print(x)
    trainval_label= np.array(graph_info['trainval_label'])
    idx = (trainval_label==x).nonzero()[0]
    
    train_features_all = graph_info['trainval_feature']
    train_features_1 = train_features_all[idx,:]
    
    sum=[0.0]*2048
    sum=np.array(sum)
    #print(sum.type())
    cnt=0
    for y in train_features_1:
        cnt+=1
        sum+=y

    sum/=cnt
    
    avg=torch.tensor(sum).float()
    avg=avg.unsqueeze(0)
    
    if is_first:
      is_first=False
      train_center = avg
    else:     
      train_center = torch.cat((train_center, avg),0) 
    target_VC = train_center.cuda()
 
  kmeans1 = KMeans(n_clusters=xargs.clusters, random_state=1330).fit(target_VC.cpu().numpy())
  vis_centers = torch.tensor(kmeans1.cluster_centers_).float().cuda()
  network1= obtain_semantic_models(xargs.semantic_name, att_centers)
  network2= obtain_semantic_models1(xargs.visual_name, vis_centers, xargs.class_per_it)
  relation_network = obtain_relation_models(xargs.relation_name, 2048, 2048)
  network1 = network1.cuda()
  network2 = network2.cuda()
  relation_network = relation_network.cuda()
 
  
  _,vis_dim = vis_centers.shape
  _,att_dim = att_centers.shape


  encoder_vis = encoder_template(vis_dim, xargs.latent_size, xargs.hidden_size_vis, xargs.hidden_size_sem, att_dim)
  network3= encoder_vis.cuda()
  
  parameters = [{'params': list(network1.parameters())},
              {'params': list(network2.parameters())}, 
              {'params': list(relation_network.parameters())},
              {'params': list(network3.parameters())},
              ]
  optimizer  = torch.optim.Adam(parameters, lr=xargs.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=xargs.weight_decay, amsgrad=False)
  #optimizer = torch.optim.SGD(parameters, lr=xargs.lr, momentum=0.9, weight_decay=xargs.weight_decay, nesterov=True)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.1, step_size=xargs.epochs*2//3)
  logger.print('network1 : {:.2f} MB =>>>\n{:}'.format(count_parameters_in_MB(network1), network1))
  logger.print('optimizer : {:}'.format(optimizer))
  
  #import pdb; pdb.set_trace()
  model_lst_path  = logger.checkpoint('ckp-last-{:}.pth'.format(xargs.manual_seed))
  if os.path.isfile(str(model_lst_path)):
    checkpoint  = torch.load(model_lst_path)
    start_epoch = checkpoint['epoch'] + 1
    best_accs   = checkpoint['best_accs']
    network.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger.print ('load checkpoint from {:}'.format(model_lst_path))
  else:
    start_epoch, best_accs = 0, {'train': -1, 'xtrain': -1, 'zs': -1, 'gzs-seen': -1, 'gzs-unseen': -1, 'gzs-H':-1, 'best-info': None}
  
  epoch_time, start_time = AverageMeter(), time.time()
  # training
  for iepoch in range(start_epoch, xargs.epochs):
    # set some classes as fake zero-shot classes
    time_str = convert_secs2time(epoch_time.val * (xargs.epochs- iepoch), True) 
    epoch_str= '{:03d}/{:03d}'.format(iepoch, xargs.epochs)
    logger.print ('Train the {:}-th epoch, {:}, LR={:1.6f} ~ {:1.6f}'.format(epoch_str, time_str, min(lr_scheduler.get_lr()), max(lr_scheduler.get_lr())))
  
    config_train = load_configure(None, {'epoch_str': epoch_str, 'log_interval': xargs.log_interval,
                                         'loss_type': xargs.loss_type}, None)

    train_cls_loss, train_acc = train_model(xargs, train_loader, train_features,target_VC,test_class_att,test_adj_dis, train_adj_dis, network1,network2,relation_network,network3,optimizer, config_train, logger)
    lr_scheduler.step()
    if train_acc > best_accs['train']: best_accs['train'] = train_acc
    logger.print('Train {:} done, cls-loss={:.3f}, accuracy={:.2f}%, (best={:.2f}).\n'.format(epoch_str, train_cls_loss, train_acc, best_accs['train']))

    if iepoch % xargs.test_interval == 0 or iepoch == xargs.epochs -1:
      with torch.no_grad():
        xinfo = {'train_classes' : graph_info['train_classes'], 'unseen_classes': graph_info['unseen_classes']}
        evaluate_all(epoch_str, train_loader, test_unseen_loader, test_seen_loader, features, xallx_adj_dis, network1,relation_network,  xinfo, best_accs, logger)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
  
  # the final evaluation
  logger.print('final evaluation --->>>')
  with torch.no_grad():
    xinfo = {'train_classes' : graph_info['train_classes'], 'unseen_classes': graph_info['unseen_classes']}
    evaluate_all('final-eval', train_loader, test_unseen_loader, test_seen_loader, features, xallx_adj_dis, network1,relation_network, xinfo, best_accs, logger)
  logger.print('-'*200)
  logger.close()

           
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--log_dir' ,       type=str, default='',help='Save dir.')
  parser.add_argument('--data_root' ,     type=str,  default='.../data/info-files' ,help='dataset root')
  parser.add_argument('--loss_type',      type=str,  default='sigmoid-mse',             help='The loss type.')
  parser.add_argument('--semantic_name',  type=str,   default='Linear-2048')
  parser.add_argument('--visual_name',    type=str,   default='Linear-2048')
  parser.add_argument('--relation_name',  type=str,   default='PPN-256-2048-40-10')
  parser.add_argument('--clusters',       type=int,   default=3)
  parser.add_argument('--class_per_it',   type=int,   default=30)
  parser.add_argument('--num_shot'    ,   type=int,   default=1)
  parser.add_argument('--epochs',         type=int,   default=500)
  parser.add_argument('--manual_seed',    type=int,   default=34194)
  #parser.add_argument('--manual_seed',    type=int,   default=34394)
  parser.add_argument('--lr',             type=float,  default=0.00002)
  parser.add_argument('--weight_decay',   type=float,  default=0.0001)
  parser.add_argument('--num_workers',    type=int,   default= 8,     help='The number of workers.')
  parser.add_argument('--log_interval',   type=int,   default=100,     help='The log-print interval.')
  parser.add_argument('--test_interval',  type=int,   default=5,     help='The evaluation interval.')
  parser.add_argument('--latent_size' ,   type=int,   default=256)
  parser.add_argument('--latent_size1' ,   type=int,   default=85)
  parser.add_argument('--latent_size2' ,   type=int,   default=2048)
  parser.add_argument('--drop_out',       type=float, default=0.5)
  parser.add_argument('--hidden_size_vis', default=[1024])
  parser.add_argument('--hidden_size_sem', default=[256])
  parser.add_argument('--cross_entropy_lambda',    type=float, default=0.01)
  parser.add_argument('--lambda_1',       type=float, default=0.0001)
  parser.add_argument('--lambda_2',       type=float, default=0.001)
  parser.add_argument('--lambda_3',       type=float, default=0.0001)
  args = parser.parse_args()

  if args.manual_seed is None or args.manual_seed < 0:
    args.manual_seed = random.randint(1, 100000)
  assert args.log_dir is not None, 'The log_dir argument can not be None.'
  main(args)