import os, sys, time, random, argparse
#from .share_args import add_shared_args

def obtain_basic_args():
  parser = argparse.ArgumentParser(description='Train.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--log_dir' ,       type=str,                   help='Save dir.')
  parser.add_argument('--config_path',    type=str,                   help='The configuration path.')
  parser.add_argument('--strategy',       type=str,                   help='The training strategies.')
  parser.add_argument('--preprocessing',  type=str,                   help='The training strategies.')
  parser.add_argument('--num_workers',    type=int,  default=8,    help='')
  # Optimization options
  parser.add_argument('--n_hop',          type=int,                   help='Batch size for training.')
  parser.add_argument('--batch_size',     type=int,  default=2,       help='Batch size for training.')
  parser.add_argument('--epochs',         type=int,                   help='Batch size for training.')
  parser.add_argument('--manual_seed',    type=int,                   help='Batch size for training.')
  parser.add_argument('--lr',             type=float,                 help='Batch size for training.')
  parser.add_argument('--lr_step',        type=int,                   help='Batch size for training.')
  parser.add_argument('--lr_gamma',       type=float,                 help='Batch size for training.')
  parser.add_argument('--weight_decay',   type=float,                 help='Batch size for training.')
  parser.add_argument('--data_root' ,     type=str,                   help='dataset root')
  parser.add_argument('--log_interval',   type=int,   default=10,     help='The log-print interval.')
  parser.add_argument('--test_interval',  type=int,   default=10,     help='The evaluation interval.')
  parser.add_argument('--reconstruct_w',  type=float,               help='The reconstruction loss, if None, not use reconstruct error.')
  parser.add_argument('--LRD_w',  type=float,               help='The reconstruction loss, if None, not use reconstruct error.')
  parser.add_argument('--classifier',     type=str,         help='')
  parser.add_argument('--multiple',       type=float,                   help='The evaluation interval.')
  parser.add_argument('--train_over',     type=str,                   help='Training over all class or awa2 or awa2_train')
  parser.add_argument('--alpha',          type=float,               help='alpha in beta sampling')
  parser.add_argument('--mixup',          action='store_true',                help='')
  parser.add_argument('--start_epoch',    type=int,                help='')
  parser.add_argument('--syth_w',         type=float,                help='')
  parser.add_argument('--network_name',     type=str,                   help='Training over all class or awa2 or awa2_train')
  parser.add_argument('--pre_train_path',     type=str,                   help='Training over all class or awa2 or awa2_train')

    
  args = parser.parse_args()

  if args.manual_seed is None or args.manual_seed < 0:
    args.manual_seed = random.randint(1, 100000)
  assert args.log_dir is not None, 'The log_dir argument can not be None.'
  return args
