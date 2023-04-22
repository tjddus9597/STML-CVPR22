import torch, math, time, argparse, random, os, warnings
import dataset, net, utils, loss
import numpy as np
import torch.nn.functional as F
import wandb

from net.resnet import *
from net.inception import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate
from adamp import AdamP
from tqdm import *

import SSL.moco
import SSL.MeanShift

# os.environ['WANDB_MODE'] = 'offline'

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of `Self-Taught Metric Learning without Labels`'  
    + 'Our code is modified from `https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/`'
)
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR',
                    default='./logs',
                    help='Path to log folder'
                    )
# Dataset
parser.add_argument('--DATA_DIR', default='/home/tjddus9597/Data/DML', help='Path of data')
parser.add_argument('--dataset', default='cub', help='Training dataset, e.g. cub, cars, SOP', choices=['cub', 'cars', 'SOP'])

parser.add_argument('--embedding_size', default=512, type=int,
    help='Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default=120, type=int,
    dest='sz_batch',
    help='Number of samples per batch.'
)
parser.add_argument('--epochs', default=90, type=int,
    dest='nb_epochs',
    help='Number of training epochs.'
)
parser.add_argument('--gpu-id', default=0, type=int,
    help='ID of GPU that is used for training.'
)
parser.add_argument('--workers', default=8, type=int,
    dest='nb_workers',
    help='Number of workers for dataloader.'
)
parser.add_argument('--model', default='bn_inception',
    help='Model for training'
)
parser.add_argument('--optimizer', default='adamp',
    help='Optimizer setting'
)
parser.add_argument('--lr', default=1e-4, type=float,
    help='Learning rate setting'
)
parser.add_argument('--emb-lr', type=float,
    help='Learning rate Multiplication for embedding layer'
)
parser.add_argument('--weight-decay', default=1e-2, type=float,
    help='Weight decay setting'
)
parser.add_argument('--bn-freeze', default=0, type=int,
    help='Batch normalization parameter freeze'
)
parser.add_argument('--norm', default=1, type=int,
    help='teacher L2 normalization'
)
parser.add_argument('--save', default=0, type=int,
    help='Save checkpoint'
)
parser.add_argument('--resume', default='',
    help='Loading checkpoint'
)
parser.add_argument('--view', default=2, type=int,
    help='choose augmentation view'
)
parser.add_argument('--momentum', default=0.999, type=float,
    help='Momentum Update Parameter'
)
parser.add_argument('--pretrained', default=1, type=float,
    help='Training with ImageNet pretrained model'
)
parser.add_argument('--swav', default=0, type=int,
    help='Training with SwAV pretrained model'
)
parser.add_argument('--remark', default='',
    help='Any reamrk'
)
parser.add_argument('--seed', default=None, type=int,
    help='seed for initializing training. '
)

# SSL hyper-parameters
parser.add_argument('--memory-size', default=9600, type=int, help='size of memory.')
parser.add_argument('--topk', default=5, type=int, help='Nearest Neighbor for MeanShift')
parser.add_argument('--method', default='moco', choices=['moco', 'meanshift', 'byol'], type=str, help='SSL method')

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_{}_lr{}_batch{}{}'.format(args.dataset, args.model, args.method, args.embedding_size, args.optimizer, args.lr, args.sz_batch, args.remark)
# Wandb Initialization
wandb.init(project='STML_target', notes=LOG_DIR, name='{}_view{}_momentum{}_{}'.format(args.method, args.view, args.momentum, args.remark))
wandb.config.update(args)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/{}/{}_{}_{}'.format(args.model, args.dataset, args.model, args.remark)
LOG_DIR = os.path.abspath(LOG_DIR)
DATA_DIR = os.path.abspath(args.DATA_DIR)

# Dataset Loader and Sampler
is_inception = (args.model == 'bn_inception' or args.model == 'googlenet')

trn_dataset = dataset.load(
    name=args.dataset,
    root=DATA_DIR,
    mode='train',
    transform=dataset.utils.MultiTransforms(
        is_train=True,
        is_inception=is_inception,
        view=args.view))

dl_tr = torch.utils.data.DataLoader(
    trn_dataset,
    batch_size=args.sz_batch,
    shuffle=True,
    num_workers=args.nb_workers,
    drop_last=True,
    pin_memory=True
)
print('Random Sampling')

ev_dataset = dataset.load(
    name=args.dataset,
    root=DATA_DIR,
    mode='eval',
    transform=dataset.utils.MultiTransforms(
        is_train=False,
        is_inception=is_inception,
        view=1)
)

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size=args.sz_batch,
    shuffle=False,
    num_workers=args.nb_workers,
    pin_memory=True
)

# Student Model
if args.model.find('googlenet') + 1:
    model = inception_v1(embedding_size=args.embedding_size, pretrained=args.pretrained, is_norm=args.norm, is_student=True)
elif args.model.find('bn_inception') + 1:
    model = bn_inception(embedding_size=args.embedding_size, pretrained=args.pretrained, is_norm=args.norm, bn_freeze=args.bn_freeze, is_student=True)
elif args.model.find('resnet18') + 1:
    model = Resnet18(embedding_size=args.embedding_size, pretrained=args.pretrained, is_norm=args.norm, bn_freeze=args.bn_freeze, is_student=True)
elif args.model.find('resnet50') + 1:
    model = Resnet50(args.embedding_size, args.bg_embedding_size, args.pretrained, args.norm, True, bn_freeze=args.bn_freeze, swav_pretrained=args.swav, is_student=True)

print("=> creating model '{}'".format(args.model))
if args.method == 'moco' or args.method == 'MoCo':
    model_SSL = SSL.moco.MoCo(model, args.embedding_size, m=args.momentum, mem_bank_size=args.memory_size)
elif args.method == 'MSF' or args.method == 'meanshift':
    model_SSL = SSL.MeanShift.MeanShift(model, args.embedding_size, m=args.momentum, mem_bank_size=args.memory_size, topk=args.topk)
elif args.method == 'byol' or args.method == 'BYOL':
    model_SSL = SSL.MeanShift.MeanShift(model, args.embedding_size, m=args.momentum, mem_bank_size=args.memory_size, topk=1)

model_SSL = model_SSL.cuda(args.gpu_id)

if os.path.isfile(args.resume):
    print('=> teacher Loading Checkpoint {}'.format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cpu'.format(0))
    model_teacher.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> teacher No Checkpoint {}!!!!!!!!!!!!!'.format(args.resume))

if args.gpu_id == -1:
    model_SSL = nn.DataParallel(model_SSL)

param_groups = model_SSL.parameters()
# Optimizer Setting
if args.optimizer == 'sgd':
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, momentum=0.9)
elif args.optimizer == 'adam':
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay=args.weight_decay, momentum=0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))
elif args.optimizer == 'adamp':
    opt = AdamP(param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay), nesterov=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.nb_epochs)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_recall = [0]
best_epoch = 0
iteration = 0

for epoch in range(0, args.nb_epochs):
    same_idxs = []

    if args.bn_freeze:
        modules = model.encoder_q.model.modules() if args.gpu_id != -1 else model.encoder_q.module.model.modules()
        for m in modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []

    pbar = tqdm(enumerate(dl_tr))
    for batch_idx, (x, y, idx) in pbar:
        x_q = x[0].cuda(args.gpu_id, non_blocking=True)
        x_k = x[1].cuda(args.gpu_id, non_blocking=True)

        loss = model_SSL(x_q, x_k)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses_per_epoch.append(loss.item())

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Momentum {:.6f} LR {:.6f} Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr), args.momentum, opt.param_groups[0]["lr"], loss.item()))
        iteration += 1

    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()

    wandb.log({'learning_rate': opt.param_groups[0]["lr"]}, step=epoch)

    if(epoch >= 0):
        with torch.no_grad():
            print("**Evaluating...**")
            if args.dataset != 'SOP':
                k_list = [1, 2, 4, 8]
            else:
                k_list = [1, 10, 100, 1000]
            Recalls = utils.evaluate_euclid(model_SSL.encoder_q, dl_ev, k_list)
        
        for i in range(len(k_list)):
            wandb.log({"R@{}".format(k_list[i]): Recalls[i]}, step=epoch)

        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            print('Acheive best performance!!, Best Recall@1:{}'.format(best_recall[0]))

        if args.save:
            if not os.path.exists('{}'.format(LOG_DIR)):
                os.makedirs('{}'.format(LOG_DIR))
            torch.save({'model_state_dict': model_SSL.state_dict() if args.gpu_id != -1 else model_SSL.module.state_dict()},
                       '{}/best.pth'.format(LOG_DIR))
            with open('{}/best_results.txt'.format(LOG_DIR), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                if args.dataset != 'SOP':
                    for i in range(4):
                        f.write("Best Recall@{}: {:.4f}\n".format(2**i, best_recall[i] * 100))
                else:
                    for i in range(4):
                        f.write("Best Recall@{}: {:.4f}\n".format(10**i, best_recall[i] * 100))
            print('Save Best Model!')
    print('Best Epoch: {}, Best Recall@1:{}'.format(best_epoch, best_recall))
