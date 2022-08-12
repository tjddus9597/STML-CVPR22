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
from tqdm import *


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of `Self-Taught Metric Learning without Labels`'  
    + 'Our code is modified from `https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/`'
)
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', 
    default='./logs',
    help = 'Path to log folder'
)
# Dataset
parser.add_argument('--DATA_DIR',  default='./data', help = 'Path of data')
parser.add_argument('--dataset', default='cub', help = 'Training dataset, e.g. cub, cars, SOP', choices=['cub', 'cars', 'SOP'])

parser.add_argument('--embedding_size', default = 512, type = int,
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--bg_embedding_size', default = 1024, type = int,
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 120, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = 90, type = int,
    dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training, and "-1" means DataParallel'
)
parser.add_argument('--workers', default = 8, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'bn_inception',
    help = 'Model for training'
)
parser.add_argument('--optimizer', default = 'adamp',
    help = 'Optimizer setting'
)
parser.add_argument('--lr', default = 1e-4, type = float,
    help = 'Learning rate setting'
)
parser.add_argument('--emb-lr', default = 1e-4, type =float,
    help = 'Learning rate for embedding layer'
)
parser.add_argument('--fix_lr', default = False, type = utils.bool_flag,
    help = 'Learning rate Fixing'
)
parser.add_argument('--weight_decay', default = 1e-2, type = float,
    help = 'Weight decay setting'
)
parser.add_argument('--num_neighbors', default = 5, type = int,
    help = 'For balanced sampling, the number of neighbors per query'
)
parser.add_argument('--bn_freeze', default = 0, type = int,
    help = 'Batch normalization parameter freeze'
)
parser.add_argument('--student_norm', default = 0, type = int,
    help = 'student L2 normalization'
)
parser.add_argument('--teacher_norm', default = 1, type = int,
    help = 'teacher L2 normalization'
)
parser.add_argument('--save', default = True, type = utils.bool_flag,
    help = 'Save checkpoint'
)
parser.add_argument('--resume', default = '',
    help = 'Loading checkpoint'
)
parser.add_argument('--view', default = 2, type = int,
    help = 'choose augmentation view'
)
parser.add_argument('--delta', default = 1, type = float,
    help = 'Delta value of Relaxed Contrastive Loss'
)
parser.add_argument('--sigma', default = 1, type = float,
    help = 'Sigma value of Relaxed Contrastive Loss'
)
parser.add_argument('--momentum', default = 0.999, type = float,
    help = 'Momentum Update Parameter'
)
parser.add_argument('--pretrained', default = True, type = utils.bool_flag,
    help = 'Training with ImageNet pretrained model'
)
parser.add_argument('--swav', default = False, type = utils.bool_flag,
    help = 'Training with SwAV pretrained model'
)
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

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
model_name = 'STML_{}_embedding{}_{}_lr{}_batch{}'.format(args.dataset, args.model, args.embedding_size, args.optimizer, args.lr, args.sz_batch)
LOG_DIR = args.LOG_DIR + '/logs_{}/{}'.format(args.dataset, model_name)
# Wandb Initialization
wandb.init(project='STML', notes=LOG_DIR, name = '{}'.format(model_name))
wandb.config.update(args)

LOG_DIR = os.path.abspath(LOG_DIR)
DATA_DIR = os.path.abspath(args.DATA_DIR)

# Dataset Loader and Sampler
is_inception = (args.model == 'bn_inception' or args.model == 'googlenet')

dataset_sampling = dataset.load(
    name = args.dataset,
    root = DATA_DIR,
    mode = 'train',
    transform = dataset.utils.Transform_for_Sampler(
        is_train = False, 
        is_inception = is_inception)
)

dl_sampling = torch.utils.data.DataLoader(
    dataset_sampling,
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True)

trn_dataset = dataset.load(
        name = args.dataset,
        root = DATA_DIR,
        mode = 'train',
        transform = dataset.utils.MultiTransforms(
            is_train = True, 
            is_inception = is_inception,
            view = args.view))

dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = args.sz_batch,
        shuffle = True,
        num_workers = args.nb_workers,
        drop_last = True,
        pin_memory = True
    )

ev_dataset = dataset.load(
        name = args.dataset,
        root = DATA_DIR,
        mode = 'eval',
        transform = dataset.utils.make_transform(
            is_train = False, 
            is_inception = is_inception)
)

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)

# Student Model
if args.model.find('googlenet')+1:
    model_student = inception_v1(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm, True)
elif args.model.find('bn_inception')+1:
    model_student = bn_inception(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm, True, bn_freeze = args.bn_freeze)
elif args.model.find('resnet18')+1:
    model_student = Resnet18(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm, True, bn_freeze = args.bn_freeze)
elif args.model.find('resnet50')+1:
    model_student = Resnet50(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm, True, bn_freeze = args.bn_freeze, swav_pretrained=args.swav)
    
# Teacher Model
if args.model.find('googlenet')+1:
    model_teacher = inception_v1(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm, False)
elif args.model.find('bn_inception')+1:
    model_teacher = bn_inception(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm, False, bn_freeze = args.bn_freeze)
elif args.model.find('resnet18')+1:
    model_teacher = Resnet18(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm, False, bn_freeze = args.bn_freeze)
elif args.model.find('resnet50')+1:
    model_teacher = Resnet50(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm, False, bn_freeze = args.bn_freeze, swav_pretrained=args.swav)

model_student = model_student.cuda()
model_teacher = model_teacher.cuda()
for param in list(set(model_teacher.parameters())):
    param.requires_grad = False
    
if os.path.isfile(args.resume):
    print('=> teacher Loading Checkpoint {}'.format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cpu'.format(0))
    model_teacher.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> teacher No Checkpoint {}!!!!!!!!!!!!!'.format(args.resume))
    
if os.path.isfile(args.resume):
    print('=> student Loading Checkpoint {}'.format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cpu'.format(0))
    model_student.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> student No Checkpoint {}!!!!!!!!!!!!!'.format(args.resume))
    
if args.gpu_id == -1:
    model_teacher = nn.DataParallel(model_teacher)
    model_student = nn.DataParallel(model_student)
stml_criterion = loss.STML_loss(delta = args.delta, sigma = args.sigma, view=args.view, disable_mu = args.student_norm, topk=args.num_neighbors * args.view).cuda()

# Momentum Update
momentum_update = loss.Momentum_Update(momentum=args.momentum).cuda()

# Train Parameters
fc_layer_lr = args.emb_lr if args.emb_lr else args.lr
if args.gpu_id != -1:
    embedding_param = list(model_student.model.embedding_f.parameters()) + list(model_student.model.embedding_g.parameters())
else:
    embedding_param = list(model_student.module.model.embedding_f.parameters()) + list(model_student.module.model.embedding_g.parameters())
param_groups = [
    {'params': list(set(model_student.parameters()).difference(set(embedding_param))) if args.gpu_id != -1 else
                 list(set(model_student.module.parameters()).difference(set(embedding_param)))},
    {'params': embedding_param, 'lr':fc_layer_lr, 'weight_decay': float(args.weight_decay)},
]

# Optimizer Setting
if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = float(args.weight_decay))
elif args.optimizer == 'adamp':
    from adamp import AdamP
    opt = AdamP(param_groups, lr=float(args.lr), weight_decay = float(args.weight_decay), nesterov=True)
    
if not args.fix_lr:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.nb_epochs)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = args.nb_epochs)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_recall=[0]
best_epoch = 0

iteration = 0
for epoch in range(0, args.nb_epochs):
    same_idxs = []
    
    if epoch % 1 == 0:
        balanced_sampler = sampler.NNBatchSampler(trn_dataset, model_student, dl_sampling, args.sz_batch, args.num_neighbors, True)
        dl_tr = torch.utils.data.DataLoader(trn_dataset, num_workers = args.nb_workers, pin_memory = True, batch_sampler = balanced_sampler)
    
    model_student.train()
    model_teacher.eval()
    
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model_student.model.modules() if args.gpu_id != -1 else model_student.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []

    pbar = tqdm(enumerate(dl_tr))
    for batch_idx, data in pbar:
        x, y, idx = data
        y = y.squeeze().cuda(non_blocking=True)
        idx = idx.squeeze().cuda(non_blocking=True)
        
        N = len(y)
        y = torch.cat([y]*args.view)
        idx = torch.cat([idx]*args.view)

        x = torch.cat(x, dim=0)
        x_s, x_t = x, x
            
        s_g, s_f = model_student(x_s.squeeze().cuda(non_blocking=True))
        with torch.no_grad():
            t_g = model_teacher(x_t.squeeze().cuda(non_blocking=True))
          
        all_loss = stml_criterion(s_f, s_g, t_g, idx)
        
        loss = all_loss.pop('loss')

        opt.zero_grad()
        loss.backward()

        momentum_update(model_student, model_teacher)
        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Momentum: {:.6f} / Loss: {:.6f} / RC Loss: {:.6f} / KL Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr), momentum_update.momentum,
                loss.item(), all_loss.pop('RC').item(), all_loss.pop('KL').item()))
        iteration += 1
    
    scheduler.step()
    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({'loss': losses_list[-1]}, step=epoch)
    wandb.log({'learning_rate': opt.param_groups[0]["lr"]}, step=epoch)
    
    if(epoch >= 0):
        with torch.no_grad():
            print("\n**Evaluating...**")
            if args.dataset != 'SOP':
                k_list = [1,2,4,8]
                Recalls = utils.evaluate_euclid(model_student, dl_ev, k_list)

            else:
                k_list = [1,10,100,1000]
                Recalls = utils.evaluate_euclid(model_student, dl_ev, k_list)
                
        if args.dataset != 'SOP':
            for i in range(4):
                wandb.log({"R@{}".format(2**i): Recalls[i]}, step=epoch)
        else:
            for i in range(4):
                wandb.log({"R@{}".format(10**i): Recalls[i]}, step=epoch)

        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            print('Acheive best performance!!, Best Recall@1:{}'.format(best_recall[0]))
        
        if args.save:
            if not os.path.exists('{}'.format(LOG_DIR)):
                os.makedirs('{}'.format(LOG_DIR))
            torch.save({'model_state_dict':model_student.state_dict() if args.gpu_id != -1 else model_student.module.state_dict()}, 
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

              