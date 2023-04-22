import torch
import torch.nn as nn
from random import sample
import copy
import torch.nn.functional as F

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)
            
def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer

def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, model, dim=128, mem_bank_size=9600, m=0.999, T=0.07, mlp=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.m = m
        self.mem_bank_size = mem_bank_size
        self.T = T
        
        self.splits_for_bn = 4
        if self.splits_for_bn > 1:
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Get current bn layer
                    bn = get_layer(model, name)
                    # Create new split_bn layer
                    split_bn = SplitBatchNorm(bn.num_features, self.splits_for_bn)
                    # Assign split_bn
                    set_layer(model, name, split_bn)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = model
        self.encoder_k = copy.deepcopy(self.encoder_q)
        
        num_ftrs = self.encoder_q.num_ftrs
        if mlp:  # hack: brute-force replacement
            self.encoder_q.model.embedding_f = nn.Sequential(nn.Linear(num_ftrs, num_ftrs), nn.ReLU(), nn.Linear(num_ftrs, dim))
            self.encoder_k.model.embedding_f = nn.Sequential(nn.Linear(num_ftrs, num_ftrs), nn.ReLU(), nn.Linear(num_ftrs, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, mem_bank_size))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss().cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        state_dict_s = self.encoder_q.state_dict()
        state_dict_t = self.encoder_k.state_dict()
        for (k_s, v_s), (k_t, v_t) in zip(state_dict_s.items(), state_dict_t.items()):
            if 'num_batches_tracked' in k_s:
                v_t.copy_(v_s)
            else:
                v_t.copy_(v_t * self.m + (1. - self.m) * v_s)
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        
        assert self.mem_bank_size % batch_size == 0 

        # replace the targets at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.mem_bank_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        _, q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            shuffle_ids, reverse_ids = get_shuffle_ids(im_k.shape[0])
            im_k = im_k[shuffle_ids]

            _, k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = k[reverse_ids].detach()

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.criterion(logits, labels)
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return loss

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds