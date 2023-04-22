import builtins
import os
import sys
import time
import argparse
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

class MeanShift(nn.Module):
    def __init__(self, base_encoder, dim = 512, m=0.99, mem_bank_size=9600, topk=5):
        super(MeanShift, self).__init__()

        # save parameters
        self.m = m
        self.mem_bank_size = mem_bank_size
        self.topk = topk

        # create encoders and projection layers
        # both encoders should have same arch
        self.encoder_q = base_encoder
        self.encoder_t = copy.deepcopy(self.encoder_q)
        
        # save output embedding dimensions
        # assuming that both encoders have same dim
        num_ftrs = self.encoder_t.num_ftrs
        hidden_dim = num_ftrs * 2
        proj_dim = dim

        # projection layers
        self.encoder_t.model.embedding_f = self.get_mlp(num_ftrs, hidden_dim, proj_dim)
        self.encoder_q.model.embedding_f = self.get_mlp(num_ftrs, hidden_dim, proj_dim)

        # prediction layer
        self.predict_q = self.get_mlp(proj_dim, hidden_dim, proj_dim)

        # copy query encoder weights to target encoder
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False

        print("using mem-bank size {}".format(self.mem_bank_size))
        # setup queue (For Storing Random Targets)
        self.register_buffer('queue', torch.randn(self.mem_bank_size, dim))
        # normalize the queue embeddings
        self.queue = F.normalize(self.queue, dim=1)
        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        state_dict_q = self.encoder_q.state_dict()
        state_dict_t = self.encoder_t.state_dict()
        for (k_q, v_q), (k_t, v_t) in zip(state_dict_q.items(), state_dict_t.items()):
            if 'num_batches_tracked' in k_q:
                v_t.copy_(v_t)
            else:
                v_t.copy_(v_t * self.m + (1. - self.m) * v_q)

    @torch.no_grad()
    def data_parallel(self):
        self.encoder_q = torch.nn.DataParallel(self.encoder_q)
        self.encoder_t = torch.nn.DataParallel(self.encoder_t)
        self.predict_q = torch.nn.DataParallel(self.predict_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, targets):
        batch_size = targets.shape[0]

        ptr = int(self.queue_ptr)
        
        assert self.mem_bank_size % batch_size == 0 

        # replace the targets at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = targets
        ptr = (ptr + batch_size) % self.mem_bank_size  # move pointer

        self.queue_ptr[0] = ptr
        
    def get_mlp(self, inp_dim, hidden_dim, out_dim):
        mlp = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        return mlp

    def forward(self, im_q, im_t):
        # compute query features
        _, query = self.encoder_q(im_q)
        
        # compute predictions for instance level regression loss
        query = self.predict_q(query)
        query = F.normalize(query, dim=1)

        # compute target features
        with torch.no_grad():
            # update the target encoder
            self._momentum_update_target_encoder()

            # shuffle targets
            shuffle_ids, reverse_ids = get_shuffle_ids(im_t.shape[0])
            im_t = im_t[shuffle_ids]

            # forward through the target encoder
            _, current_target = self.encoder_t(im_t)
            current_target = F.normalize(current_target, dim=1)

            # undo shuffle
            current_target = current_target[reverse_ids].detach()
            self._dequeue_and_enqueue(current_target)

        # calculate mean shift regression loss
        targets = self.queue.clone().detach()
        # calculate distances between vectors
        dist_t = 2 - 2 * torch.einsum('bc,kc->bk', [current_target, targets])
        dist_q = 2 - 2 * torch.einsum('bc,kc->bk', [query, targets])

        # select the k nearest neighbors [with smallest distance (largest=False)] based on current target
        _, nn_index = dist_t.topk(self.topk, dim=1, largest=False)
        nn_dist_q = torch.gather(dist_q, 1, nn_index)

        loss = (nn_dist_q.sum(dim=1) / self.topk).mean()

        return loss


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds