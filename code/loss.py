import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Momentum_Update(nn.Module):
    """Log ratio loss function. """
    def __init__(self, momentum):
        super(Momentum_Update, self).__init__()
        self.momentum = momentum
        
    @torch.no_grad()
    def forward(self, model_student, model_teacher):
        """
        Momentum update of the key encoder
        """
        m = self.momentum

        state_dict_s = model_student.state_dict()
        state_dict_t = model_teacher.state_dict()
        for (k_s, v_s), (k_t, v_t) in zip(state_dict_s.items(), state_dict_t.items()):
            if 'num_batches_tracked' in k_s:
                v_t.copy_(v_s)
            else:
                v_t.copy_(v_t * m + (1. - m) * v_s)  
    
class RC_STML(nn.Module):
    def __init__(self, sigma, delta, view, disable_mu, topk):
        super(RC_STML, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        
    def k_reciprocal_neigh(self, initial_rank, i, topk):
        forward_k_neigh_index = initial_rank[i,:topk]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:topk]
        fi = np.where(backward_k_neigh_index==i)[0]
        return forward_k_neigh_index[fi]

    def forward(self, s_emb, t_emb, idx):
        if self.disable_mu:
            s_emb = F.normalize(s_emb)
        t_emb = F.normalize(t_emb)

        N = len(s_emb)        
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)
        
        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb) 
            W_P = torch.exp(-T_dist.pow(2) / self.sigma)
            
            batch_size = len(s_emb) // self.view
            W_P_copy = W_P.clone()
            W_P_copy[idx.unsqueeze(1) == idx.unsqueeze(1).t()] = 1

            topk_index = torch.topk(W_P_copy, self.topk)[1]
            topk_half_index = topk_index[:, :int(np.around(self.topk/2))]

            W_NN = torch.zeros_like(W_P).scatter_(1, topk_index, torch.ones_like(W_P))
            V = ((W_NN + W_NN.t())/2 == 1).float()

            W_C_tilda = torch.zeros_like(W_P)
            for i in range(N):
                indNonzero = torch.where(V[i, :]!=0)[0]
                W_C_tilda[i, indNonzero] = (V[:,indNonzero].sum(1) / len(indNonzero))[indNonzero]
                
            W_C_hat = W_C_tilda[topk_half_index].mean(1)
            W_C = (W_C_hat + W_C_hat.t())/2
            W = (W_P + W_C)/2

            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)
        
        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight
        
        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb)-1))
        
        return loss
    
class KL_STML(nn.Module):
    def __init__(self, disable_mu, temp=1):
        super(KL_STML, self).__init__()
        self.disable_mu = disable_mu
        self.temp = temp
    
    def kl_div(self, A, B, T = 1):
        log_q = F.log_softmax(A/T, dim=-1)
        p = F.softmax(B/T, dim=-1)
        kl_d = F.kl_div(log_q, p, reduction='sum') * T**2 / A.size(0)
        return kl_d

    def forward(self, s_f, s_g):
        if self.disable_mu:
            s_f, s_g = F.normalize(s_f), F.normalize(s_g)

        N = len(s_f)
        S_dist = torch.cdist(s_f, s_f)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)
        
        S_bg_dist = torch.cdist(s_g, s_g)
        S_bg_dist = S_bg_dist / S_bg_dist.mean(1, keepdim=True)
        
        loss = self.kl_div(-S_dist, -S_bg_dist.detach(), T=1)
        
        return loss
    
class STML_loss(nn.Module):
    def __init__(self, sigma, delta, view, disable_mu, topk):
        super(STML_loss, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        self.RC_criterion = RC_STML(sigma, delta, view, disable_mu, topk)
        self.KL_criterion = KL_STML(disable_mu, temp=1)

    def forward(self, s_f, s_g, t_g, idx):
        # Relaxed contrastive loss for STML
        loss_RC_f = self.RC_criterion(s_f, t_g, idx)
        loss_RC_g = self.RC_criterion(s_g, t_g, idx)
        loss_RC = (loss_RC_f + loss_RC_g)/2
        
        # Self-Distillation for STML
        loss_KL = self.KL_criterion(s_f, s_g)
        
        loss = loss_RC + loss_KL
        
        total_loss = dict(RC=loss_RC, KL=loss_KL, loss=loss)
        
        return total_loss
