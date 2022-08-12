import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
import torch.utils.model_zoo as model_zoo
    
def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output

class Resnet18(nn.Module):
    def __init__(self, embedding_size, bg_embedding_size = 512, pretrained = True, is_norm=True, is_student = True, bn_freeze = True):
        super(Resnet18, self).__init__()

        self.model = resnet18(pretrained)
        self.is_norm = is_norm
        self.is_student = is_student
        self.embedding_size = embedding_size
        self.bg_embedding_size = bg_embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding_g = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        nn.init.orthogonal_(self.model.embedding_g.weight)
        nn.init.constant_(self.model.embedding_g.bias, 0)
        
        if is_student:
            self.model.embedding_f = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.orthogonal_(self.model.embedding_f.weight)
            nn.init.constant_(self.model.embedding_f.bias, 0)

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)
        x = max_x + avg_x
        feat = x.view(x.size(0), -1)
        
        if self.is_student:
            x_f = self.model.embedding_f(feat)
            if self.is_norm:
                x_f = l2_norm(x_f)
            
        x_g = self.model.embedding_g(feat)
        
        if self.is_student:
            return x_g, x_f
        else:
            return x_g
                
class Resnet34(nn.Module):
    def __init__(self, embedding_size, bg_embedding_size = 512, pretrained = True, is_norm=True, is_student = True, bn_freeze = True):
        super(Resnet34, self).__init__()

        self.model = resnet34(pretrained)
        self.is_norm = is_norm
        self.is_student = is_student
        self.embedding_size = embedding_size
        self.bg_embedding_size = bg_embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding_g = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        nn.init.orthogonal_(self.model.embedding_g.weight)
        nn.init.constant_(self.model.embedding_g.bias, 0)
        
        if is_student:
            self.model.embedding_f = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.orthogonal_(self.model.embedding_f.weight)
            nn.init.constant_(self.model.embedding_f.bias, 0)

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        feat = x.view(x.size(0), -1)
        
        if self.is_student:
            x_f = self.model.embedding_f(feat)
            if self.is_norm:
                x_f = l2_norm(x_f)
            
        x_g = self.model.embedding_g(feat)
        
        if self.is_student:
            return x_g, x_f
        else:
            return x_g

class Resnet50(nn.Module):
    def __init__(self, embedding_size, bg_embedding_size = 2048, pretrained = True, is_norm=True, is_student = True, bn_freeze = True, swav_pretrained = False):
        super(Resnet50, self).__init__()

        if swav_pretrained:
            self.model = torch.hub.load('facebookresearch/swav', 'resnet50')
        else:
            self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.is_student = is_student
        self.embedding_size = embedding_size
        self.bg_embedding_size = bg_embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding_g = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        nn.init.orthogonal_(self.model.embedding_g.weight)
        nn.init.constant_(self.model.embedding_g.bias, 0)
        
        if is_student:
            self.model.embedding_f = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.orthogonal_(self.model.embedding_f.weight)
            nn.init.constant_(self.model.embedding_f.bias, 0)

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    
    def _initialize_weights(self, is_student = True):
        if is_student:
            nn.init.orthogonal_(self.model.embedding_f.weight)
            nn.init.constant_(self.model.embedding_f.bias, 0)
        
        nn.init.orthogonal_(self.model.embedding_g.weight)
        nn.init.constant_(self.model.embedding_g.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        feat = x.view(x.size(0), -1)
        
        if self.is_student:
            x_f = self.model.embedding_f(feat)
            if self.is_norm:
                x_f = l2_norm(x_f)
            
        x_g = self.model.embedding_g(feat)
        
        if self.is_student:
            return x_g, x_f
        else:
            return x_g