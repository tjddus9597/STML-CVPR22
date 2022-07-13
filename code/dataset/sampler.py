import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder
from tqdm import *
from scipy import linalg
# import faiss

class ClassBalancedBatchSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, data_source, batch_size, images_per_class=3, ignore_index=None):
        self.targets = data_source.ys
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.ignore_index = ignore_index
        self.reverse_index, self.ignored = self._build_reverse_index()

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample_batch()

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target == self.ignore_index:
                ignored.append(i)
                continue
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored

    def sample_batch(self):
        # Real batch size is self.images_per_class * (self.batch_size // self.images_per_class)
        num_classes = self.batch_size // self.images_per_class
        sampled_classes = np.random.choice(list(self.reverse_index.keys()), num_classes, replace=False)

        sampled_indices = []
        for cls in sampled_classes:
            # Need replace = True for datasets with non-uniform distribution of images per class
            sampled_indices.extend(np.random.choice(self.reverse_index[cls],
                                                    self.images_per_class,
                                                    replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.batch_size
    
class NNBatchSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, data_source, model, seen_dataloader, batch_size, nn_per_image = 5, using_feat = True, is_norm = False):
        self.batch_size = batch_size
        self.nn_per_image = nn_per_image
        self.using_feat = using_feat
        self.is_norm = is_norm
        self.num_samples = data_source.__len__()
        self.nn_matrix, self.dist_matrix = self._build_nn_matrix(model, seen_dataloader)

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample_batch()
            
    def _predict_batchwise(self, model, seen_dataloader):
        device = "cuda"
        model_is_training = model.training
        model.eval()

        ds = seen_dataloader.dataset
        A = [[] for i in range(len(ds[0]))]
        with torch.no_grad():
            # extract batches (A becomes list of samples)
            for batch in tqdm(seen_dataloader):
                for i, J in enumerate(batch):
                    # i = 0: sz_batch * images
                    # i = 1: sz_batch * labels
                    # i = 2: sz_batch * indices
                    if i == 0:
                        # move images to device of model (approximate device)
                        if self.using_feat:
                            J, _ = model(J.cuda())
                        else:
                            _, J = model(J.cuda())
                            
                        if self.is_norm:
                            J = F.normalize(J, p=2, dim=1)
                            
                    for j in J:
                        A[i].append(j)
                        
        model.train()
        model.train(model_is_training) # revert to previous training state

        return [torch.stack(A[i]) for i in range(len(A))]
    
    def _build_nn_matrix(self, model, seen_dataloader):
        # calculate embeddings with model and get targets
        X, T, _ = self._predict_batchwise(model, seen_dataloader)
        
        # get predictions by assigning nearest 8 neighbors with cosine
        K = self.nn_per_image * 1
        nn_matrix = []
        dist_matrix = []
        xs = []
        
        for x in X:
            if len(xs)<5000:
                xs.append(x)
            else:
                xs.append(x)            
                xs = torch.stack(xs,dim=0)

                dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
                dist_emb = X.pow(2).sum(1) + dist_emb.t()

                ind = dist_emb.topk(K, largest = False)[1].long().cpu()
                dist = dist_emb.topk(K, largest = False)[0]
                nn_matrix.append(ind)
                dist_matrix.append(dist.cpu())
                xs = []
                del ind

        # Last Loop
        xs = torch.stack(xs,dim=0)
        dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
        dist_emb = X.pow(2).sum(1) + dist_emb.t()
        ind = dist_emb.topk(K, largest = False)[1]
        dist = dist_emb.topk(K, largest = False)[0]
        nn_matrix.append(ind.long().cpu())
        dist_matrix.append(dist.cpu())
        nn_matrix = torch.cat(nn_matrix, dim=0)
        dist_matrix = torch.cat(dist_matrix, dim=0)
        
        return nn_matrix, dist_matrix


    def sample_batch(self):
        num_image = self.batch_size // self.nn_per_image
        sampled_queries = np.random.choice(self.num_samples, num_image, replace=False)
        sampled_indices = self.nn_matrix[sampled_queries].view(-1)

        return sampled_indices

    def __len__(self):
        return self.num_samples // self.batch_size
    