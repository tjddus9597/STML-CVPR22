from .base import *

class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform = None, k_fold_eval = False, fold_idx=0):
        self.root = root + '/CUB_200_2011'
        self.mode = mode
        self.transform = transform
        
        if k_fold_eval:
            num_class_per_fold = 100 // 4
            val_class_split = range(num_class_per_fold * fold_idx, num_class_per_fold * (fold_idx+1))
            if self.mode == 'train':
                self.classes = [x for x in range(0,100) if (x not in val_class_split)]
            elif self.mode == 'val':
                self.classes = val_class_split
            elif self.mode =='eval':
                self.classes = range(100,200)
        else:
            if self.mode == 'train':
                self.classes = range(0,100)
            elif self.mode == 'eval':
                self.classes = range(100,200)
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform, k_fold_eval, fold_idx)
        index = 0
        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(i[0])
                index += 1