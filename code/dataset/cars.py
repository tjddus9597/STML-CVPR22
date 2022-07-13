from .base import *
import scipy.io

class Cars(BaseDataset):
    def __init__(self, root, mode, transform = None, k_fold_eval = False, fold_idx=0):
        self.root = root + '/cars196'
        self.mode = mode
        self.transform = transform
        
        if k_fold_eval:
            num_class_per_fold = 98 // 4
            val_class_split = range(num_class_per_fold * fold_idx, num_class_per_fold * (fold_idx+1))
            if self.mode == 'train':
                self.classes = [x for x in range(0,98) if (x not in val_class_split)]
            elif self.mode == 'val':
                self.classes = val_class_split
            elif self.mode =='eval':
                self.classes = range(98, 196)
        else:
            if self.mode == 'train':
                self.classes = range(0,98)
            elif self.mode == 'eval':
                self.classes = range(98,196)
                
        BaseDataset.__init__(self, self.root, self.mode, self.transform, k_fold_eval, fold_idx)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]
        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in self.classes: # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1