from .base import *

class SOP(BaseDataset):
    def __init__(self, root, mode, transform = None, k_fold_eval = False, fold_idx=0):
        self.root = root + '/Stanford_Online_Products'
        self.mode = mode
        self.transform = transform
        
        if k_fold_eval:
            num_class_per_fold = 11318 // 4
            val_class_split = range(num_class_per_fold * fold_idx, num_class_per_fold * (fold_idx+1))
            if self.mode == 'train':
                self.classes = [x for x in range(0,11318) if (x not in val_class_split)]
            elif self.mode == 'val':
                self.classes = val_class_split
            elif self.mode =='eval':
                self.classes = range(11318,22634)
        else:        
            if self.mode == 'train':
                self.classes = range(0,11318)
            elif self.mode == 'eval':
                self.classes = range(11318,22634)
            
        BaseDataset.__init__(self, self.root, self.mode, self.transform, k_fold_eval, fold_idx)
        metadata = open(os.path.join(self.root, 'Ebay_train.txt' if self.classes == range(0, 11318) else 'Ebay_test.txt'))
        for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                if int(class_id)-1 in self.classes:
                    self.ys += [int(class_id)-1]
                    self.I += [int(image_id)-1]
                    self.im_paths.append(os.path.join(self.root, path))