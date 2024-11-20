"""

author: Maximilian Springenberg & Yuxin Xue
association: Fraunhofer HHI
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset(root, train_dset=True, pth_out='./', **kwargs):
    if '64' in root:
        print('attempting to load imagenet-64')
        return ImageNet64(root, train=train_dset)
    else:
        print('attempting to load imagenet-32')
        return ImageNet32(root, train=train_dset)


def scale_img(img):
    return img * 2 - 1


def scale_img_inv(img):
    return (img + 1) / 2


def load_imagenet32_pairs(root, train=True):
    """
    This method loads all images and labels into RAM. Please note that (due to overhead and saniy checks) list-access
    is faster than access of arrays/ tensors via indexing. Hence storing the dataset in a list-object is preferable.

    Args:
        root: root of the imagenet-32 dataset
        train: indicates whether to load the train-split or val-split
    Returns:
        (images, labels) a list containing all images and a list containing all labels
    """
    # selecting dataset split
    sub_dirs = ('Imagenet32_train_npz',) if train else ('Imagenet32_val_npz',)
    return load_imagenet_pairs(root, sub_dirs, size=32)


def load_imagenet64_pairs(root, train=True):
    """
    This method loads all images and labels into RAM. Please note that (due to overhead and saniy checks) list-access
    is faster than access of arrays/ tensors via indexing. Hence storing the dataset in a list-object is preferable.

    Args:
        root: root of the imagenet-64 dataset
        train: indicates whether to load the train-split or val-split
    Returns:
        (images, labels) a list containing all images and a list containing all labels
    """
    # selecting dataset split
    sub_dirs = ('Imagenet64_train_part1_npz', 'Imagenet64_train_part2_npz') if train else ('Imagenet64_val_npz',)
    return load_imagenet_pairs(root, sub_dirs, size=64)


def load_imagenet_pairs(root, sub_dirs, size):
    # sanity check for completeness now rather than when it's to late ;)
    assert all([sd in os.listdir(root) for sd in sub_dirs]), f'the root should contain all sub-directories: {sub_dirs}'
    # searching for files
    files = []
    for root in [os.path.join(root, d) for d in sub_dirs]:
        for r, _, fs in os.walk(root):
            for f in fs:
                files.append(os.path.join(r, f))
    # reserving data
    images = list()
    labels = list()
    # loading data
    c = 0
    for f in files:
        # load npz
        batch = np.load(f)
        # decompress to npy
        b_data = list([e.reshape(3, size, size) for e in batch['data']])
        b_labels = list([e for e in batch['labels']])
        # set up data-range
        next_c = c + len(b_data)
        # add to storage
        images += b_data
        labels += b_labels
        # increase counter
        c = next_c
    return images, labels


class ImageNet(Dataset):
    """ImageNet-64 adapter"""

    def __init__(self, root, size, train=True, verbose=True, class_label_list='in_classes.txt', classes=None,
                 **kwargs):
        assert size in [64, 32], 'currently only ImageNet-64 & ImageNet-32 are supported'
        super().__init__()
        load_fn = load_imagenet64_pairs if size == 64 else load_imagenet32_pairs
        if verbose:
            print(f'[DATASET I/O] loading ImageNet-{size} npz files', flush=True)
        self.data, self.labels = load_fn(root, train=train)
        if verbose:
            print(f'[DATASET I/O] finished ImageNet-{size} npz files', flush=True)
        self.__N = len(self.data)
        # check if there is some documentation of class-names
        try:
            with open(class_label_list, 'r') as f:
                self.classes = [l.strip() for l in f.readlines()][:10]
        # if not just enumerate
        except:
            self.classes = [str(i) for i in range(len(self.labels))]
        # legacy from openai repo, used for cond. inputs
        self.local_classes = None if classes else self.labels

    def __len__(self):
        return self.__N

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        x = torch.from_numpy(x.astype(np.float32)) / 255.
        y = int(y) - 1  # NOTE: ImageNet64 original files DON'T have zero-based numbering
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = y
        return scale_img(x), out_dict


class ImageNet32(ImageNet):
    """ImageNet-64 adapter"""

    def __init__(self, root, train=True, verbose=True, **kwargs):
        super().__init__(root=root, train=train, verbose=verbose, size=32, **kwargs)


class ImageNet64(ImageNet):
    """ImageNet-64 adapter"""

    def __init__(self, root, train=True, verbose=True, **kwargs):
        super().__init__(root=root, train=train, verbose=verbose, size=64, **kwargs)
