import os
import torch
import math
from torchvision import transforms
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms.functional as tf
import cv2
from typing import (
    Sequence,
    TypeVar
)
IMG_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG',
                  'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP']
T_co = TypeVar('T_co', covariant=True)

def smooth(data, sm=10):
    smooth_data = []
    for i in range(data.shape[0]):
        if i == 0:
            smooth_data.append(data[i])
        elif i+1 <= sm:
            smooth_data.append(data[:i].mean())
        else:
            smooth_data.append(data[i-sm:i].mean())
    return smooth_data

class spm_train_dataset(Dataset):
    def __init__(
        self,
        input_dir,
        gt_dir,
        data_shape,
        color_mode="RGB"
    ):
        super().__init__()
        self.color_mode = color_mode
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.img_list = self.img_list = [file_name for file_name in os.listdir(self.input_dir) if file_name.split('.')[-1] in IMG_EXTENSIONS]
        self.resolution = data_shape
        self.img_trans = transforms.Compose([transforms.Resize(self.resolution),
                                         transforms.ToTensor()
                                         ])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_input = Image.open(os.path.join(self.input_dir, self.img_list[idx])).convert(self.color_mode)
        gt = Image.open(os.path.join(self.gt_dir, self.img_list[idx])).convert('L')
        return self.img_trans(img_input), self.img_trans(gt)

class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def _loader_subset(loader: DataLoader, num_images: int, randomize=False) -> DataLoader:
    my_dataset = loader.dataset
    lng = len(my_dataset)
    fixed_indices = range(0, lng - lng % num_images, lng // num_images)
    if randomize:
        overlap = True
        fixed_indices_set = set(fixed_indices)
        maxatt = 5
        cnt = 0
        while overlap and cnt < maxatt:
            indices = [random.randint(0, lng - 1) for _ in range(0, num_images)]
            overlap = len(set(indices).intersection(fixed_indices_set)) > 0
            cnt += 1
    else:
        indices = fixed_indices
    return DataLoader(
        Subset(my_dataset, indices),
        batch_size=1,
        shuffle=False
    )



def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

class val_dataset(Dataset):
    # Define the val dataset
    def __init__(
        self,
        input_dir,
        data_shape
    ):
        super().__init__()
        self.input_dir = input_dir
        self.img_list = [file_name for file_name in os.listdir(self.input_dir) if file_name.split('.')[-1] in IMG_EXTENSIONS]
        self.resolution = data_shape
        self.img_name = ""
        self.img_trans = transforms.Compose([transforms.Resize(self.resolution),
                                         transforms.ToTensor(),
                                         # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
                                         ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # If there is a problem with an image in the data set, read the next one
        try:
            self.img_name = self.img_list[idx]
            img_input = Image.open(os.path.join(self.input_dir, self.img_name)).convert('RGB')
        except:
            self.img_name = self.img_list[idx+1]
            img_input = Image.open(os.path.join(self.input_dir, self.img_name)).convert('RGB')
        return {"image": self.img_trans(img_input), "name": self.img_name}

class rm_train_dataset(Dataset):
    def __init__(
        self,
        corrupted_dir,
        gt_dir,
        data_shape,
        train = True
    ):
        super().__init__()
        self.train = train
        self.corrupted_dir = corrupted_dir
        self.gt_dir = gt_dir
        self.img_list = [file_name for file_name in os.listdir(self.corrupted_dir) if file_name.split('.')[-1] in IMG_EXTENSIONS]
        self.resolution = data_shape
        self.img_name = ""
        self.img_trans = transforms.Compose([transforms.Resize(self.resolution),
                                         transforms.ToTensor(),
                                         # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
                                         ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # If there is a problem with an image in the data set, read the next one
        try:
            self.img_name = self.img_list[idx]
            img_corrupted = Image.open(os.path.join(self.corrupted_dir, self.img_name)).convert('RGB')
            img_gt = Image.open(os.path.join(self.gt_dir, self.img_name)).convert('RGB')
        except:
            self.img_name = self.img_list[idx+1]
            img_corrupted = Image.open(os.path.join(self.corrupted_dir, self.img_name)).convert('RGB')
            img_gt = Image.open(os.path.join(self.gt_dir, self.img_name)).convert('RGB')
        # scale to [-1,1]
        img_corrupted = self.img_trans(img_corrupted)*2-1
        img_gt = self.img_trans(img_gt)*2-1
        if self.train:
            if torch.rand(1) < 0.5:
                img_corrupted = tf.hflip(img_corrupted)
                img_gt = tf.hflip(img_gt)
            if torch.rand(1) < 0.5:
                img_corrupted = tf.vflip(img_corrupted)
                img_gt = tf.vflip(img_gt)

        return {"corrupted": img_corrupted, "gt": img_gt}

def save_sp(input:torch._tensor, save_dir:str):
    toPIL = transforms.ToPILImage()
    input = input/2+1
    return toPIL(input.detach().cpu().squeeze()).save(save_dir)

def gray2bgr(input:torch._tensor):
    return torch.cat((input, input, input), dim=1)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img



# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14

def transform_augment(img_list, split='val', min_max=(0, 1)):
    totensor = transforms.ToTensor()
    hflip = transforms.RandomHorizontalFlip()
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img
