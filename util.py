import os
import torch
import math
from torchvision import transforms
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.utils import make_grid
import cv2

IMG_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG',
                  'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP']


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
