import time
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from metrics import psnr, ssim

# The metrics
def get_psnr_ssim(gt_path, output_path, resize_flag=False):
    # resize_flag determine whether the image needs to be scaled
    start_time = time.process_time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ssim = ssim.SSIM().to(device)
    _psnr = psnr.PSNR(255.0).to(device)
    totensor = transforms.ToTensor()
    im_list = os.listdir(output_path)
    gt_list = os.listdir(gt_path)
    psnr_list = []
    ssim_list = []
    for path_1, path_2 in zip(gt_list, im_list):
        gt_img_path = os.path.join(gt_path, path_1)
        gt_img = Image.open(gt_img_path).convert("RGB")

        out_img_path = os.path.join(output_path, path_2)
        if resize_flag:
            out_img = Image.open(out_img_path).convert("RGB").resize((256, 64), Image.BICUBIC)
        else:
            out_img = Image.open(out_img_path).convert("RGB")
        gt_img = totensor(gt_img).unsqueeze(0).to(device)
        out_img = totensor(out_img).unsqueeze(0).to(device)
        ssim_list.append(_ssim(gt_img, out_img).cpu().item())
        gt_img = gt_img * 255.
        out_img = out_img * 255.
        psnr_list.append(_psnr(gt_img, out_img).cpu().item())

    print(f"PSNR:{round(np.mean(psnr_list), 2)}, img_number:{len(gt_list)}")
    print(f"SSIM:{round(np.mean(ssim_list), 4)}")
    end_time = time.process_time()
    print(f"process_time:{end_time - start_time}s")