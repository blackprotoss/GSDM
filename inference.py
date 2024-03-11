import os
import torch
import random
from model.model import GSDM
import argparse
import logging
import numpy as np
from tqdm import tqdm
import util
import warnings
import yaml
warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Load yaml file
def load_yaml(args):
    gpu_ids = args.gpu_ids
    with open(args.config, 'r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)  # 读取yaml文件
    opt["phase"] = args.phase
    opt["SPM"]["resume_state"] = os.path.join(args.ckpt_dir, "spm.pt")
    opt["RM"]["path"]["resume_state"] = os.path.join(args.ckpt_dir, "rm")
    if args.input_dir is not None:
        opt["val_dataset"]["input_dir"] = args.input_dir
    if args.output_dir is not None:
        opt["val_dataset"]["output_dir"] = args.output_dir
    if gpu_ids is not None:
        opt["RM"]['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False
    opt["save_sp"] = args.save_sp
    return opt


def main():
    set_seed(1234)
    parser = argparse.ArgumentParser(description='val')
    parser.add_argument('--config', type=str, default='./config/final.yaml',
                        help='yaml file for configuration')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint')
    parser.add_argument('--input_dir', type=str, default=r'./input', help='The path of input_img')
    parser.add_argument('--output_dir', type=str, default='./output', help='The path to save img')
    parser.add_argument( '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--save_sp', default=False, help="Whether to save sp_img")

    # parse configs
    args = parser.parse_args()
    opt = load_yaml(args)

    # Set log output
    logger = logging.getLogger('GSDM')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # construct val_dataloader
    val_set = util.val_dataset(input_dir=opt["val_dataset"]["input_dir"],
                          data_shape=opt["val_dataset"]["resolution"])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    if not os.path.exists(opt["val_dataset"]["output_dir"]):
        os.makedirs(opt["val_dataset"]["output_dir"])
    logger.info('Initial Dataset Finished')

    # model
    logger.info('Initial Model Finished')
    my_model = GSDM(opt)
    my_model.load_network()
    logger.info('Loading Model Finished')

    # start text_image_inpainting
    for input_data in tqdm(val_loader):
        output = my_model.inference(input_data["image"])
        if opt["save_sp"]:
            sp = util.gray2bgr(output["SPM"])
            final = torch.cat((sp, output["RM"]), dim=2)
        else:
            final = output["RM"]
        sr_img = util.tensor2img(final)

        # input_data["name"] is a list that contains only the img name
        util.save_img(sr_img, os.path.join(opt["val_dataset"]["output_dir"], input_data["name"][0]))
        logger.info(f'{input_data["name"]} processed successfully')


if __name__ == '__main__':
    main()

    ##  Metrics are provided for evaluating the output image.
    ##  The test images have been released previously.
    # from metrics.eval import get_psnr_ssim
    # get_psnr_ssim(gt_path=r"./gt", output_path="./output")
