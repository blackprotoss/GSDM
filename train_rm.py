import os
import torch
import random
from model.model import DDPM, SPM
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
    with open(args.config, 'r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)  # 读取yaml文件
    opt["phase"] = args.phase
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False
    return opt


def main():
    set_seed(1234)
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/train_rm.yaml',
                        help='yaml file for configuration')
    parser.add_argument( '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    # parse configs
    args = parser.parse_args()
    opt = load_yaml(args)

    # Set log output
    logger = logging.getLogger('RM')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # construct train_dataloader
    train_set = util.rm_train_dataset(corrupted_dir=opt["train_dataset"]["corrupted_dir"],
                                gt_dir=opt["train_dataset"]["gt_dir"],
                          data_shape=opt["train_dataset"]["resolution"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt["train_dataset"]["batch_size"],
                                             shuffle=opt["train_dataset"]["use_shuffle"],
                                             num_workers=opt["train_dataset"]["num_workers"], pin_memory=True)

    logs_path = os.path.join(opt["path"]["log"], opt["train_dataset"]["dataset_name"], "rm_examples")
    ckpt_path = os.path.join(opt["path"]["log"], opt["train_dataset"]["dataset_name"], "rm_checkpoints")
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    logger.info('Initial Dataset Finished')

    # model

    my_model = DDPM(opt)
    my_model.load_network(logger=logger)
    logger.info('Loading RM_model Finished')

    spm = SPM(in_channels=3, cum=opt["SPM_cum"])
    state_dict = torch.load(opt["path"]["SPM_pretrain"])
    spm.load_state_dict(state_dict)
    device = torch.device('cuda')
    spm = spm.to(device)
    spm.eval()
    logger.info('Loading pretrain SPM_model Finished')

    current_step = getattr(my_model,"begin_step",0)
    current_epoch = getattr(my_model,"begin_epoch",0)
    n_iter = opt['train']['n_iter']

    # start training
    while current_step < n_iter:
        for train_data in tqdm(train_loader, desc=f"epoch:{current_epoch}, starting_step: {current_step}"):
            current_step += 1
            if current_step > n_iter:
                break
            # get predict_structure_map
            pre_structure =spm(((train_data["corrupted"]+1)/2).to(device))  # scale to [0,1] for spm
            train_data["structure"] = pre_structure.repeat_interleave(3, dim=1) # To accommodate channels
            my_model.feed_data(train_data)
            my_model.optimize_parameters()
            # log
            if current_step % opt['train']['print_freq'] == 0:
                logs = my_model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)
            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                my_model.save_network(save_dir=ckpt_path,epoch=current_epoch,iter_step= current_step)
            if current_step % opt['train']['val_freq'] == 0:
                logger.info("Generating images...")
                my_model.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                loader = util._loader_subset(train_loader, opt["train"]["n_validation_images"], randomize=False)
                list_save_img = []
                for val_batch in loader:
                    img_corrupted = val_batch["corrupted"]
                    pre_structure = spm(((img_corrupted+1)/2).to(device))  # scale to [0,1] for spm
                    pre_structure = pre_structure.repeat_interleave(3, dim=1).cpu()
                    img_gt = val_batch["gt"].squeeze()
                    input_data = torch.cat((img_corrupted, pre_structure), dim=1)
                    with torch.no_grad():
                        my_model.feed_data(input_data)
                        prediction = my_model.test()
                        list_save_img.append(torch.cat((img_corrupted.squeeze(), pre_structure.squeeze(),
                                                        prediction.cpu().squeeze(), img_gt), dim=2))
                final_img = torch.cat(list_save_img,dim=1)
                final_img = util.tensor2img(final_img)
                util.save_img(final_img, os.path.join(logs_path, f"step_{str(current_step).zfill(6)}.png"))
                my_model.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')
                logger.info(f'f"step_{current_step}.png generated successfully')
        current_epoch += 1


if __name__ == '__main__':
    main()

