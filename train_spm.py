import torch
from torch.utils import data
import os
from model.loss import build_l1_loss, build_style_loss, WeightedBCELoss, crnn_loss
import numpy as np
from util import spm_train_dataset, smooth
from tqdm import tqdm
from matplotlib import pyplot as plt
from model.model import SPM
import argparse
import yaml


def load_yaml(args):
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # 读取yaml文件
    return cfg

def main():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/train_spm.yaml',
                        help='yaml file for configuration')
    args = parser.parse_args()
    cfg = load_yaml(args)

    # Prepare loss weights
    omega_rec = cfg["omega_rec"]
    omega_sem = cfg["omega_sem"]
    theta_l1 = 1
    theta_bce = 1


    # loading data
    spm_data = spm_train_dataset(
        input_dir=cfg["train_dataset"]["corrupted_dir"],
        gt_dir=cfg["train_dataset"]["gt_dir"],
        data_shape=cfg["train_dataset"]["resolution"],
        color_mode="RGB"
    )
    data_loader = data.DataLoader(dataset=spm_data, pin_memory=True,
                                  batch_size=cfg["train_dataset"]["batch_size"],
                                  num_workers=cfg["train_dataset"]["num_workers"],
                                  shuffle=cfg["train_dataset"]["use_shuffle"])

    # load model
    model = SPM(in_channels=3, cum=cfg["cum"])
    device = torch.device('cuda')
    model = model.to(device)
    now_epoch = 0
    if cfg["resume_state"]:
        state_dict = torch.load(cfg["resume_state"])
        model.load_state_dict(state_dict)
        now_epoch = int(cfg["resume_state"].split('.')[0].split('_')[-1])
    model.train()

    # prepare each loss
    crnn = crnn_loss()
    bce = WeightedBCELoss().to(device)

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["spm_learning_rate"],
                                 betas=(cfg["beta1"], cfg["beta2"]))
    iteration = len(data_loader)

    # prepare paths
    checkpoint_savedir = os.path.join(cfg["log"],cfg["train_dataset"]["dataset_name"],"spm_checkpoints")
    lossfig_path = os.path.join(cfg["log"],cfg["train_dataset"]["dataset_name"],"spm_lossfig")
    os.makedirs(checkpoint_savedir, exist_ok=True)
    os.makedirs(lossfig_path, exist_ok=True)

    while now_epoch < cfg["epoch"]:
        loss_list = []
        print('\nEpoch: %d' % (now_epoch + 1))
        step = 0
        for data_batch in tqdm(data_loader, desc=f"epoch{now_epoch + 1}_training"):
            corrupted_batch, gt_batch = data_batch
            corrupted_batch, gt_batch = corrupted_batch.to(device), gt_batch.to(device)
            optimizer.zero_grad()
            outputs = model(corrupted_batch)
            outputs = (outputs + 1.0) / 2

            # Construct loss function
            loss_l1 = build_l1_loss(x_t=gt_batch, x_o=outputs)
            loss_bce = bce(out=outputs, target=gt_batch)
            loss_style = build_style_loss(x_t=gt_batch, x_o=outputs)
            loss_rec = theta_l1 * loss_l1 + theta_bce * loss_bce
            loss_crnn = crnn.build_loss(x_o=outputs, x_t=gt_batch)


            loss_sem = loss_style + loss_crnn
            loss = omega_rec * loss_rec + omega_sem * loss_sem
            loss_list.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            if step % cfg["print_freq"] == 0:
                print(f"loss_rec:{round(loss_rec.detach().cpu().item(), 5)}")
                print(f"loss_sem:{round(loss_sem.detach().cpu().item(), 5)}")
            step += 1
        torch.save(model.state_dict(), checkpoint_savedir + f"/spm_{now_epoch}.pt")
        fig = plt.figure(figsize=(10, 6))
        x = np.array(range(iteration))
        loss_all = np.array(loss_list)
        loss_all_ = smooth(loss_all, 50)  # Window moving average
        loss_all_ = np.array(loss_all_)
        plt.plot(x, loss_all, x, loss_all_)
        plt.ylabel("the value of loss", fontsize=14)
        plt.ylim(0, 10)
        plt.xlabel("Iteration", fontsize=14)
        plt.savefig(lossfig_path + f"/loss_fig_{now_epoch}.png")
        plt.ion()
        plt.pause(15)
        plt.close()
        now_epoch += 1


if __name__ == '__main__':
    main()
