import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from .networks import Conv_bn_block, Res_block
logger = logging.getLogger('base')

# define structure prediction module
class SPM(torch.nn.Module):
    def __init__(self, in_channels, cum, get_feature_map=False):
        super().__init__()
        self.cnum = cum
        self.get_feature_map = get_feature_map
        self.res_block = Res_block(in_channels, self.cnum)  # low-level visual features are extracted
        self._conv1_1 = Conv_bn_block(in_channels=in_channels, out_channels=self.cnum, kernel_size=3, stride=1, padding=2, dilation=2)
        self._conv1_2 = Conv_bn_block(in_channels=self.cnum, out_channels=self.cnum, kernel_size=3, stride=1,  padding=2, dilation=2)
        self._pool1 = torch.nn.Conv2d(in_channels = self.cnum, out_channels = 2 * self.cnum, kernel_size = 3, stride = 2, padding = 1)
        # ---------------------------
        self._conv2_1 = Conv_bn_block(in_channels=2 * self.cnum, out_channels=2 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)

        self._conv2_2 = Conv_bn_block(in_channels=2 * self.cnum, out_channels=2 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)
        self._pool2 = torch.nn.Conv2d(in_channels=2 * self.cnum, out_channels=4 * self.cnum, kernel_size=3, stride=2,
                                      padding=1)
        # ---------------------------
        self._conv3_1 = Conv_bn_block(in_channels=4 * self.cnum, out_channels=4 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)

        self._conv3_2 = Conv_bn_block(in_channels=4 * self.cnum, out_channels=4 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)
        self._pool3 = torch.nn.Conv2d(in_channels=4 * self.cnum, out_channels=8 * self.cnum, kernel_size=3, stride=2,
                                      padding=1)
        # ---------------------------
        # Upsampling decoding phase
        self._conv4_1 = Conv_bn_block(in_channels=8 * self.cnum, out_channels=8 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)

        self._conv4_2 = Conv_bn_block(in_channels=8 * self.cnum, out_channels=8 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)
        self._deconv1 = torch.nn.ConvTranspose2d(8 * self.cnum, 4 * self.cnum, kernel_size=2, stride=2)
        self._bn1 = torch.nn.BatchNorm2d(4 * self.cnum)
        # ---------------------------
        self._conv5_1 = Conv_bn_block(in_channels=4 * self.cnum, out_channels=4 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)

        self._conv5_2 = Conv_bn_block(in_channels=4 * self.cnum, out_channels=4 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)
        self._deconv2 = torch.nn.ConvTranspose2d(4 * self.cnum, 2 * self.cnum, kernel_size=2, stride=2)
        self._bn2 = torch.nn.BatchNorm2d(2 * self.cnum)

        # ---------------------------
        self._conv6_1 = Conv_bn_block(in_channels=2 * self.cnum, out_channels=2 * self.cnum, kernel_size=3,
                                      stride=1,  padding=2, dilation=2)

        self._conv6_2 = Conv_bn_block(in_channels=2 * self.cnum, out_channels=2 * self.cnum, kernel_size=3, stride=1,
                                      padding=2, dilation=2)

        self._deconv3 = torch.nn.ConvTranspose2d(2 * self.cnum, self.cnum, kernel_size=2, stride=2)
        self._bn3 = torch.nn.BatchNorm2d(self.cnum)
        # ----------------
        self._conv7 = torch.nn.Conv2d(in_channels=self.cnum,out_channels=1,  kernel_size=3, stride=1,
                                      padding=1)

    def forward(self, x):
        x = self.res_block(x)
        x = self._conv1_1(x)  # [16,32,64,256]
        x = self._conv1_2(x)  # [16,32,64,256]
        f1 = x

        x = torch.nn.functional.elu(self._pool1(x), alpha=1.0)  # [16,64,32,128]
        x = self._conv2_1(x)  # [16,64,32,128]
        x = self._conv2_2(x)
        f2 = x  # [16,64,32,128]

        x = torch.nn.functional.elu(self._pool2(x), alpha=1.0)
        x = self._conv3_1(x)  # [16,128,16,64]
        x = self._conv3_2(x)

        f3 = x  # [16,128,16,64]
        x = torch.nn.functional.elu(self._pool3(x), alpha=1.0)
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        f4 = x # [16,256,8,32]

        x = self._deconv1(x)
        x = torch.add(x, f3)  # skip connection
        x = torch.nn.functional.elu(self._bn1(x), alpha=1.0)
        x = self._conv5_1(x)
        x = self._conv5_2(x)
        x = self._deconv2(x)
        x = torch.add(x, f2)  # skip connection
        x = torch.nn.functional.elu(self._bn2(x), alpha=1.0)  # [16,32,64,256]
        x = self._conv6_1(x)
        x = self._conv6_2(x)
        x = self._deconv3(x)
        x = torch.add(x, f1)  #
        x = torch.nn.functional.elu(self._bn3(x), alpha=1.0)

        x = torch.tanh(self._conv7(x)) # [batch,1,64,256]

        if self.get_feature_map:
            return x, [f1, f2, f3, f4]
        else:
            return x

    def load_network(self, opt):
        if opt["SPM"]["resume_state"] is not None:
            state_dict = torch.load(opt["SPM"]["resume_state"])
            self.load_state_dict(state_dict, strict=False)

# using DDPM as Reconstruction module
class DDPM(BaseModel):
    def __init__(self, opt):
        # opt['gpu_ids'] = gpu_ids
        # opt["distributed"] = distributed
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))#通过opt配置输入生成器并放到device上去
        self.schedule_phase = None
        self.set_T(opt['model']['diffusion']['sampling_timesteps'])
        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':#训练阶段
            self.netG.train()#在networks.py中netG是一个Gaussian diffusion模型，该模型是nn.Module的子类
            # find the parameters to optimize
            if opt['model']['finetune_norm']:#如果finetune_norm为true则优化transformer的参数
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())#优化所有参数

            self.optG = torch.optim.Adam(#使用Adam优化器
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()#用来记录训练日志
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()  #
        l_pix = self.netG(self.data)  # 扩散模型
        # need to average in multi-gpu
        b, c, h, w = self.data['gt'].shape  # 优化参数 应该是output(HR)的shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()  # 反向传播
        self.optG.step()  # 参数更新

        # set log
        self.log_dict['l_pix'] = l_pix.item()  # 更新日志

    # 逆扩散过程
    def set_T(self,step):
        self.netG.sampling_timesteps=step

   #逆扩散过程
    def test(self, continous=False):
        self.netG.eval()
        if isinstance(self.netG, nn.DataParallel):
            self.Output = self.Reconstruction_Parallel()
        else:
            self.Output = self.Reconstruction()
        return self.Output

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.Output = self.netG.module.sample(batch_size, continous)
            else:
                self.Output = self.netG.sample(batch_size, continous)
        self.netG.train()
        return self.Output


    def Reconstruction(self):
        return self.netG.super_resolution(self.data)

    # Use.module for multiple Gpus to access the actual model
    def Reconstruction_Parallel(self):
        return self.netG.module.super_resolution(self.data)

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase#更改阶段
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, save_dir,epoch, iter_step):
        gen_path = os.path.join(
            save_dir, 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            save_dir, 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self,logger):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            if self.opt['phase'] == 'train':
                # optimizer
                opt_path = '{}_opt.pth'.format(load_path)
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']


# define build model framework of GSDM
class GSDM(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Configuring SPM
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.SPM = SPM(
            in_channels=opt["SPM"]["in_channels"],
            cum=opt["SPM"]["cum"],
            get_feature_map=opt["SPM"]["feature_map"]
        ).to(self.device)


        # Configuring RM
        opt["RM"]["gpu_ids"] = opt["gpu_ids"]
        opt["RM"]['distributed'] = opt['distributed']
        self.RM = DDPM(opt["RM"])
        if opt["phase"] == 'val':
            self.RM.set_new_noise_schedule(opt["RM"]['model']['beta_schedule']['val'], schedule_phase='val')

    # prepare data for RM
    def prepare_data(self, input, sp):
        # input : [0,1] scaled to [-1,1]
        ret_x = input*2-1
        return torch.cat((ret_x, sp, sp, sp), dim=1)

    def feed_data(self,my_data):
        return my_data.to(self.device)

    def inference(self, x):
        x = self.feed_data(x)
        with torch.no_grad():
            self.SPM.eval()
            sp = self.SPM(x)
            rm_input = self.prepare_data(input=x, sp=sp)
            self.RM.feed_data(rm_input)
            return {"SPM": sp, "RM":self.RM.test()}

    def load_network(self, logger):
        # load checkpoint
        self.SPM.load_network(self.opt)
        self.RM.load_network(logger=logger)



