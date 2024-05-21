"""
作者:Mr.Zhu
日期:2023//03//23
"""
from torch import nn
import torch
from torchvision.models import vgg19
from crnn.models import crnn
from torchvision.transforms import Resize

class Vgg19(torch.nn.Module):
    def __init__(self):

        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained=True).features)
        self.features = torch.nn.ModuleList(features).eval()

    def forward(self, x):

        results = []
        for ii, model in enumerate(self.features):
            x = model(x)

            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results


def build_vgg_loss(x_o,x_t):
    loss_semantic = []
    for i, f in enumerate(x_o):
        loss_semantic.append(build_l1_loss(f, x_t[i]))
    return sum(loss_semantic)

class WeightedBCELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super(WeightedBCELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, out, target, weights=None):
        out = out.clamp(self.epsilon, 1 - self.epsilon)
        if weights is not None:
            assert len(weights) == 2
            loss = weights[0] * (target * torch.log(out)) + weights[1] * ((1 - target) * torch.log(1 - out))
        else:
            loss = 2 * target * torch.log(out) +  (1 - target) * torch.log(1 - out)
        return torch.neg(torch.mean(loss))

def build_l1_loss(x_t, x_o):
    return torch.mean(torch.abs(x_t - x_o))


def build_bce_loss(x_t, x_o, weights=False, punish=4):
    if weights:  # 文本区域是否加权重为2
        lamda_1 = x_t*punish+1
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lamda_1 = torch.ones(size=x_t.shape,dtype=torch.float32).to(device)
    l_mean = torch.nn.BCELoss(weight=lamda_1, reduction="mean")
    loss = l_mean(x_o, x_t)
    return loss

def build_gram_matrix(x):

    x_shape = x.shape
    c, h, w = x_shape[1], x_shape[2], x_shape[3]
    matrix = x.view((-1,c, h * w))
    matrix1 = torch.transpose(matrix, 1, 2)
    gram = torch.matmul(matrix, matrix1) / (h * w * c)
    return gram

def build_style_loss(x_o,x_t):
    x_o = build_gram_matrix(x_o)
    x_t = build_gram_matrix(x_t)
    return build_l1_loss(x_o, x_t)

class crnn_loss():
    def __init__(self):
        self.model_path = 'crnn/data/crnn.pth'
        self.model = crnn.CRNN(32, 1, 37, 256, asloss=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print('loading CRNN pretrained model from %s' % self.model_path)
        self.model.load_state_dict(torch.load(self.model_path))
        self.resize = Resize([32, 100])
        self.model.eval()
    def build_loss(self, x_o, x_t):
        x_o = self.resize(x_o)
        x_t = self.resize(x_t)
        pre_o, conv_o = self.model(x_o)
        pre_t, conv_t = self.model(x_t)
        loss_perception = build_l1_loss(conv_t, conv_o)
        return loss_perception





