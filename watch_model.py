import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from tensorboardX import SummaryWriter
writer = SummaryWriter('log') #建立一个保存数据用的东西

from models.two_d.unet import Unet
x=torch.rand(16, 1, 128, 128)
model = Unet(in_channels=1, classes=1)

with SummaryWriter(comment='Unet') as w:
    w.add_graph(model, (x,))