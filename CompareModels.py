import torch
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data.dataset import Dataset


def normalize(x):
    y = x.div(255)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    y[:, 0, :, :] = (y[:, 0, :, :] - mean[0]) / std[0]
    y[:, 1, :, :] = (y[:, 1, :, :] - mean[1]) / std[1]
    y[:, 2, :, :] = (y[:, 2, :, :] - mean[2]) / std[2]
    return y


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        return self.conv2d(self.reflection_pad(x))


class Res(nn.Module):
    def __init__(self, numChannels):
        super(Res, self).__init__()
        bn_flag = True
        self.conv1 = Conv(numChannels, numChannels, 3, stride=1)
        self.in1 = nn.InstanceNorm2d(numChannels, affine=bn_flag)
        self.relu = nn.ReLU()
        self.conv2 = Conv(numChannels, numChannels, 3, stride=1)
        self.in2 = nn.InstanceNorm2d(numChannels, affine=bn_flag)
    def forward(self, x):
        residual = x
        output = self.in2(self.conv2(self.relu(self.in1(self.conv1(x)))))
        return residual + output


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(DeConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample)
        self.conv = Conv(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(self.upsample(x))


class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()
        bn_flag = True
        self.relu = nn.ReLU()
        self.conv1 = Conv(3, 32, 9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=bn_flag)
        self.conv2 = Conv(32, 64, 3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=bn_flag)
        self.conv3 = Conv(64, 128, 3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=bn_flag)
        self.res1 = Res(128)
        self.res2 = Res(128)
        self.res3 = Res(128)
        self.res4 = Res(128)
        self.res5 = Res(128)
        self.deconv1 = DeConv(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=bn_flag)
        self.deconv2 = DeConv(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=bn_flag)
        self.conv4 = Conv(32, 3, 9, stride=1)

    def forward(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = self.res5(self.res4(self.res3(self.res2(self.res1(x)))))
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        return self.conv4(x)

def LoadImage(fname, scale=False):
    # load image and convert to tensor wrapped in a variable
    if scale is True:
        loader = transforms.Compose([transforms.Scale((imsize, imsize)),
                                     transforms.CenterCrop(imsize),
                                     transforms.ToTensor(),
                                     transforms.Lambda(lambda x: x.mul(255))])
    else:
        loader = transforms.Compose([transforms.Scale((imsize, imsize)),
                                     transforms.CenterCrop(imsize),
                                     transforms.ToTensor()])
    image = Image.open(fname).convert('RGB')
    data = loader(image)
    data = Variable(data.cuda(), volatile=True)
    data = data.unsqueeze(0)
    return data


def SaveImage(tensor_transformed, fname, factor=255):
    def RGB(image):
        return (image.transpose(0, 2, 3, 1)*factor).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(RGB(tensor_transformed.data.cpu().numpy())[0])
    result.save(fname)


imsize = 256

gen = glob.glob('SavedModels/*.model')
s = torch.load(gen[0])
print(gen[0])
image = LoadImage('amber.jpg', scale=True)
SaveImage(s(image), 'SavedImages/candy2.png', factor=1)

