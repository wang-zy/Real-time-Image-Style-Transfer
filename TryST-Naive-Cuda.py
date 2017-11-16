from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
from torchvision import transforms

imsize = 512

def LoadImage(fname):
    # load image and convert to tensor wrapped in a variable
    loader = transforms.Compose([transforms.Scale((imsize, imsize)), transforms.ToTensor()])
    image = Image.open(fname)
    data = loader(image)
    data = Variable(data, requires_grad=False)
    data = data.unsqueeze(0)
    return data


def SaveImage(Data):
    # convert torch tensor back into image and view it
    data = torch.squeeze(Data)
    f = transforms.ToPILImage()
    img = f(data)
    img.save('output.png')


def Gram(Fmap):
    # calculate Gram matrix from feature map
    numImages, numFeatures, W, H = Fmap.size()
    Fmap = Fmap.view(numImages * numFeatures, W * H)
    G = torch.mm(Fmap, Fmap.t())
    return G.div(numImages * numFeatures * W * H)


def StyleGram(style):
    Style_layers = [0, 2, 5, 7, 10]
    style_gram = {}
    for i in range(max(Style_layers) + 1):
        layer = vgg[i]
        style = layer(style)
        if i in Style_layers:
            style_gram[i] = Gram(style)
    return style_gram


# def normalize(x):
#     mean = [0.485, 0.456, 0.40]
#     std = [0.229, 0.224, 0.225]
#     y = x.clone()
#     y[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
#     y[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
#     y[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
#     return y


def TotalLoss(pred, content):
    global Style_Gram
    global vgg
    global loss_fn

    Loss = Variable(torch.zeros(1).cuda(), requires_grad=False)
    Style_layers = [0, 2, 5, 7, 10]
    Content_layer = 7
    alpha = 1000.0
    for i in range(max(max(Style_layers), Content_layer) + 1):
        layer = vgg[i]
        pred = layer(pred)
        if i == Content_layer:
            content = layer(content)
            ContentLoss = loss_fn(pred.clone(), content)
            Loss = Loss + ContentLoss
        elif i < Content_layer:
            content = layer(content)
        if i in Style_layers:
            Loss = Loss + alpha**2 * loss_fn(Gram(pred.clone()), Style_Gram[i])
    return Loss, ContentLoss, Loss - ContentLoss

# load pretrained VGG16 network
vgg = models.vgg16(True).features
vgg.cuda()
for paras in vgg.parameters():
    paras.requires_grad = False

dtype = torch.cuda.FloatTensor

target = Variable(torch.randn(1, 3, imsize, imsize).cuda(), requires_grad=True)

# load content image and style image
style_data = LoadImage('van.jpg').type(dtype)
Style_Gram = StyleGram(style_data)
content_data = LoadImage('guo.jpg').type(dtype)

loss_fn = nn.MSELoss()

optimizer = torch.optim.LBFGS([target])
nStep = 0
while nStep < 700:
    def closure():
        global nStep
        target.data.clamp_(0, 1)
        nStep += 1
        optimizer.zero_grad()
        loss, CL, SL = TotalLoss(target, content_data)
        if nStep % 50 == 0:
            print(nStep, 'Total:', loss.data[0], 'Content:', CL.data[0], 'Style:', SL.data[0])
        loss.backward()
        return loss
    optimizer.step(closure)

SaveImage(target.data.clone().cpu())
