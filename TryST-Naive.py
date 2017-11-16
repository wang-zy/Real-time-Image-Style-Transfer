from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data.dataset import Dataset


def LoadImage(fname):
    # load image and convert to tensor wrapped in a variable
    # transforms.Normalize((0.485, 0.456, 0.40), (0.229, 0.224, 0.225))
    loader = transforms.Compose([transforms.Scale((128, 128)), transforms.ToTensor()])
    image = Image.open(fname)
    data = loader(image)
    data = Variable(data, requires_grad=False)
    data = data.unsqueeze(0)
    return data


def ViewImage(Data):
    # convert torch tensor back into image and view it
    data = torch.squeeze(Data.data)
    f = transforms.ToPILImage()
    img = f(data)
    img.show()


def SaveImage(Data):
    # convert torch tensor back into image and view it
    data = torch.squeeze(Data.data)
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
    global loss_fn
    global Style_Gram

    Loss = Variable(torch.zeros(1), requires_grad=False)
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
for paras in vgg.parameters():
    paras.requires_grad = False

# x = Variable(torch.randn(1, 3, 128, 128), requires_grad=True)
# load content image and style image
style_data = LoadImage('van.jpg')
Style_Gram = StyleGram(style_data)
loss_fn = nn.MSELoss()
content_data = LoadImage('content.jpg')
x = Variable(content_data.data.clone(), requires_grad=True)

optimizer = torch.optim.LBFGS([x])


for i in range(1):
    def closure():
        optimizer.zero_grad()
        loss, CL, SL = TotalLoss(x, content_data)
        print(i, 'loss:', loss.data[0], CL.data[0], SL.data[0])
        loss.backward()
        return loss
    optimizer.step(closure)

SaveImage(x)
ViewImage(content_data)
