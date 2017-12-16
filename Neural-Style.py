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

imsize = 256
torch.backends.cudnn.benchmark=True

def LoadImage(fname):
    # load image and convert to tensor wrapped in a variable
    loader = transforms.Compose([transforms.Scale((imsize, imsize)),
                                 transforms.CenterCrop(imsize),
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: x.mul(255))])
    image = Image.open(fname).convert('RGB')
    data = loader(image)
    data = Variable(data, volatile=True)
    data = data.unsqueeze(0)
    return data


# def SaveImage(tensor_orig, tensor_transformed, filename):
#     def RGB(image):
#         return (image.transpose(0, 2, 3, 1)).clip(0, 255).astype(np.uint8)
#     result = Image.fromarray(RGB(tensor_transformed.data.cpu().numpy())[0])
#     orig = Image.fromarray(RGB(tensor_orig.data.cpu().numpy())[0])
#     new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
#     new_im.paste(orig, (0, 0))
#     new_im.paste(result, (result.size[0] + 5, 0))
#     new_im.save(filename)

def SaveImage(tensor_transformed, fname):
    def RGB(image):
        return (image.transpose(0, 2, 3, 1)).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(RGB(tensor_transformed.data.cpu().numpy())[0])
    result.save(fname)


def Gram(Fmap):
    # calculate Gram matrix from feature map
    numImages, numFeatures, W, H = Fmap.size()
    Fmap = Fmap.view(numImages, numFeatures, W * H)
    G = torch.bmm(Fmap, Fmap.transpose(1, 2))
    return G.div(numFeatures * W * H)


def StyleGram(style):
    global vgg
    style_gram = {}
    feature_style = vgg(style)
    for layer in feature_style.keys():
        style_gram[layer] = Variable(Gram(feature_style[layer]).data, requires_grad=False)
    return style_gram


class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.layers = [3, 8, 15, 22]
        self.cLayer = 8  # content layer
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = {}
        for nLayer in range(max(self.layers)+1):
            x = self.vgg[nLayer](x)
            if nLayer in self.layers:
                output[nLayer] = x
        return output


def TotalLoss(pred, content):
    global Style_Gram
    global vgg
    global loss_fn
    global contentWeight
    global styleWeight
    global tvWeight

    # Total variation regularization
    tvLoss = tvWeight * (torch.sum(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])) +
                         torch.sum(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])))

    # content loss
    feature_pred = vgg(pred)
    feature_content = vgg(content)
    f_pred = feature_pred[vgg.cLayer]
    f_content = Variable(feature_content[vgg.cLayer].data, requires_grad=False)
    n1, n2, n3, n4 = f_pred.size()
    contentLoss = contentWeight * torch.sum((f_pred - f_content) ** 2) / (n1*n2*n3*n4)

    # style loss
    styleLoss = 0.
    for layer in feature_pred.keys():
        gram_y = Gram(feature_pred[layer])
        gram_s = Style_Gram[layer].expand_as(gram_y)
        styleLoss += styleWeight * torch.sum((gram_y - gram_s)**2)

    totalLoss = contentLoss + styleLoss + tvLoss
    return totalLoss, contentLoss, styleLoss, tvLoss


def normalize(x):
    y = x.div(255)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    y[:, 0, :, :] = (y[:, 0, :, :] - mean[0]) / std[0]
    y[:, 1, :, :] = (y[:, 1, :, :] - mean[1]) / std[1]
    y[:, 2, :, :] = (y[:, 2, :, :] - mean[2]) / std[2]
    return y


# load pretrained VGG16 network
vgg = LossNet()
vgg.cuda()

dtype = torch.cuda.FloatTensor

target = Variable(torch.randn(1, 3, imsize, imsize).cuda(), requires_grad=True)


contentWeight = 1.0
styleWeight = 5.0
tvWeight = 1e-3

# load content image and style image
style_data = LoadImage('StyleImages/van.jpg').type(dtype)
Style_Gram = StyleGram(normalize(style_data))
# content_data = LoadImage('Dataset/CropData/2.jpg').type(dtype)
content_data = LoadImage('amber.jpg').type(dtype)
loss_fn = nn.MSELoss().cuda()

optimizer = torch.optim.LBFGS([target])
nStep = 0
while nStep < 400:
    def closure():
        global nStep
        target.data.clamp_(0, 255)
        nStep += 1
        optimizer.zero_grad()
        loss, CL, SL, TVL = TotalLoss(normalize(target), normalize(content_data))
        if nStep % 1 == 0:
            print(nStep, 'Total:', loss.data[0], 'Content:', CL.data[0], 'Style:', SL.data[0], 'TV:', TVL.data[0])
        loss.backward()
        return loss
    optimizer.step(closure)

SaveImage(target, '1e-3.png')
# van 1-10
# mosaic 1

# fig1: for one style and content, sweep relative weight of style and content from 0 to infty
# conpare the results

# fig2: for certain style transfer weight, change tv_weight

# fig3: change layers in vgg for content and style reconstruction respecitively (one layer at a time)


