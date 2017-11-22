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


class COCODataset(Dataset):
    def __init__(self, path, transform):
        self.images = glob.glob(path)
        self.transform = transform
    def __getitem__(self, index):
        fname = self.images[index]
        img = Image.open(fname)
        return self.transform(img)
    def __len__(self):
        return len(self.images)


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


def SaveImage(tensor_orig, tensor_transformed, filename):
    def RGB(image):
        return (image.transpose(0, 2, 3, 1)).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(RGB(tensor_transformed.data.cpu().numpy())[0])
    orig = Image.fromarray(RGB(tensor_orig.data.cpu().numpy())[0])
    new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
    new_im.paste(orig, (0, 0))
    new_im.paste(result, (result.size[0] + 5, 0))
    new_im.save(filename)


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
    contentLoss = contentWeight * loss_fn(feature_pred[8], Variable(feature_content[8].data, requires_grad=False))

    # style loss
    styleLoss = 0.
    for layer in feature_pred.keys():
        gram_s = Style_Gram[layer]
        gram_y = Gram(feature_pred[layer])
        styleLoss += styleWeight * loss_fn(gram_y, gram_s.expand_as(gram_y))

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


torch.backends.cudnn.benchmark=True
dtype = torch.cuda.FloatTensor

imsize = 256
batchSize = 4

# Define loss network
vgg = LossNet()
vgg.cuda()

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])

dataset = COCODataset('Dataset/PreData/*.jpg', transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=4)


# load content image and style image
style_data = LoadImage('candy.jpg').type(dtype)
Style_Gram = StyleGram(normalize(style_data))

stylenet = StyleNet()
stylenet.cuda()

loss_fn = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(stylenet.parameters(), lr=1e-3)

contentWeight = 1.0
styleWeight = 8e3
tvWeight = 1e-7

counter = 0
loss_list = []

for j in range(2):
    for i, content_data in enumerate(loader):
        content_data = Variable(content_data.cuda(), requires_grad=False)
        y_pred = stylenet(content_data)
        loss, CL, SL, TVL = TotalLoss(normalize(y_pred), normalize(Variable(content_data.data.clone(), volatile=True)))
        loss_list.append([loss.data[0], CL.data[0], SL.data[0], TVL.data[0]])
        if counter % 500 == 0:
            fname = 'models/epoch_'+str(j)+'_iter_'+str(i)+'_'+'{:.3f}'.format(loss.data[0]) + \
                    '_'+'{:.3f}'.format(CL.data[0])+'_'+'{:.3f}'.format(SL.data[0])
            torch.save(stylenet, fname+'.model')
            SaveImage(content_data[0,:,:,:].unsqueeze(0), stylenet(content_data[0,:,:,:].unsqueeze(0)), fname+'.png')
        if counter % 50 == 0:
            print(i, 'Total:', loss.data[0], 'Content:', CL.data[0], 'Style:', SL.data[0], 'TV:', TVL.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
    np.save('Loss', np.asarray(loss_list))

# candy 8e3 - model2
# van 5e4 - model1  /   1e5 models 3/4
# mosaic 8e4 - model
