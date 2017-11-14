from PIL import Image
import glob
from torchvision import transforms
import os
import numpy as np
import torch
from os import listdir

##############################################
#
# Preprecess the COCO dataset
# (run different section separately)
# 1. Resize to 256x256
# 2. Remove all grey scale images
# 3. Calculate mean values for each channel
# 4. Calculate std for each channel
#
##############################################


# # resize COCO dataset
# dir = 'Dataset/train2014/'
# gen = listdir(dir)

# for i in range(len(gen)):
#     fname = dir + gen[i]
#     im = Image.open(fname)
#     img = im.resize((256, 256), Image.ANTIALIAS)
#     img.save('Dataset/PreData/'+str(i)+'.jpg', 'JPEG')

# get mean value of all samples
gen = glob.glob('Dataset/PreData/*.jpg')

loader = transforms.ToTensor()

# # remove grey scale images
# for i in range(len(gen)):
#     fname = gen[i]
#     im = Image.open(fname)
#     data = loader(im)
#
#     if data.shape[0] != 3:
#         print i, gen[i]
#         os.remove(gen[i])

# r_mean = 0.0
# g_mean = 0.0
# b_mean = 0.0
#
# for i in range(len(gen)):
#     if i % 1000 == 0:
#         print i
#     fname = gen[i]
#     im = Image.open(fname)
#     data = loader(im)
#
#     size = data.shape[1] * data.shape[2] * 1.0
#     r_mean += torch.sum(data[0, :, :]) / size
#     g_mean += torch.sum(data[1, :, :]) / size
#     b_mean += torch.sum(data[2, :, :]) / size
#
# r_mean /= len(gen)
# g_mean /= len(gen)
# b_mean /= len(gen)
#
# print r_mean
# print g_mean
# print b_mean

# calculate std for each dimension for all samples
r_mean = 0.471116568055
g_mean = 0.446522854439
b_mean = 0.406828638561

r_var = 0.0
g_var = 0.0
b_var = 0.0

for i in range(len(gen)):
    if i % 1000 == 0:
        print i
    im = Image.open(gen[i])
    data = loader(im)
    r_var += torch.var(data[0, :, :] - r_mean)
    g_var += torch.var(data[1, :, :] - g_mean)
    b_var += torch.var(data[2, :, :] - b_mean)

r_var /= len(gen)
g_var /= len(gen)
b_var /= len(gen)

print np.sqrt(r_var)
print np.sqrt(g_var)
print np.sqrt(b_var)

r_std = 0.244088944037
g_std = 0.239412672854
b_std = 0.243587113539
