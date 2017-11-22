from PIL import Image
import glob
from torchvision import transforms
import os
import numpy as np
import torch
from os import listdir


# resize COCO dataset
gen = glob.glob('Dataset/train2014/*.jpg')
loader = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(256), transforms.ToTensor()])
f = transforms.ToPILImage()

for i in range(len(gen)):
    if i % 1000 == 0:
        print(i)
    im = Image.open(gen[i]).convert('RGB')
    data = loader(im)
    img = f(data)
    img.save('Dataset/CropData/'+str(i)+'.jpg', 'JPEG')

