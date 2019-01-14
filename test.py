import numpy as np
import torch
import torch.nn as nn
from scipy.misc import imresize
from unet import UNET
import os
from dataload  import ImagLabelDataset
import matplotlib.pyplot as plt

path_abs = os.getcwd()

def show_surface(image_idx,x_data,y_data,y_true):
    x_data_resize = x_data
    y_data_resize = y_data
    x_data_resize = imresize(x_data_resize, (204, 992))
    y_data_resize = imresize(y_data_resize, (204, 992)).astype('float32')
    y_data_resize_true = imresize(y_true, (204, 992))

    #plt.plot(x, y)

    plt.subplot(5, 1, 1)
    plt.imshow(x_data_resize, cmap='gray')
    plt.title('surface tracking image ' + str(image_idx))

    plt.subplot(5, 1, 2)
    plt.imshow(y_data_resize, cmap='gray')
    plt.title('predicted' + str(image_idx))

    plt.subplot(5, 1, 3)
    plt.imshow(y_data_resize_true, cmap='gray')
    plt.title('true ' + str(image_idx))

    plt.subplot(5, 1, 4)
    x = []
    y = []
    #y_data_resize[y_data_resize >=1] =1
    #print(y_data_resize)
    for j in range(992):
        for i in range(204):
            if (y_data_resize)[i,j] >=255 :
                x.append(j)
                y.append(i)
                break

    #print('x len',len(x))
    #print('y len',len(y))
    plt.plot(x, y)
    plt.imshow(x_data_resize, cmap='gray')
    plt.title('predict surface ' + str(image_idx))

    plt.subplot(5, 1, 5)
    x = []
    y = []
    for j in range(992):
        for i in range(204):
            if (y_data_resize_true)[i, j] >= 1.0:
                x.append(j)
                y.append(i)
                break

    # print('x len',len(x))
    # print('y len',len(y))
    plt.plot(x, y)
    plt.imshow(x_data_resize, cmap='gray')
    plt.title('predict surface ' + str(image_idx))
    plt.show()


net = UNET()
net.load_state_dict(torch.load('Unet_200_varyWH_dice'))
test_set = ImagLabelDataset(npz_file='save_xy_reshaped.npz',istrain = False)

def test(i):
    x_data = test_set[i]['image'].unsqueeze_(0)
    y_true = test_set[i]['label']
    print(x_data.size(),y_true.size())
    net.eval()
    output = net(x_data)
    output = torch.squeeze(output)
    output = output.detach().numpy()
    x_data = torch.squeeze(x_data)
    x_data = x_data.detach().numpy()
    y_true = torch.squeeze(y_true)
    y_true = y_true.detach().numpy()
    #print(type(output))
    #plt.imshow(output, cmap = 'gray')
    #plt.show()
    #plt.imshow(x_data)
    show_surface(i,x_data,output,y_true)


if __name__ == "__main__":
    for i in range(5):
        test(i)