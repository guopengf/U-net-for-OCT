from torch.autograd import Variable
from unet import UNET
from dataload  import ImagLabelDataset
import numpy as np
import torch
from scipy.misc import imresize
import os
import torch.nn as nn
import torch.nn.functional as F

num_epochs = 10
batch_size = 2

torch.manual_seed(42)
path_abs = os.getcwd()


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def hellinger_distance(y_pred, y_true, size_average=True):
    n = y_pred.size(0)
    dif = torch.sqrt(y_true) - torch.sqrt(y_pred)
    dif = torch.pow(dif, 2)
    dif = torch.sqrt(torch.sum(dif)) / 1.4142135623730951
    if size_average:
        dif = dif / n
    return dif

class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

def train(isResum = False ,checkpoint = None, path2save = 'test_checkpoint.pth.tar'):
    cuda = torch.cuda.is_available()
    net = UNET()
    checkpointfile = checkpoint
    if cuda:
        net = net.cuda()

    if isResum:
        print("=> loading checkpoint '{}'".format(checkpointfile))
        checkpoint = torch.load(checkpointfile)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion = checkpoint['criterion']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpointfile, checkpoint['epoch']))

    else:
        start_epoch = 0
        criterion =dice_loss
        #criterion = hellinger_distance
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    train_set = ImagLabelDataset(npz_file='save_xy_reshaped.npz',istrain = True)
    test_set = ImagLabelDataset(npz_file='save_xy_reshaped.npz',istrain = False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    print("Data Loading done ...")
    for epoch in range(num_epochs):

        train_loss = Average()
        net.train()
        for i, data in enumerate(train_loader):
            image = data['image']
            label = data['label']
            image= Variable(image)
            label = Variable(label)
            if cuda:
                image = image.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            outputs = net(image)
            #loss = criterion(outputs, label, size_average=False)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.data[0], image.size(0))

        val_loss = Average()
        net.eval()
        for data_test in test_loader:
            image = data_test['image']
            label = data_test['label']
            image = Variable(image)
            label = Variable(label)
            if cuda:
                image = image.cuda()
                label = label.cuda()

            outputs = net(image)
            vloss = criterion(outputs, label)
            val_loss.update(vloss.data[0], image.size(0))

        print("Epoch {}, Loss: {}, Validation Loss: {}".format(epoch+1+start_epoch, train_loss.avg, val_loss.avg))

    save_checkpoint({
        'epoch': epoch + 1 +start_epoch,
        'state_dict': net.state_dict(),
        'train_loss.avg': train_loss.avg,
        'optimizer': optimizer.state_dict(),
        'criterion': criterion
    },path2save)

    return net


def save_checkpoint(state, filename='test_checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == "__main__":
    net = train(isResum = True,checkpoint = 'test_checkpoint.pth.tar', path2save = 'test_checkpoint2.pth.tar' )
    #net = train()
    #torch.save(net.state_dict(), path_abs+'/Unet_400_varyWH_dice')
