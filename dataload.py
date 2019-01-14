import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.misc import imresize

path_abs = os.getcwd()


class ImagLabelDataset(Dataset):
    """OCT segmentation dataset."""

    def __init__(self, npz_file, transform=None, istrain=True):
        """
        Args:
            npz_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.istrain = istrain
        self.data_xy = np.load(npz_file)
        self.x_data = self.data_xy['x']
        self.y_data = self.data_xy['y']
        self.transform = transform

        if self.istrain:
            self.x_data_train = self.x_data[:20]
            self.y_data_train = self.y_data[:20]
        else:
            self.x_data_test = self.x_data[20:]
            self.y_data_test = self.y_data[20:]


    def __len__(self):
        if self.istrain:
            return len(self.data_xy['x'])-5
        else:
            return len(self.data_xy['x'])-20

    def __getitem__(self, idx):
        if self.istrain:
            image = self.x_data_train[idx]
            # rezide
            image = imresize(image, (204, 992)).astype('float32')
            #new axis
            image = image[np.newaxis,:]
            label = self.y_data_train[idx]
            # rezide 0 and 1 semantic map
            label = imresize(label, (116, 900)).astype('float32')
            label[label >= 1] = 1

            label = label[np.newaxis,:]
            sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}
            if self.transform:
                sample = self.transform(sample)

            return sample
        else:
            image = self.x_data_test[idx]
            # rezide
            image = imresize(image, (204, 992)).astype('float32')
            # new axis
            image = image[np.newaxis, :]
            label = self.y_data_test[idx]
            # rezide 0 and 1 semantic map
            label = imresize(label, (116, 900)).astype('float32')
            label[label >= 1] = 1
            label = label[np.newaxis, :]
            sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}
            if self.transform:
                sample = self.transform(sample)

            return sample

'''
dataset = ImagLabelDataset(npz_file='save_xy_reshaped.npz')
sample = dataset[0]
print(0, sample['image'].shape, sample['label'].shape)


'''
'''
x_data = sample['image']
y_data = sample['label']
x=[]
y=[]
for j in range(992):
    for i in range(204):
        if (y_data)[i,j] == 1.0:
            x.append(j)
            y.append(i)
            break

print('x len',len(x))
print('y len',len(y))

plt.plot(x,y)
plt.xlim([0,992])
plt.ylim([0,204])
plt.title('surface tracking image ' + str(1))
plt.imshow(x_data, cmap = 'gray')
plt.show()
'''