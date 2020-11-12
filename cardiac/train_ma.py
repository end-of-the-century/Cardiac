from __future__ import print_function
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from sobel import SobelComputer
import os
import os.path as osp
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DAEFFNet import mynet
datadir = '/home/guolibao/cardiac/'
voc_root = os.path.join(datadir, 'cardiac-4ch')
os.environ['CUDA_VISIBLE_DEVICES']='0'

def read_images(root_dir, train):
    txt_fname = root_dir + '/dic/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r')as f:
        images = f.read().split()

    if train:
        data_list = [os.path.join(root_dir, 'train', i) for i in images]

        label_list = [os.path.join(root_dir, 'train_ma_labels', i) for i in images]
    else:
        data_list = [os.path.join(root_dir, 'val', i) for i in images]
        label_list = [os.path.join(root_dir, 'val_ma_labels', i) for i in images]
    return data_list, label_list

class Cardiac(Dataset):


    def __init__(self, root_dir=voc_root, train=True, trsf=None):
        self.root_dir = root_dir
        self.trsf = trsf
        self.data_list, self.label_list = read_images(root_dir, train)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image, label = self.data_list[idx], self.label_list[idx]
        image, label = Image.open(image).convert('RGB'), Image.open(label)
        #print(image.shape)
        #print(label.shape)
        image = transforms.Resize((512, 512), interpolation=Image.BILINEAR)(image)
        label = transforms.Resize((512, 512), interpolation=Image.NEAREST)(label)
        sample = {'image': image, 'label': label}
        if self.trsf:
            sample = self.trsf(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label, dtype='int'))
        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0., 0., 0.], std=[1., 1., 1.]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = transforms.Normalize(self.mean, self.std)(image)
        return {'image': image, 'label': label}


def main():
    transforms_train = transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    voc_data = {'train': Cardiac(root_dir=voc_root, train=True,
                                    trsf=transforms_train),
                }
    dataloaders = {'train': DataLoader(voc_data['train'], batch_size=1,
                                       shuffle=True, num_workers=8),
                   }
    dataset_sizes = {x: len(voc_data[x]) for x in ['train']}


    Network_ma=mynet(2)

    num_epoch =80
    criterion = nn.NLLLoss(ignore_index=255)

    # observer that all parameters are being optimized

    optimizer = optim.SGD(Network_ma.parameters(), lr=2e-3, momentum=0.99)
    sobel_compute = SobelComputer()
    # (LR) Decreased  by a factor of 10 every 2000 iterations
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.9)  #
    Network_ma = nn.DataParallel(Network_ma).cuda()


    # %% Train
    for t in range(num_epoch):  #

        Network_ma.train()  # Set model to training mode
        tbar = tqdm(dataloaders['train'])
        running_loss = 0

        # Iterate over data.
        for i, sample in enumerate(tbar):
            exp_lr_scheduler.step()
            inputs, labels = sample['image'], sample['label']


            inputs=inputs.cuda()
            labels = labels.cuda()


            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # forward
                outputs = Network_ma(inputs)
                label = labels.unsqueeze(1)

                label = label.cuda()
                # print(label.shape)
                pre_sobel, label_sobel = sobel_compute.compute_edges(outputs, label.float())
                sobel_loss = F.l1_loss(pre_sobel, label_sobel)

                outputs = F.log_softmax(outputs, dim=1)
                loss = criterion(outputs, labels.long())
                # backward + optimize
                loss = loss + 0.1*sobel_loss
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / dataset_sizes['train']
        print('Training Results({}): '.format(t))
        print('Loss: {:4f}'.format(train_loss))
        if (t+1)%5==0:
            torch.save(Network_ma, 'model_ma/mynet_sobel_%d.pkl'%(t+1))





# %%
if __name__ == '__main__':
    main()