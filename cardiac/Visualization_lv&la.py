from __future__ import print_function
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import os
import os.path as osp
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

#from skimage import io
from skimage import segmentation as seg
import glob


datadir = '/home/guolibao/cardiac'
voc_root = os.path.join(datadir, 'cardiac-4ch')
os.environ['CUDA_VISIBLE_DEVICES']='3'
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

def read_images(root_dir, train):
    txt_fname = root_dir + '/dic/' + ('train.txt' if train else 'val_ll.txt')
    with open(txt_fname, 'r')as f:
        images = f.read().split()

    if train:
        data_list = [os.path.join(root_dir, 'train', i) for i in images]

        label_list = [os.path.join(root_dir, 'train_labels', i) for i in images]
    else:
        data_list = [os.path.join(root_dir, 'val', i) for i in images]
       # print(data_list)
        label_list = [os.path.join(root_dir, 'val_labels', i) for i in images]
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
        image = transforms.Resize((512, 512), interpolation=Image.BILINEAR)(image)
        label = transforms.Resize((512, 512), interpolation=Image.BILINEAR)(label)
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

    transforms_val = transforms.Compose([  # transforms.Resize(448),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    voc_data = {
                'val': Cardiac(root_dir=voc_root, train=False,
                                  trsf=transforms_val)}

    mynet = torch.load('/home/guolibao/Data/unet_aspp/unetaspp_mv45.pkl')

    mynet.eval()  # Set model to evaluate mode
    mynet.cuda()
    colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]
    cm = np.array(colormap, dtype='uint8')
    txt_fname = '/home/guolibao/cardiac/cardiac-4ch/dic/val.txt'
    with open(txt_fname, 'r')as f:
        images = f.read().split()
    imgs = [os.path.join(datadir, 'cardiac/val_labels', i) for i in images]
    for i, img in enumerate(imgs):
        val_sample = voc_data['val'][i]
        val_image = val_sample['image'].cuda()
        val_label = val_sample['label']
        val_output = mynet(val_image.unsqueeze(0))
        val_pred = val_output.max(dim=1)[1].squeeze(0).data.cpu().numpy()
        val_label = val_label.long().data.numpy()
        val_image = val_image.squeeze().data.cpu().numpy().transpose((1, 2, 0))
        val_image = val_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        val_image *= 255
        val_image = val_image.astype(np.uint8)
        val_pred_tian = cm[val_pred]
        val_pred_bin=seg.mark_boundaries(val_image,val_label,color=(128,0,0))
        val_pred_bin = seg.mark_boundaries(val_pred_bin, val_pred, color=(0, 128, 0))

        plt.imsave(osp.join('/home/guolibao/cardiac/pred_test/unetaspp_mv', img.split('/')[-1]), val_pred_tian)
        plt.imsave(osp.join('/home/guolibao/cardiac/pred_test/unetaspp_mv_bin', img.split('/')[-1]), val_pred_bin)



if __name__ == '__main__':
    main()