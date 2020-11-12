from __future__ import print_function
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import os
import cv2
import os.path as osp
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


datadir = '/home/guolibao/cardiac'
voc_root = os.path.join(datadir, 'cardiac-4ch')
os.environ['CUDA_VISIBLE_DEVICES']='1'
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


def fast_hist(label_pred, label_gt, num_category):
    mask = (label_gt >= 0) & (label_gt < num_category)  # include background
    hist = np.bincount(
        num_category * label_pred[mask] + label_gt[mask].astype(int),
        minlength=num_category ** 2).reshape(num_category, num_category)
    return hist

def evaluation_metrics(label_preds, label_gts, num_category):
    """Returns evaluation result.
      - pixel accuracy
      - mean accuracy
      - mean IoU
      - frequency weighted IoU
      - dice
      - recall
      - pre
    """
    hist = np.zeros((num_category, num_category))
    for p, g in zip(label_preds, label_gts):
        tmp = (g < 10)
        hist += fast_hist(p[tmp], g[tmp], num_category)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        macc = np.diag(hist) / hist.sum(axis=0)
    macc = np.nanmean(macc)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist))
    #miou = np.nanmean(iou)
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = 2 * np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1))
    #mdice = np.nanmean(dice)
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.diag(hist) / hist.sum(axis=1)
    #mrecall = np.nanmean(recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        pre = np.diag(hist) / hist.sum(axis=0)
    #mpre = np.nanmean(pre)
    freq = hist.sum(axis=0) / hist.sum()
    fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
    return  iou, dice, recall, pre,acc


def main():

    transforms_val = transforms.Compose([  # transforms.Resize(448),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    voc_data = {
                'val': Cardiac(root_dir=voc_root, train=False,
                                  trsf=transforms_val)}
    dataloaders = {
                   'val': DataLoader(voc_data['val'], batch_size=1,
                                     shuffle=False, num_workers=8)}

    dataset_sizes = {x: len(voc_data[x]) for x in [ 'val']}
    num_category = 2
    mynet_ma = torch.load('/home/guolibao/PycharmProjects/fcn/model_ma/bisenet_ma55.pkl')

    mynet_ma.eval()  # Set model to evaluate mode
    mynet_ma.cuda()


    Dice_Ma=0
    # Recall

    Recall_Ma=0

    # Precision

    Precision_Ma=0
    # jacc

    Jacc_Ma=0

    Acc_ma=0
    for sample in tqdm(dataloaders['val']):
        # res_rec=[]
        inputs, labels = sample['image'], sample['label']
        #print(labels)
        #print(labels.shape)

        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = mynet_ma(inputs)



        outputs = F.log_softmax(outputs, dim=1)

        preds = outputs.data.cpu().numpy()

        labels = labels.data.cpu().numpy ()


        h, w = labels.shape[1:]
        ori_h, ori_w = preds.shape[2:]
        preds = np.argmax(ndimage.zoom(preds, (1., 1., 1. * h / ori_h, 1. * w / ori_w), order=1), axis=1)

        for pred, label in zip(preds, labels):
            iou_ma, dice_ma, recall_ma, pre_ma ,acc= evaluation_metrics(pred, label, num_category)

            Acc_ma+=acc

            jacc_Ma = iou_ma[1]
            Jacc_Ma+=jacc_Ma
            #Dice
            dice_Ma=dice_ma[1]
            Dice_Ma+=dice_Ma
            #Recall
            recall_Ma=recall_ma[1]
            Recall_Ma+=recall_Ma
            #precision
            precision_Ma=pre_ma[1]
            Precision_Ma+=precision_Ma







    acc_ma = Acc_ma / dataset_sizes['val']
    val_jacc_ma=Jacc_Ma/dataset_sizes['val']

    val_dice_ma=Dice_Ma/dataset_sizes['val']

    val_recall_ma=Recall_Ma/dataset_sizes['val']

    val_pre_ma=Precision_Ma/dataset_sizes['val']



    print('Validation Results: ')


    print('acc_Ma acc:{:4f}'.format(acc_ma))
    print('Jacc_Ma acc:{:4f}'.format(val_jacc_ma))
    print('Dice_Ma acc:{:4f}'.format(val_dice_ma))
    print('Recall_Ma acc:{:4f}'.format(val_recall_ma))
    print('Precision_Ma acc:{:4f}'.format(val_pre_ma))





# %%
if __name__ == '__main__':
    main()