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
        #image = transforms.Resize((512, 512), interpolation=Image.BILINEAR)(image)
        #label = transforms.Resize((512, 512), interpolation=Image.BILINEAR)(label)
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
    return  iou, dice, recall, pre, acc



def main():
    transforms_val = transforms.Compose([  # transforms.Resize(448),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    voc_data = {'val': Cardiac(root_dir=voc_root, train=False,
                                  trsf=transforms_val)}
    dataloaders = {
                   'val': DataLoader(voc_data['val'], batch_size=1,
                                     shuffle=False, num_workers=8)}
    dataset_sizes = {x: len(voc_data[x]) for x in [ 'val']}
    num_category = 3
    mynet= torch.load('/home/guolibao/Data/unet_aspp/unetaspp_mv45.pkl')

    mynet.eval()  # Set model to evaluate mode
    mynet.cuda()

    # dice
    Dice_Lv = 0
    Dice_La = 0
    Dice_mean = 0
    # Recall
    Recall_Lv = 0
    Recall_La = 0
    Recall_mean = 0
    # Precision
    Precision_Lv = 0
    Precision_La = 0
    Precision_mean = 0
    # jacc
    Jacc_Lv = 0
    Jacc_La = 0
    Jacc_mean = 0

    accu=0
    for sample in tqdm(dataloaders['val']):
        # res_rec=[]
        inputs, labels = sample['image'], sample['label']
        inputs = inputs.cuda()
        labels = labels.cuda()
        # forward
        outputs = mynet(inputs)
        outputs = F.log_softmax(outputs, dim=1)
        preds = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        h, w = labels.shape[1:]
        ori_h, ori_w = preds.shape[2:]
        preds = np.argmax(ndimage.zoom(preds, (1., 1., 1. * h / ori_h, 1. * w / ori_w), order=1), axis=1)
        for pred, label in zip(preds, labels):
            iou, dice, recall, pre ,acc= evaluation_metrics(pred, label, num_category)
            accu+=acc
            #1 is background,2 is lv,3 is la
            jacc_la = iou[2:]
            jacc_la = np.mean(jacc_la)
            jaccll = iou[1:]
            jacc_mean = np.mean(jaccll)
            jacc_lv = jaccll[:1]
            jacc_lv = np.mean(jacc_lv)
            Jacc_Lv += jacc_lv
            Jacc_La += jacc_la
            Jacc_mean += jacc_mean

            dice_la = dice[2:]
            dice_la = np.mean(dice_la)
            dice_ll = dice[1:]
            dice_mean = np.mean(dice_ll)
            dice_lv = dice_ll[:1]
            dice_lv = np.mean(dice_lv)
            Dice_La += dice_la
            Dice_Lv += dice_lv
            Dice_mean += dice_mean

            recall = np.nan_to_num(recall)
            recall_la = recall[2:]
            recall_la = np.mean(recall_la)
            recall_ll = recall[1:]
            recall_mean = np.mean(recall_ll)
            recall_lv = recall_ll[:1]
            recall_lv = np.mean(recall_lv)
            Recall_La += recall_la
            Recall_Lv += recall_lv
            Recall_mean += recall_mean

            precision_la = pre[2:]
            precision_la = np.mean(precision_la)
            precision_ll = pre[1:]
            precision_mean = np.mean(precision_ll)
            precision_lv = precision_ll[:1]
            precision_lv = np.mean(precision_lv)
            Precision_La += precision_la
            Precision_Lv += precision_lv
            Precision_mean += precision_mean


    val_jacc_lv = Jacc_Lv / dataset_sizes['val']
    val_jacc_la = Jacc_La / dataset_sizes['val']
    val_jacc_mean = Jacc_mean / dataset_sizes['val']

    val_dice_lv = Dice_Lv / dataset_sizes['val']
    val_dice_la = Dice_La / dataset_sizes['val']
    val_dice_mean = Dice_mean / dataset_sizes['val']

    val_recall_lv = Recall_Lv / dataset_sizes['val']
    val_recall_la = Recall_La / dataset_sizes['val']
    val_recall_mean = Recall_mean / dataset_sizes['val']

    val_pre_lv = Precision_Lv / dataset_sizes['val']
    val_pre_la = Precision_La / dataset_sizes['val']
    val_pre_mean = Precision_mean / dataset_sizes['val']

    val_acc=accu/dataset_sizes['val']


    print('Validation Results: ')


    print('val_acc acc:{:4f}'.format(val_acc))
    print('Jacc_lv acc:{:4f}'.format(val_jacc_lv))
    print('Jacc_la acc:{:4f}'.format(val_jacc_la))
    print('Jacc_mean acc:{:4f}'.format(val_jacc_mean))
    print('Dice_lv acc:{:4f}'.format(val_dice_lv))
    print('Dice_la acc:{:4f}'.format(val_dice_la))
    print('Dice_mean acc:{:4f}'.format(val_dice_mean))
    print('Recall_lv acc:{:4f}'.format(val_recall_lv))
    print('Recall_la acc:{:4f}'.format(val_recall_la))
    print('Recall_mean acc:{:4f}'.format(val_recall_mean))
    print('Precision_lv acc:{:4f}'.format(val_pre_lv))
    print('Precision_la acc:{:4f}'.format(val_pre_la))
    print('Precision_mean acc:{:4f}'.format(val_pre_mean))

# %%
if __name__ == '__main__':
    main()