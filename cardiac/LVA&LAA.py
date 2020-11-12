from __future__ import print_function
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import os
import os.path as osp
import cv2
import torch
import numpy as np
import math
import json


import matplotlib.pyplot as plt

path='/home/guolibao/cardiac/result_all/'
datadir = '/home/guolibao/cardiac'
voc_root = os.path.join(datadir, 'cardiac-4ch')
os.environ['CUDA_VISIBLE_DEVICES']='3'
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

def read_images(root_dir, train):
    txt_fname = root_dir + '/dic/' + ('train.txt' if train else 'val.txt')
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
    #imgs = glob.glob('/home/guolibao/cardiac-jbhi/cardiac-4ch/val/*.png')
    txt_fname = '/home/guolibao/cardiac/cardiac-4ch/dic/val.txt'
    with open(txt_fname, 'r')as f:
        images = f.read().split()
    imgs=[os.path.join(datadir, 'cardiac/val_labels', i) for i in images]
    json_fname = '/home/guolibao/cardiac/cardiac-4ch/dic/val_json.txt'
    with open(json_fname, 'r')as f:
        json_name = f.read().split()
    line_segment=[os.path.join(datadir, 'line_segment', i) for i in json_name]
    #print(line_segment)
    result_pred_txt = path + 'pspnet_lv_pred' + '.txt'
    pred_txt = open(result_pred_txt, 'w')
    #print(imgs)

    for i, img in enumerate(imgs):
        val_sample = voc_data['val'][i]
        val_image = val_sample['image'].cuda()

        val_output = mynet(val_image.unsqueeze(0))
        val_pred = val_output.max(dim=1)[1].squeeze(0).data.cpu().numpy()

        val_pred_tian = cm[val_pred]
        val_pred_tian=cv2.cvtColor(val_pred_tian, cv2.COLOR_RGB2GRAY)
        height = val_pred_tian.shape[0]
        width = val_pred_tian.shape[1]
        area_lv = 0
        area_la = 0

        #area_all = height * width
        #
        for i in range(height):
            for j in range(width):
                if val_pred_tian[i, j] == 38:
                    area_lv = area_lv + 1
                    # gray = img[i, j]
                    # new_img[i, j] = np.uint8(gray)
                elif val_pred_tian[i, j] == 75:
                    area_la = area_la + 1
                else:
                    pass
        for name_line in line_segment:
            with open(name_line, 'r') as f:
                json_data = json.load(f)
                json_p1 = json_data['shapes'][0]['points'][0]
                json_p2 = json_data['shapes'][0]['points'][1]
                json_p1 = np.array(json_p1)
                json_p2 = np.array(json_p2)
                json_p = json_p2 - json_p1
                line_length = math.hypot(json_p[0], json_p[1])
        Area_Lv = (25 * area_lv) / (line_length * line_length)
        Area_La = (25 * area_la) / (line_length * line_length)
        #plt.imshow(val_pred_tian)
        #plt.show()
        print(Area_Lv)
        print(Area_La)
        pred_num = str(Area_Lv) + ',' + str(Area_La)  + '\n'
        pred_txt.write(pred_num)
        #print(lv_vol_pred)
    pred_txt.close()




# %%
if __name__ == '__main__':
    main()