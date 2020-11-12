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
import json
from skimage import measure
import glob
from skimage import segmentation as seg
import math
path='/home/guolibao/cardiac/result_all/'
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

def save_max_objects(img):
    labels=measure.label(img)
    jj=measure.regionprops(labels)
    if len(jj)==1:
        out=img
    else:
        num=labels.max()
        del_array=np.array([0]*(num+1),dtype=np.uint8)
        for k in range(num):
            if k==0:
                initial_area=jj[0].area
                save_index=1
            else:
                k_area=jj[k].area

                if initial_area<k_area:
                    initial_area=k_area
                    save_index=k+1
        del_array[save_index]=1
        del_mask=del_array[labels]
        out=img*del_mask
    return out
class Point():
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def vector(self,other):
        xx=self.x-other.x
        yy=self.y-other.y
        return Point(xx,yy)
    def cross(self,other):
        return (self.x*other.y-self.y*other.x)
    def dot(self,other):
        xx=self.x*other.x
        yy=self.y*other.y
        return (xx+yy)
    def mochang(self):
        return (self.x**2+self.y**2)**0.5
class Point_line_2d():
    def __init__(self,x,y,a,b,A,B):
        self.x=x
        self.y=y
        self.a=a
        self.b=b
        self.A=A
        self.B=B
    def calculate(self):
        point=Point(self.x,self.y)
        point_in_line=Point(self.a,self.b)
        point_in_zero=Point(0,0)
        point_line_direction=Point(self.A,self.B)

        vector1=point.vector(point_in_line)
        vector2=point_in_zero.vector(point_line_direction)

        return math.fabs(vector1.cross(vector2))/vector2.mochang()
def calculate_ma_mv(x):
    #img = cv2.imread(x)
    img = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    img=save_max_objects(img)
    contours, hierarchy = cv2.findContours(img, 3, 2)
    bindary=contours[0]
    point_all = int(bindary.shape[0])
    y_min=bindary[0][0][1]
    y_max=10000000
    #x1 = 0
    #y1 = 0
    point1 = []
    point2 = []
    #point3 = []
    for i in range(point_all):
        # print(i)
        if bindary[i][0][1] >= y_min:
            y_min = bindary[i][0][1]
            # print(h)
            # print(arr[i][0])
            point1 = bindary[i][0]
    # print(point1)
    for i in range(point_all):
        if bindary[i][0][1] <= y_max:
            y_max = bindary[i][0][1]
            point2 = bindary[i][0]
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0] - point1[0]
    y2 = point2[1] - point1[1]
    list = []
    for i in range(point_all):
        point3_1 = bindary[i][0]
        # print(point3)
        x = point3_1[0]
        y = point3_1[1]
        p_l_2d = Point_line_2d(x, y, x1, y1, x2, y2)
        D = p_l_2d.calculate()
        list.append(D)
    index = list.index(max(list))
    point3 = bindary[index][0]
    middle = [(point1[0] + point3[0]) / 2, (point1[1] + point3[1]) / 2]
    point_mid = (int(middle[0]), int(middle[1]))
    p1 = np.array(point2)
    p2 = np.array(middle)
    p3 = p2 - p1
    ma = math.hypot(p3[0], p3[1])
    P_ma_1=np.array(point1)
    P_ma_2=np.array(point3)
    p_ma=P_ma_2-P_ma_1
    mv=math.hypot(p_ma[0],p_ma[1])
    return point1,point2,point3,point_mid,ma,mv

def main():

    transforms_val = transforms.Compose([  # transforms.Resize(448),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    voc_data = {
                'val': Cardiac(root_dir=voc_root, train=False,
                                  trsf=transforms_val)}



    mynet_ma = torch.load('/home/guolibao/PycharmProjects/fcn/model_ma/bisenet_ma55.pkl')

    mynet_ma.eval()  # Set model to evaluate mode
    mynet_ma.cuda()

    colormap = [[0, 0, 0], [128, 0, 0]]
    cm = np.array(colormap, dtype='uint8')

    txt_fname = '/home/guolibao/cardiac/cardiac-4ch/dic/val.txt'
    with open(txt_fname, 'r')as f:
        images = f.read().split()
    imgs = [os.path.join(datadir, 'cardiac/val_labels', i) for i in images]
    json_fname = '/home/guolibao/cardiac/cardiac-4ch/dic/val_json.txt'
    with open(json_fname, 'r')as f:
        json_name = f.read().split()
    line_segment = [os.path.join(datadir, 'line_segment', i) for i in json_name]
    # print(line_segment)
    result_pred_txt = path + 'pspnet_ma_pred' + '.txt'
    pred_txt = open(result_pred_txt, 'w')
    for i, img in enumerate(imgs):
        val_sample = voc_data['val'][i]
        val_image = val_sample['image'].cuda()

        val_output = mynet_ma(val_image.unsqueeze(0))
        val_pred = val_output.max(dim=1)[1].squeeze(0).data.cpu().numpy()

        val_image = val_image.squeeze().data.cpu().numpy().transpose((1, 2, 0))
        val_image = val_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        val_image *= 255

        val_pred_ma = cm[val_pred]
        _, _, _, _, ma, mv = calculate_ma_mv(val_pred_ma)

        for name_line in line_segment:
            with open(name_line, 'r') as f:
                json_data = json.load(f)
                json_p1 = json_data['shapes'][0]['points'][0]
                json_p2 = json_data['shapes'][0]['points'][1]
                json_p1 = np.array(json_p1)
                json_p2 = np.array(json_p2)
                json_p = json_p2 - json_p1
                line_length = math.hypot(json_p[0], json_p[1])
        MA = (5 * ma) / line_length
        MV = (5 * mv) / line_length
        print(MA)
        print(MV)
        pred_num = str(MV) + ',' + str(MA) + '\n'
        pred_txt.write(pred_num)
        # print(lv_vol_pred)
    pred_txt.close()




# %%
if __name__ == '__main__':
    main()