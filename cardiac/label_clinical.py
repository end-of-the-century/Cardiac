import json
import numpy as np
import cv2
import math
import glob
from skimage import measure
import os
datadir = '/home/guolibao/cardiac'
path='/home/guolibao/cardiac/result_all/'
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

def calculate_area_lv(x):

    img=cv2.imread(x)
    #plt.imshow(img)
    #plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(img)
    #plt.show()
    #ret, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    height=img.shape[0]
    width=img.shape[1]
    area_lv=0
    area_la=0
    #new_img=np.zeros((height,width),np.uint8)
    #print(h)
    #print(l)
    area_all=height*width
    #
    for i in range(height):
        for j in range(width):
            if img[i, j] == 15:
                area_lv=area_lv+1
                #gray = img[i, j]
                #new_img[i, j] = np.uint8(gray)
            elif img[i,j]==75:
                area_la=area_la+1
            else:
                pass
                #gray = 255 - img[i, j]
                #new_img[i, j] = np.uint8(gray)
    #plt.imshow(img)
    #plt.show()
    #ret, binary = cv2.threshold(new_img, 200, 250, cv2.THRESH_BINARY)
    #contours, hierarchy = cv2.findContours(binary, 3, 2)
    #@plt.imshow(binary)
    #plt.show()

    #cnt_lv=contours[0]

    #cnt_lv = contours[1]
    #area_lv = cv2.contourArea(cnt_lv)
    #area_la=cv2.contourArea(cnt_la)
    return area_all,area_lv,area_la
def calculate_ma_mv(x):
    img = cv2.imread(x)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

txt_fname = '/home/guolibao/cardiac/cardiac-4ch/dic/val.txt'
with open(txt_fname, 'r')as f:
    images = f.read().split()
img_lv_la = [os.path.join(datadir, 'cardiac-4ch/val_labels', i) for i in images]
img_ma=[os.path.join(datadir, 'cardiac-4ch/val_ma_labels', i) for i in images]
json_fname = '/home/guolibao/cardiac/cardiac-4ch/dic/val_json.txt'
with open(json_fname, 'r')as f:
    json_name = f.read().split()
line_segment = [os.path.join(datadir, 'line_segment', i) for i in json_name]
result_label_txt=path+'label'+'.txt'
label_txt=open(result_label_txt,'w')
for name_lv,name_ma,name_line in zip(img_lv_la,img_ma,line_segment):
    print(name_lv)
    print(name_ma)
    print(name_line)
    _,area_lv,area_la=calculate_area_lv(name_lv)
    _,_,_,_,ma,mv=calculate_ma_mv(name_ma)
    with open(name_line, 'r') as f:
        json_data = json.load(f)
        json_p1 = json_data['shapes'][0]['points'][0]
        json_p2 = json_data['shapes'][0]['points'][1]
        json_p1=np.array(json_p1)
        json_p2=np.array(json_p2)
        json_p=json_p2-json_p1
        line_length=math.hypot(json_p[0],json_p[1])
    Area_Lv=(25*area_lv)/(line_length*line_length)
    Area_La=(25*area_la)/(line_length*line_length)
    MA=(5*ma)/line_length
    MV=(5*mv)/line_length
    lv_vol_label=(8*Area_Lv*Area_Lv)/(3*math.pi*MA)
    label_num=str(Area_Lv)+','+str(Area_La)+','+str(MA)+','+str(MV)+','+str(lv_vol_label)+'\n'
    label_txt.write(label_num)
    print(lv_vol_label)
label_txt.close()
