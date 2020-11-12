import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random
def bland_altman_plot(data1,data2,*args,**kwargs):
    data1=np.asarray(data1)
    data2=np.asarray(data2)
    mean=np.mean([data1,data2],axis=0)
    diff=data1-data2
    md=np.mean(diff)
    sd=np.std(diff,axis=0)
    print(md)
    print(md+1.96*sd)
    print(md-1.96*sd)
    plt.scatter(mean,diff,s=3,color=(0,0,0.8),*args,**kwargs)
    plt.axhline(md,color='gray',linestyle='--')
    plt.axhline(md+1.96*sd,color='red')
    plt.axhline(md-1.96*sd,color='red')

Area_lv_label=[]
Area_la_label=[]
MA_label=[]
MV_label=[]
LV_VOL_label=[]

Area_lv_pred=[]
Area_la_pred=[]
MA_pred=[]
MV_pred=[]
LV_VOL_pred=[]



font2={'family':'Times New Roman',
       'weight':'normal',
       'size':14,
       }
plt.xlabel('Mean of LVV (ml)',font2)
plt.ylabel('Difference of LVV (ml)',font2)
label=open('/home/guolibao/cardiac/result_all/label.txt').readlines()
pred_ours=open('/home/guolibao/cardiac/result_all/dcn_sobel_pred.txt').readlines()
for content in label:
    LV_VOL_label.append(float(content.split(',')[4]))
    #Dice_la_psp.append(float(content.split(',')[1]))
for content in pred_ours:
    LV_VOL_pred.append(float(content.split(',')[4]))

bland_altman_plot(LV_VOL_label,LV_VOL_pred)
plt.tick_params(labelsize=14)

#plt.title('Bland -Altman Plot')
#plt.show()
plt.savefig('LVV_bland.png')