import math
path='/home/guolibao/cardiac/result_all/'
LVA=[]
LVD=[]

LVA_LAA=open('/home/guolibao/cardiac/result_all/pspnet_lv_pred.txt').readlines()
LVD_MVD=open('/home/guolibao/cardiac/result_all/pspnet_ma_pred.txt').readlines()

for content in LVA_LAA:
    LVA.append(float(content.split(',')[0]))

for content in LVD_MVD:
    LVD.append(float(content.split(',')[1]))
result_pred_txt = path + 'LVV' + '.txt'
pred_txt = open(result_pred_txt, 'w')
for lva,lvd in zip(LVA,LVD):
    lv_vol_pred = (8 * lva * lva) / (3 * math.pi * lvd)
    pred_num = str(lv_vol_pred) + '\n'
    pred_txt.write(pred_num)
    print(lv_vol_pred)
pred_txt.close()