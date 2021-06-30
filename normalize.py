import torchio as tio
import torch
import numpy as np
import torch.tensor as tensor
ct = tio.ScalarImage('img/coronacases_org_008.nii.gz')
#np.save('ct',ct.numpy())
# np.set_printoptions(threshold=np.inf)
# print(ct.numpy())
# rescale = tio.RescaleIntensity(
#     out_min_max=(-512, 512))
# ct_normalized = rescale(ct)
# print(ct_normalized.numpy())
# np.save('ct_normalized',ct_normalized.numpy())

## 把numpy打印保存下来
# test=np.load('ct_normalized.npy',encoding = "latin1")  #加载文件
# doc = open('1.txt', 'a')  #打开一个存储文件，并依次写入
# print(test, file=doc)  #将打印内容写入文件中

#
#print(ct.data)
ct.data = torch.clamp(ct.data,min = -512,max=512)
print(ct.data)
ctt = ct
print(ctt.data)
