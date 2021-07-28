#测试采样方式

import torch
import torchio as tio
transform = tio.Resample((1, 1, 1)) 

subject = tio.Subject(
    source=tio.ScalarImage('org/new/LNDb-0001.nii.gz'),
    label=tio.Image('mask/LNDb-0001_rad1.nii.gz',type=tio.SAMPLING_MAP),
)
a = transform(subject)
print(a.source.affine)

'''
[[  -1.     0.     0.   158. ]
 [   0.    -1.     0.   309. ]
 [   0.     0.     1.  -297.5]
 [   0.     0.     0.     1. ]]
'''