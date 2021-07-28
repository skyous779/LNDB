#查看z轴上的像素和，方便去除。
import nibabel as nib
import numpy as np
import glob


nii_list = glob.glob('org/new/*.nii.gz')
for k in range(len(nii_list)):

    X = nib.load(nii_list[k])
    print(nii_list[k])
    X_data = X.get_data()
    n = X_data.shape[2]
    for i in range(0,n):
        XX = X_data[:, :, i]
        sum_nii = np.sum(XX)
        if sum_nii < 150000:
            print('第'+str(i+1)+'层:'+str(sum_nii))
