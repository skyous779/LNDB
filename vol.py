import nibabel as nib
import numpy as np
X = nib.load("volume-0.nii")

#把仿射矩阵和头文件都存下来
affine = X.affine.copy()
hdr = X.header.copy()

#取数据
X_data = X.get_data()  
#print(X_data)
#像素归一化

X_data= (X_data+300)/600*255
X_data=np.clip(X_data,0,255)

new_nii = nib.Nifti1Image(X_data, affine, hdr)

#保存nii文件，后面的参数是保存的文件名,这里覆盖原来的nii的格式
nib.save(new_nii, "volume-0_pre.nii")