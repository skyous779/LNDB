import SimpleITK as sitk
import nibabel as nib
import numpy as np
import glob
from tqdm import tqdm



#转化数据
# mhd_list =  glob.glob('mask/LNDb-00*.mhd')
# print(mhd_list)
# for i in tqdm(range(len(mhd_list))):  #tqdm

#     img = sitk.ReadImage(mhd_list[i])
#     sitk.WriteImage(img, 'mask/nii/'+mhd_list[i][5:19]+'.nii.gz')

nii_list = glob.glob('mask/nii/*.nii.gz')

print(nii_list)
for i in tqdm(range(len(nii_list))):
    X = nib.load(nii_list[i])

    #把仿射矩阵和头文件都存下来
    affine = X.affine.copy()
    hdr = X.header.copy()

    #取数据
    X_data = X.get_data()  
 
    #把像素值大于0的都改成255
    X_data[X_data>0] = 255
    #形成新的nii文件
    new_nii = nib.Nifti1Image(X_data, affine, hdr)
 
    #保存nii文件，后面的参数是保存的文件名,这里覆盖原来的nii的格式
    nib.save(new_nii, nii_list[i])

    #nib.save(new_nii, 'dsfsf')
