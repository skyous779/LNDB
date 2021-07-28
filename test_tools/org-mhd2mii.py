import SimpleITK as sitk
import nibabel as nib
import numpy as np
import glob
from tqdm import tqdm

##############################转化数据######################################################
mhd_list =  glob.glob('org/LNDb-0*.mhd')
print(mhd_list)
for i in tqdm(range(len(mhd_list))):  #tqdm

    img = sitk.ReadImage(mhd_list[i])
    sitk.WriteImage(img, 'org/nii/' + mhd_list[i][4:13]+'.nii.gz')

nii_list = glob.glob('org/nii/*.nii.gz')


print(nii_list)

################################肺窗###################################################
for i in tqdm(range(len(nii_list))):
    X = nib.load(nii_list[i])

    #把仿射矩阵和头文件都存下来
    affine = X.affine.copy()
    hdr = X.header.copy()

    #取数据
    X_data = X.get_data()  
    #print(X_data)
    #像素归一化
    X_data=np.clip(X_data,-1000,400)
    X_data= (X_data+1000)/1401 * 255

    #形成新的nii文件
    new_nii = nib.Nifti1Image(X_data, affine, hdr)
 
    #保存nii文件，后面的参数是保存的文件名,这里覆盖原来的nii的格式
    nib.save(new_nii, nii_list[i])


