import SimpleITK as sitk
import nibabel as nib
import numpy as np
import glob
from tqdm import tqdm



# #######################################转化数据(mhd -> nii)#########################
# mhd_list =  glob.glob('mask/mhd_mask/*.mhd')         #源文件路径(将代码放到同源文件一个目录上)
# #print(mhd_list)
# for i in range(len(mhd_list)):  #tqdm
    
#     print(mhd_list[i][14:-4])
#     img = sitk.ReadImage(mhd_list[i])
#     sitk.WriteImage(img, 'mask/mask_nii/'+mhd_list[i][14:-4]+'.nii.gz')   ###输出路径



##############################像素值修改###########################################
nii_list = glob.glob('mask/mask_nii/*rad1.nii.gz')   #j基于rad1上合并像素

nii_list = sorted(nii_list)
#print(nii_list)
#print(nii_list[0][:-8])

for i in tqdm(range(len(nii_list))):

    #nii_list[i] 为第i个rad1—mask的路径
    rad_name = nii_list[i][:-8]   #编号
    #print(rad_name[14:])


    nii_list_num = sorted(glob.glob(rad_name+'*.nii.gz'))

    X = nib.load(nii_list[i])
    #把仿射矩阵和头文件都存下来
    affine = X.affine.copy()
    hdr = X.header.copy()

    #读取数据
    X_data = X.get_data() 
    X_data[X_data>0] = 255
    
    #合并数据
    for j in range(len(nii_list_num) - 1):
        #print(nii_list_num[j+1])
        Y = nib.load(nii_list_num[j+1])
        Y_data = Y.get_data() 
        X_data[Y_data>0] = 255

    new_nii = nib.Nifti1Image(X_data, affine, hdr)
    nib.save(new_nii, 'mask/mask_merge/'+rad_name[14:]+'.nii.gz')


    
    





    
    
    # X = nib.load(nii_list[i])
    # #把仿射矩阵和头文件都存下来
    # affine = X.affine.copy()
    # hdr = X.header.copy()

    # #取数据
    # X_data = X.get_data()  
 
    # #把像素值大于0的都改成255
    # X_data[X_data>0] = 255
    # #形成新的nii文件
    # new_nii = nib.Nifti1Image(X_data, affine, hdr)
 
    # #保存nii文件，后面的参数是保存的文件名,这里覆盖原来的nii的格式
    # nib.save(new_nii, nii_list[i])

    # #nib.save(new_nii, 'dsfsf')
