import nibabel as nib
import numpy as np
import glob

org_list =  glob.glob('prodata/org' + '/*.nii.gz')
# print(org_list)
# print(type(org_list))
# print(type(org_list[1]))


for i in range(len(org_list)):
    # #加载图片 
    # if i+8 == 10:
    #     X = nib.load('img/coronacases_org_0'+ str(i+8) +'.nii.gz')
    # else:
    #     X = nib.load('img/coronacases_org_00'+ str(i+8) +'.nii.gz')

    #加载label 
    X = nib.load(org_list[i])
    #X = nib.load('label/coronacases_00'+ str(i+8) +'.nii.gz')
    #X = nib.load('img/coronacases_org_0'+ str(i+10) +'.nii.gz')
    
    #把仿射矩阵和头文件都存下来
    #print(dir(X))
    affine = X.affine.copy()
    hdr = X.header.copy()
    
    #取数据
    X_data = X.get_data()  
    #print(X_data)
    #像素归一化
    X_data=np.clip(X_data,-512,512)
    X_data= (X_data+512)/1024 * 255
    
    #label
    #X_data= (X_data)/2 * 255
    
    #形成新的nii文件
    new_nii = nib.Nifti1Image(X_data, affine, hdr)
 
    #保存nii文件，后面的参数是保存的文件名
    #nib.save(new_nii, 'img/new/coronacases_org_00'+ str(i+8) +'.nii.gz')

    nib.save(new_nii, 'pro_'+org_list)

for i in range(len(org_list)):
    # #加载图片 
    # if i+8 == 10:
    #     X = nib.load('img/coronacases_org_0'+ str(i+8) +'.nii.gz')
    # else:
    #     X = nib.load('img/coronacases_org_00'+ str(i+8) +'.nii.gz')

    #加载label 
    X = nib.load(org_list[i])
    #X = nib.load('label/coronacases_00'+ str(i+8) +'.nii.gz')
    #X = nib.load('img/coronacases_org_0'+ str(i+10) +'.nii.gz')
    
    #把仿射矩阵和头文件都存下来
    #print(dir(X))
    affine = X.affine.copy()
    hdr = X.header.copy()
    
    #取数据
    X_data = X.get_data()  
    #print(X_data)
    #像素归一化
    X_data=np.clip(X_data,-512,512)
    X_data= (X_data+512)/1024 * 255
    
    #label
    #X_data= (X_data)/2 * 255
    
    #形成新的nii文件
    new_nii = nib.Nifti1Image(X_data, affine, hdr)
 
    #保存nii文件，后面的参数是保存的文件名
    #nib.save(new_nii, 'img/new/coronacases_org_00'+ str(i+8) +'.nii.gz')

    nib.save(new_nii, 'pro_'+org_list)