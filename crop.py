import nibabel as nib
import numpy as np
import glob

mask_list = sorted(glob.glob('mask/mask_merge/*nii.gz'))
org_list = sorted(glob.glob('/*nii.gz'))
assert len(mask_list) == len(org_list),'wocao'+'len(org) ~= len(mask)'

def my_clip(data,mask):
    x,y,z = data.shape
    ####################z##################
    for i in range(z):
        #print(i)
        z_sum = np.sum(data[:,:,i])
        if z_sum > 150000:
            break
    z1 = i


    for i in range(z):
        #print(i)
        z_sum = np.sum(data[:,:,-i-1])
        if z_sum > 150000:
            break
    z2 = -i-1

    ######################x#####################
    for i in range(x):
        #print(i)
        x_sum = np.sum(data[i,:,:])
        if x_sum > 0:
            break
    x1 = i


    for i in range(x):
        #print(i)
        x_sum = np.sum(data[-i-1,:,:])
        if x_sum > 0:
            break
    x2 = -i-1

    #################y############
    for i in range(y):
        #print(i)
        y_sum = np.sum(data[:,i,:])
        if y_sum > 0:
            break
    y1 = i
    for i in range(y):
        #print(i)
        y_sum = np.sum(data[:,-i-1,:])
        if y_sum > 0:
            break
    y2 = -i-1

    data = data[x1:x2,y1:y2,z1:z2]
    mask = mask[x1:x2,y1:y2,z1:z2]
    return data,mask




for i in range(len(mask_list)):
    mask = nib.load(mask_list[i])
    org  = nib.load(org_list[i])

    org_affine = org.affine.copy()
    org_hdr = org.header.copy()

    mask_affine = mask.affine.copy()
    mask_hdr = mask.header.copy()

    assert mask_affine == org_affine, org_list[i]+':仿射矩阵有误'

    mask_data = mask.get_data()
    org_data = org.get_data()
    org_data,mask_data = my_clip(org_data,mask_data)

    new_org = nib.Nifti1Image(org_data, org_affine, org_hdr)
    new_mask = nib.Nifti1Image(mask_data, mask_affine, mask_hdr)

    nib.save(new_org, '保存地址')
    nib.save(new_mask,'保存地址')




################################debug######################
'''
X = nib.load("org/new/LNDb-0001.nii.gz")
# clip = nib.load("org/new/clip.nii.gz")
# print(X)
# print(clip.spacing)
a,b,c = [abs(X.affine[0,0]),abs(X.affine[1,1]),abs(X.affine[2,2])]
print(a,b,c)
affine = X.affine.copy()
hdr = X.header.copy()
X_data = X.get_data()
x,y,z = X_data.shape

print(x,y,z)
data,mask = my_clip(X_data,mask)
print(data.shape)
# new_nii = nib.Nifti1Image(X_clip_data, affine, hdr)
# nib.save(new_nii, "org/new/clip.nii.gz")
'''