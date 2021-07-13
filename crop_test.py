'''
检查数据预处理后的图片，是否存在数据全黑的情况
'''

import torchio as tio
import SimpleITK as sitk
import nibabel as nib
import numpy as np

# subject = tio.Subject(
#     chest_ct=tio.ScalarImage('org/nii/LNDb-0001.nii.gz'),
#     heart_mask=tio.LabelMap('mask/nii/LNDb-0001_rad1.nii.gz'),
# )
# print(subject)

#测试affine的维度
# X = nib.load('org/nii/LNDb-0001.nii.gz')
# X_mask = nib.load('mask/nii/LNDb-0001_rad1.nii.gz')
# Xdata = X.get_data
# print(Xdata)
# print(Xdata)



#一下保存两种不同的crop结果
# transform = tio.CropOrPad(
#     (180, 180, 180),
#     mask_name='heart_mask',
# )
# transformed = transform(subject)
# print(transformed)
# # print(transformed.chest_ct.shape)
# # print(transformed.chest_ct.affine.shape)
# # print(transformed.heart_mask.affine)

# affine = transformed.chest_ct.affine.copy()
# ct = transformed.chest_ct.data.numpy().copy().squeeze()
# print(ct.shape)
# mask = transformed.heart_mask.data.numpy().copy().squeeze()
# ct_nii = nib.Nifti1Image(ct, affine)
# mask_nii = nib.Nifti1Image(mask, affine)
# nib.save(ct_nii, 'ct1.nii.gz')
# nib.save(mask_nii, 'mask.nii.gz')

from pathlib import Path
from glob import glob
from hparam import hparams as hp
def check_zeros(images_dir,labels_dir):
    images_dir = Path(images_dir)
    image_paths = sorted(images_dir.glob(hp.fold_arch))
    print(image_paths)

    labels_dir = Path(labels_dir)
    label_paths = sorted(labels_dir.glob(hp.fold_arch))   #进行遍历排序
    print(label_paths)

    i = 0
    for (image_path, label_path) in zip(image_paths, label_paths):
        
        #原版
        subject = tio.Subject(
            source=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path),
        )
        my_transform = tio.CropOrPad(
            (256, 256, 256),
            mask_name='label', #以label为中心进行剪切
        )
        transformed = my_transform(subject)
        overcoming = np.any(transformed.label.data.numpy())
        i += 1
        print(i,':',overcoming)  

check_zeros(hp.source_train_dir,hp.label_train_dir)

