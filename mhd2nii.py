import SimpleITK as sitk
import glob
from tqdm import tqdm

#转化数据

mhd_list =  glob.glob('mask/LNDb*.mhd')
print(mhd_list)
for i in tqdm(range(len(mhd_list))):  #tqdm

    img = sitk.ReadImage(mhd_list[i])
    sitk.WriteImage(img, 'mask/nii/'+mhd_list[i][5:19]+'.nii.gz')

nii_list = glob.glob('mask/nii/*.nii.gz')
print(nii_list)

