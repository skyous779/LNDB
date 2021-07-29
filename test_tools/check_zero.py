#检测数据像素是否有全部为1的图片
from pathlib import Path
from glob import glob

import numpy as np
import nibabel as nib


image_paths = sorted(glob('nii_lung_affine/*.nii.gz'))
print(image_paths)

for i in range(len(image_paths)):
    X = nib.load(image_paths[i])
    X_data = X.get_data() 
    print("max:",np.max(X_data),"min:",np.min(X_data))




