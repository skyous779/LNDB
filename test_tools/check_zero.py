#检测数据像素是否有全部为1的图片
from pathlib import Path
from glob import glob

import numpy as np
import nibabel as nib


image_paths = sorted(glob('/home/workspace/LNDB/2D_train/imagesTr/*.nii.gz'))
print(len(image_paths))

for i in range(len(image_paths)):
    X = nib.load(image_paths[i])
    X_data = X.get_data() 
    print('name:',image_paths[i],"max:",np.max(X_data),"min:",np.min(X_data),'shape:',X_data.shape,X.affine.copy())




