#检测数据像素是否有全部为1的图片

from pathlib import Path
from glob import glob
from hparam import hparams as hp
import torchio as tio
import numpy as np

images_dir = 'mask/nii/'
images_dir = Path(images_dir)
image_paths = sorted(images_dir.glob(hp.fold_arch))
print(image_paths)

i = 0
for (image_path) in zip(image_paths):
    image=tio.LabelMap(image_path)
    overcoming = np.any(image.data.numpy())
    i += 1
    print(image_path,':',overcoming)      
