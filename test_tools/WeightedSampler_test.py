from glob import glob
from pathlib import Path
from hparam import hparams as hp
import torchio as tio
from torchio import Queue

images_dir = hp.source_train_dir
labels_dir = hp.label_train_dir

images_dir = Path(images_dir)
image_paths = sorted(images_dir.glob(hp.fold_arch))

labels_dir = Path(labels_dir)
label_paths = sorted(labels_dir.glob(hp.fold_arch))   #进行遍历排序
subjects = []
patch_size = 32
sampler = tio.data.WeightedSampler(patch_size, 'sampling_map')

transforms = tio.CropOrPad(
    (512,512,512),   #爆内存
)
for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        source=tio.ScalarImage(image_path),
        sampling_map=tio.Image(label_path,type=tio.SAMPLING_MAP),
    )
    # transform = tio.CropOrPad(
    #     (512,512,512),
    #     mask_name="sampling_map",    #爆内存
    # )

    #subject = transform(subject)
    # for patch in sampler(subject,num_patches=1):
    #     print(patch[tio.LOCATION])
    subjects.append(subject)
training_set = tio.SubjectsDataset(subjects, transforms)    

queue_dataset = Queue(
    training_set,
    5,
    5,
    #UniformSampler(patch_size), #Randomly extract patches from a volume with uniform probability.
    sampler,
)
print(training_set)
print(queue_dataset)