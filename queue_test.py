import numpy as np
from medpy.io import load,save
from hparam import hparams as hp
from utils.metric import metric
from data_function import MedData_train
import torchio as tio
from torch.utils.data import DataLoader

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir

train_dataset = MedData_train(source_train_dir,label_train_dir) #主要用的queue的特征

# #确定各个class
# print(train_dataset.training_set)
# print(train_dataset.subjects)
# print(len(train_dataset.subjects))
# print(train_dataset.queue_dataset.subjects_dataset)
# print(train_dataset.queue_dataset.subjects_dataset)

#尝试WeightedSampler
patch_size = 128
queue_length = 5
samples_per_volume = 5
sampler = tio.data.WeightedSampler(patch_size, 'label')
my_queue_dataset = tio.Queue(
    train_dataset.training_set,
    queue_length,
    samples_per_volume,
    tio.data.WeightedSampler(patch_size, 'label'), #Randomly extract patches from a volume with uniform probability.
)
# print(dir(my_queue_dataset))
# print(my_queue_dataset.patches_list)

train_loader = DataLoader(my_queue_dataset, 
                        batch_size=1, 
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True)
print(train_loader.dataset)
for i, batch in enumerate(train_loader):
    print(type(batch))
                    