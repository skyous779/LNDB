from glob import glob
from os.path import dirname, join, basename, isfile
import sys
from data_function import MedData_train
from numpy.lib.utils import source
sys.path.append('./')
import csv
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path

from hparam import hparams as hp


images_dir = 'source_dataset'
images_dir = Path(images_dir)
image_paths = sorted(images_dir.glob(hp.fold_arch))
# print(image_paths)
# print('-----------------------')

label_dir = 'label_dataset'
label_dir = Path(label_dir)
label_paths = sorted(label_dir.glob(hp.fold_arch))
# print(label_paths)

#测试tio链接dataset和label：
# subjects = []
# for (image_path, label_path) in zip(image_paths, label_paths):
#                 subject = tio.Subject(
#                     source=tio.ScalarImage(image_path),
#                     label=tio.LabelMap(label_path),
#                 )
#                 # print('source: ',tio.ScalarImage(image_path),type(tio.ScalarImage(image_path)))
#                 # print('label: ',tio.LabelMap(label_path),type(tio.LabelMap(label_path)))
#                 # print('subject: ',subject,type(subject))
                
# subjects.append(subject)
# print(subjects)


# images_dir = 'img'
# images_dir = Path(images_dir)
# image_paths = sorted(images_dir.glob(hp.fold_arch))
# print(image_paths)
# print('-----------------------')

# images_dir = 'label'
# images_dir = Path(images_dir)
# image_paths = sorted(images_dir.glob(hp.fold_arch))
# print(image_paths)
# print('-----------------------')

train_dataset = MedData_train(images_dir,label_dir)
# print("train_dataset: ",type(train_dataset),train_dataset) 
print(train_dataset.queue_dataset) 