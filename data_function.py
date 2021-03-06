from glob import glob
from os.path import dirname, join, basename, isfile
import sys
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
from torchio.data import WeightedSampler
from torchio.data import LabelSampler
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


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

        if hp.mode == '3d':
            patch_size = hp.patch_size
        elif hp.mode == '2d':
            patch_size = (hp.patch_size,hp.patch_size,1)
        else:
            raise Exception('no such kind of mode!')

        queue_length = 32
        samples_per_volume = 16
    
        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))

            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))   #进行遍历排序

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                
                #原版
                #suject :一份带原片和标签
                # subject = tio.Subject(
                #     source=tio.ScalarImage(image_path),
                #     label=tio.LabelMap(label_path),
                # )
                # #subjects:一个集合   
                # self.subjects.append(subject)

                #第一版修改，失败
                # ct = tio.ScalarImage(image_path)
                # rescale = tio.RescaleIntensity(
                #     out_min_max=(-512, 512))
                # ct_normalized = rescale(ct)

                # #第二版修改
                # ct = tio.ScalarImage(image_path)
                # ct.data = torch.clamp(ct.data,min =-512,max=512)
                
                #第三版修改,对数据集进行一个裁剪，使它能够很好的缩小图像
                
                # subject = tio.Subject(
                #     source=tio.ScalarImage(image_path),
                #     label=tio.LabelMap(label_path),
                # )
                # my_transform = tio.CropOrPad(
                #     (256, 256, 256),
                #     mask_name='label',
                # )
                # transformed = my_transform(subject)
                # self.subjects.append(transformed)
                
                #第四版
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.Image(label_path,type=tio.SAMPLING_MAP),
                )
                #subjects:一个集合   
                self.subjects.append(subject)

        else:#各大分割的label
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))

            artery_labels_dir = Path(labels_dir+'/artery')
            self.artery_label_paths = sorted(artery_labels_dir.glob(hp.fold_arch))

            lung_labels_dir = Path(labels_dir+'/lung')
            self.lung_label_paths = sorted(lung_labels_dir.glob(hp.fold_arch))

            trachea_labels_dir = Path(labels_dir+'/trachea')
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(hp.fold_arch))

            vein_labels_dir = Path(labels_dir+'/vein')
            self.vein_label_paths = sorted(vein_labels_dir.glob(hp.fold_arch))


            for (image_path, artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths,self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    atery=tio.LabelMap(artery_label_path),
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
                )
                self.subjects.append(subject)


        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)  #一个集合

    #修改了采样
        self.queue_dataset = Queue(
            self.training_set,
            queue_length,
            samples_per_volume,
            #UniformSampler(patch_size), #Randomly extract patches from a volume with uniform probability.
            #WeightedSampler(patch_size, 'label'),
            LabelSampler(patch_size,label_name='label',label_probabilities={0: 0.2, 255: 0.8}),
            num_workers = 0,
        )




    def transform(self):   
        if hp.mode == '3d':     
            training_transform = Compose([
            # ToCanonical(),
            #去掉crop,因为自定义了crop对数据进行了预处理
            CropOrPad((hp.crop_or_pad_size, hp.crop_or_pad_size, hp.crop_or_pad_size), mask_name='label',padding_mode='reflect',),
            RandomMotion(),
            RandomBiasField(),
            ZNormalization(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
            Resample((1,1,1)),    #重采样
            OneOf({
                RandomAffine(): 0.8,
                RandomElasticDeformation(): 0.2,
            }),])
            
        elif hp.mode == '2d':
            training_transform = Compose([
            CropOrPad((hp.crop_or_pad_size, hp.crop_or_pad_size,1), padding_mode='reflect'),
            # RandomMotion(),
            RandomBiasField(),
            ZNormalization(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
            #Resample((1,1,1)),
            # OneOf({
            #     RandomAffine(): 0.8,
            #     RandomElasticDeformation(): 0.2,
            # }),
            ])
        else:
            raise Exception('no such kind of mode!')


        return training_transform




class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):


        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            # labels_dir = Path(labels_dir)
            # self.label_paths = sorted(labels_dir.glob(hp.fold_arch))
            
            #for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            for image_path  in  zip(self.image_paths):    
                # ct = tio.ScalarImage(image_path)
                # ct.data = torch.clamp(ct.data,min =-512,max=512)
                
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    #label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)
        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))

            artery_labels_dir = Path(labels_dir+'/artery')
            self.artery_label_paths = sorted(artery_labels_dir.glob(hp.fold_arch))

            lung_labels_dir = Path(labels_dir+'/lung')
            self.lung_label_paths = sorted(lung_labels_dir.glob(hp.fold_arch))

            trachea_labels_dir = Path(labels_dir+'/trachea')
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(hp.fold_arch))

            vein_labels_dir = Path(labels_dir+'/vein')
            self.vein_label_paths = sorted(vein_labels_dir.glob(hp.fold_arch))


            for (image_path, artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths,self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    atery=tio.LabelMap(artery_label_path),
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
                )
                self.subjects.append(subject)


        # self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=None)


    # def transform(self):

    #     training_transform = Compose([
    #     ZNormalization(),
    #     ])
        


    #     return training_transform



