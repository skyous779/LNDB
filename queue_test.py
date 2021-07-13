import numpy as np
from medpy.io import load,save
from hparam import hparams as hp
from utils.metric import metric
from data_function import MedData_train

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir

train_dataset = MedData_train(source_train_dir,label_train_dir) #主要用的queue的特征
print(train_dataset.queue_dataset.subjects_dataset)


# train_loader = DataLoader(train_dataset.queue_dataset, 
#                         batch_size=args.batch, 
#                         shuffle=True,
#                         pin_memory=True,
#                         drop_last=True)