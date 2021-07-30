class hparams:

    train_or_test = 'train'
    debug = False
    mode = '3d' # '2d or '3d'
    out_class = 1
    in_class = 1

    crop_or_pad_size = 128
    patch_size = 64  #每次卷积的大小（batch每次输入patch数目？）

    fold_arch = '*.nii.gz'


    source_train_dir = '/home/workspace/LNDB/imageTr_2/'
    label_train_dir = '/home/workspace/LNDB/labelsTr/'
    source_test_dir = 'imagesTs_crop/'
    label_test_dir = 'labelsTs_crop/'

    output_dir_test = 'results'
