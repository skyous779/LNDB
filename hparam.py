class hparams:

    train_or_test = 'train'
    debug = False
    mode = '3d' # '2d or '3d'
    out_class = 1
    in_class = 1

    crop_or_pad_size = 200
    patch_size = 64  #每次卷积的大小（batch每次输入patch数目？）

    fold_arch = '*.nii.gz'


    source_train_dir = 'nii_lung_affine/'
    label_train_dir = 'labelsTr/'
    source_test_dir = 'org/new'
    label_test_dir = 'mask'

    output_dir_test = 'results'
