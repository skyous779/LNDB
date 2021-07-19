class hparams:

    train_or_test = 'test'
    debug = False
    mode = '3d' # '2d or '3d'
    out_class = 1
    in_class = 1

    crop_or_pad_size = 512
    patch_size = 64   #每次卷积的大小（batch每次输入patch数目？）

    fold_arch = '*.nii.gz'


    source_train_dir = 'org/nii'
    label_train_dir = 'mask/nii'
    source_test_dir = 'org/nii'
    label_test_dir = 'mask/nii'

    output_dir_test = 'results'
