'''快速分割肺，但是细节不够好'''
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import scipy.ndimage
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, feature
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import glob


# 路径
# root = os.path.join(os.getcwd(), 'mhd_raw_exp/')
# paths = os.listdir(root)
# print(paths)




# 绘制3D
def plot_3d(image, threshold=-400):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    # p = p[:,:,::-1]

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

# 绘图函数
# 会卡死
def plot_ct_scan(scan, num_column=4, jump=1):
    num_slices = len(scan)
    num_row = (num_slices//jump + num_column - 1) // num_column
    f, plots = plt.subplots(num_row, num_column, figsize=(num_column*5, num_row*5))
    for i in tqdm(range(0, num_row*num_column)):
        plot = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]
        plot.axis('off')
        if i < num_slices//jump:
            plot.imshow(scan[i*jump], cmap=plt.cm.bone)
            plt.show()

# 肺部分割
def get_segmented_lungs(im, spacing, threshold=-400):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < threshold

    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)

    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)

    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)

    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)

    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    return binary

# 分割优化
def extract_main(mask, spacing, vol_limit=[0.68, 8.2]):
    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    label = measure.label(mask, connectivity=1)

    properties = measure.regionprops(label)

    for prop in properties:
        if prop.area * voxel_vol < vol_limit[0] * 1e6 or prop.area * voxel_vol > vol_limit[1] * 1e6:
            mask[label == prop.label] = 0

    return mask

#保存
def save_itk(array, origin, spacing, filename):
    itkimage = sitk.GetImageFromArray(array, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True) 

# data2 = sitk.ReadImage('LNDb-0008.nii.gz')
# print(data2.GetSpacing())
def seg(path,save_path):

    #scan : img的array
    #spacing : 仿射矩阵
    #origin : 头文件
    #path ： nii/*.nii.gz
    nii_list = sorted(glob.glob(path))
    for i in range(len(nii_list)):

        data = sitk.ReadImage(nii_list[i])
        #print(nii_list[i][7:-4])
        #读取信息
        spacing = data.GetSpacing()
        Origin = data.GetOrigin()
        scan = sitk.GetArrayFromImage(data)

        #窗口化
        scan = np.clip(scan,-1000,400)
        scan_max = np.max(scan)
        scan_min = np.min(scan)  
        #print(scan_max,scan_min)  

        # 掩膜
        mask = np.array([get_segmented_lungs(slice.copy(), spacing) for slice in scan])

        #优化
        mask = extract_main(mask, spacing)

        scan[~mask] = scan_min
        scan_max = np.max(scan)
        scan_min = np.min(scan)
        #print(scan_max,scan_min)    #2780 -1024
        scan = (scan-scan_min)/(scan_max-scan_min)*255  #0-255
        
        #提高对比度
        scan = scan = np.clip(scan,0,150)
        scan_max = np.max(scan)
        scan_min = np.min(scan)
        #print(scan_max,scan_min)    #2780 -1024
        scan = (scan-scan_min)/(scan_max-scan_min)*255  #0-255    

        save_itk(scan, Origin, spacing, save_path + nii_list[i][7:-4]+'.nii.gz')

seg('mhd100/*.mhd','nii_lung/')
















'''
#path = paths[0]
data = sitk.ReadImage('LNDb-0008.mhd')
spacing = data.GetSpacing()
Origin = data.GetOrigin()
print(spacing)

scan = sitk.GetArrayFromImage(data)

#已经验证有效
#save_itk(scan, Origin, spacing, 'try_simpleitk_save.nii.gz')  #已经验证有效
print(scan.shape)

# 掩膜
mask = np.array([get_segmented_lungs(slice.copy(), spacing) for slice in scan])

#优化
mask = extract_main(mask, spacing)
scan[~mask] = 0


save_itk(scan, Origin, spacing, 'simpleitk_save.nii.gz')
print(spacing)


# plot_ct_scan(scan, jump=1)

# 调用
#plot_3d(scan)
'''