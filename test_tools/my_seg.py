#根据matlab的思路做的python版的全局阈值分割法，虽然速度比较慢，但分割比较准确！
import numpy as np
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import scipy as sp
from scipy import ndimage
from tqdm import tqdm
import glob



def cul_S(path,n):
    X = nib.load(path)
    X_data = X.get_data()
    z_data = X_data[:,:,n]
    S = np.sum(z_data)
    print('The S of ',n+1,'is:',S)
    return S

# for n in range(33):
#     S = cul_S('test.nii.gz',n)



#阈值分割
def IterationThreshold(img):
    
    eps = 1
    #初始化T
    r, c = img.shape
    T = int(np.sum(img)/(r*c))

    

    while 1:
        G1, G2, cnt1, cnt2 = 0, 0, 0, 0
        for i in range(r):
            for j in range(c):
                if img[i][j] >= T: G1 += img[i][j]; cnt1 += 1
                else: G2 += img[i][j]; cnt2 += 1

        u1 = int(G1 / cnt1) #前景平均值
        u2 = int(G2 / cnt2) #背景平均值
        T2 = (u1 + u2) / 2  #新阈值
        dis = abs(T2 - T)
        if(dis <= eps): break
        else :T = T2
    #print(T)

    new_img = np.zeros((r, c),np.uint8)
    for i in range(r):
        for j in range(c):
            if img[i][j] >= T: new_img[i][j] = 255  
            else: new_img[i][j] = 0
    #print(np.max(new_img),np.min(new_img))
    return new_img




#开运算
def open_cul(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 矩形结构
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return open

#最大连通区域填充
def find_max_region(mask_sel):
    contours,hierarchy = cv2.findContours(mask_sel,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 
    #找到最大区域并填充 
    area = []
 
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
 
    max_idx = np.argmax(area)
 
    max_area = cv2.contourArea(contours[max_idx])
 
    for k in range(len(contours)):
    
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel

#空洞填充
def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

#利用开运算去除小于1000的面积
def delete_1000(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 
    #找到最大区域并填充 
    area = []
    
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
 
    max_idx = np.argmax(area)
 
    max_area = cv2.contourArea(contours[max_idx])
 
    for k in range(len(contours)): 
        if area[k] < 1000:

            cv2.fillPoly(img, [contours[k]], 0)
    return img


def compose_seg_lung(org_path,save_path):
    org_list = sorted(glob.glob(org_path+'*nii.gz'))
    print(org_list)
    
    for num_org in tqdm(range(67, len(org_list))):
        try:  
            X = nib.load(org_list[num_org])
            X_data = X.get_data() 
            r, c, h = X_data.shape
            affine = X.affine.copy()
            hdr = X.header.copy()

            for n in range(h):
                org = X_data[:,:,n]
                img = org.copy()
                img = IterationThreshold(img)
                img = open_cul(img)
                img2 = img.copy()
                img = find_max_region(img)
                img = flood_fill(img)
                out_img = img - img2
                out_img = delete_1000(out_img)
                # cv2.imshow('lung', out_img)
                # cv2.waitKey(10)
                out_img = out_img/255
                X_data[:,:,n] = org*out_img 
            new_nii = nib.Nifti1Image(X_data, affine, hdr)
            nib.save(new_nii, save_path+org_list[num_org][-16:]) 
        except:
            print(org_list[num_org][-16:]+" has error")


compose_seg_lung('nii/','nii_lung/')





# X = nib.load('nii\LNDb-0001.nii.gz')
# X_data = X.get_data() 
# r, c, h = X_data.shape
# affine = X.affine.copy()
# hdr = X.header.copy()
# test = nib.load("nii\LNDb-0001.nii.gz")
# data = test.get_data()
# img = data[:,:,200]
# # cv2.imshow('1', img)
# # cv2.waitKey(1000)
# img = IterationThreshold(img)
# # cv2.imshow('1', img)
# # cv2.waitKey(1000)    #等待毫秒
# img = open_cul(img)
# img2 = img.copy()
# # cv2.imshow('2', img)
# # cv2.waitKey(1000)    #等待毫秒
# img = find_max_region(img)
# cv2.imshow('3', img)
# cv2.waitKey(1000)    #等待毫秒

# img = flood_fill(img)
# cv2.imshow('4', img)
# cv2.waitKey(1000)    #等待毫秒

# cv2.imshow('2', img2)
# cv2.waitKey(1000)    #等待毫秒
# out_img = img - img2  #img2需要copy,不然img2永远指向img
# cv2.imshow('5', out_img)
# cv2.waitKey(1000)    #等待毫秒







# for n in tqdm(range(h)):
#     org = X_data[:,:,n]
#     img = org.copy()
#     img = IterationThreshold(img)
#     img = open_cul(img)
#     img2 = img.copy()
#     img = find_max_region(img)
#     img = flood_fill(img)
#     out_img = img - img2
#     out_img = delete_1000(out_img)
#     cv2.imshow('lung', out_img)
#     cv2.waitKey(10)
#     out_img = out_img/255
#     X_data[:,:,n] = org*out_img

    
#    n += 1
# new_nii = nib.Nifti1Image(X_data, affine, hdr)
# nib.save(new_nii, 'test.nii.gz')



    #cv2.imshow('1', iry)

    

    # avg = 0
    # for i in range(r):
    #     for j in range(c):
    #         avg += iry[i][j]
    #T =int(avg/(r*c))
    #初始化T
#     T = int(np.sum(iry)/(r*c))

#     while 1:
#         G1, G2, cnt1, cnt2 = 0, 0, 0, 0
#         for i in range(r):
#             for j in range(c):
#                 if iry[i][j] >= T: G1 += iry[i][j]; cnt1 += 1
#                 else: G2 += iry[i][j]; cnt2 += 1

#         u1 = int(G1 / cnt1) #前景平均值
#         u2 = int(G2 / cnt2) #背景平均值
#         T2 = (u1 + u2) / 2  #新阈值
#         dis = abs(T2 - T)
#         if(dis <= eps): break
#         else :T = T2



#     new_img = np.zeros((r, c),np.uint8)
#     for i in range(r):
#         for j in range(c):
#             if iry[i][j] >= T: new_img[i][j] = 255  #这里取反设置为1
#             else: new_img[i][j] = 0

#     X_data[:,:,n] = new_img
#     print(n)
#     n += 1
    
# new_nii = nib.Nifti1Image(X_data, affine, hdr)
# nib.save(new_nii, 'test.nii.gz')
#     #cv2.imshow('2', new_img)
#     #cv2.waitKey()
