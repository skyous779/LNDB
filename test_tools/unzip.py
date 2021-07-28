#/usr/bin/python
#coding=utf-8
#对mask进行有选择地批量解压
import os,sys  
import zipfile 
import glob
from tqdm import tqdm
#zip_list =  glob.glob('LNDb-00*.zip')

#print(zip_list)
# for n in tqdm(range(len(zip_list))): 
#     print(zip_list[n])
#     zip_file = zipfile.ZipFile(zip_list[n])   
#     for names in zip_file.namelist():
#         zip_file.extract(names,"mhd100/")
#     zip_file.close()
i = 1
zip_file = zipfile.ZipFile("masks.zip")
for names in zip_file.namelist():       
    #print(names[6:],i)
    i += 1
    zip_file.extract(names,"mask/mhd_mask")
    if i == 348:
        zip_file.close()
        break




# open_path='e:\\data'
# save_path='e:\\data' 
# os.chdir(open_path)#转到路径
# #首先，通过zipfile模块打开指定位置zip文件
# #传入文件名列表，及列表文件所在路径，及存储路径
# def Decompression(files,file_path,save_path):
#     os.getcwd()#当前路径
#     os.chdir(file_path)#转到路径
#     for file_name in files:
#         print(file_name)
#         r = zipfile.is_zipfile(file_name)#判断是否解压文件
#         if r:
#             zpfd = zipfile.ZipFile(file_name)#读取压缩文件
#             os.chdir(save_path)#转到存储路径
#             zpfd.extractall()
#             zpfd.close()

# def files_save(open_path):
#     for file_path,sub_dirs,files in os.walk(open_path):#获取所有文件名，路径
#         print(file_path,sub_dirs,files)
#         Decompression(files,file_path,save_path)

# files_save(open_path)