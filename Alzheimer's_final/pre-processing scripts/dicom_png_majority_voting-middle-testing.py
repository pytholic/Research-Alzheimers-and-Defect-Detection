#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pydicom as dicom
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import imageio
import numpy as np
import re
import nums_from_string


# In[40]:


CATEGORIES = ['AD', 'CN', 'MCI']
inDir = 'F:/LAB WORK/RIT/LAB/dataset_large/dicom_data/dicom_test/'
outDir = 'F:/LAB WORK/RIT/LAB/dataset_large/png_data3/test_scanwise/'


# In[41]:


for categ in CATEGORIES:
    folders = os.listdir(inDir + categ + '/')
    print(len(folders))


# In[42]:


categ = CATEGORIES[0]
i = 1
folders = os.listdir(inDir + categ + '/')
print(folders)


# In[13]:


categ = CATEGORIES[0]
i = 1

folders = os.listdir(inDir + categ + '/')
# print(folders)
# print(len(folders))
for folder in folders:
    PATH = os.path.join(inDir, categ, folder)
    folder_2 = os.listdir(PATH)
    for folder in folder_2:
        PATH_2 = os.path.join(PATH, folder)
        folder_3 = os.listdir(PATH_2)
        for folder in folder_3:
            PATH_3 = os.path.join(PATH_2, folder)
            folder_4 = os.listdir(PATH_3)
            print(folder_4)
            for folder in folder_4:
                PATH_4 = os.path.join(PATH_3, folder)
                print(PATH_4)
                images = [os.path.basename(x) for x in glob.glob(PATH_4 + '/*.dcm')]
                for f in images:
                    split = int(f.split('_')[13])
                    print(split)
                break
            break
        break
    break


# In[26]:


categ = CATEGORIES[0]
i = 1

folders = os.listdir(inDir + categ + '/')
# print(folders)
# print(len(folders))
for folder in folders:
    PATH = os.path.join(inDir, categ, folder)
    folder_2 = os.listdir(PATH)
    for folder in folder_2:
        PATH_2 = os.path.join(PATH, folder)
        folder_3 = os.listdir(PATH_2)
        for folder in folder_3:
            PATH_3 = os.path.join(PATH_2, folder)
            folder_4 = os.listdir(PATH_3)
            print(folder_4)
            for folder in folder_4:
                PATH_4 = os.path.join(PATH_3, folder)
                print(PATH_4)
                images = [os.path.basename(x) for x in glob.glob(PATH_4 + '/*.dcm')]
                for f in images:
                    nums = nums_from_string.get_nums(f)
                    idx = int(nums[4])
                    print(idx)
#                     split = re.split('_', f)
#                     split_2 = int(split[13])
#                     print(split[13])
                break
            break
        break
    break


# There is issue with split methods. Below method works fine.

# In[45]:


categ = CATEGORIES[2]
i = 1
#os.mkdir(outDir + categ + '/' + f'{i}')
folders = os.listdir(inDir + categ + '/')
for folder in folders:
    PATH = os.path.join(inDir, categ, folder)
    folder_2 = os.listdir(PATH)
    for folder in folder_2:
        PATH_2 = os.path.join(PATH, folder)
        folder_3 = os.listdir(PATH_2)
        for folder in folder_3:
            PATH_3 = os.path.join(PATH_2, folder)
            folder_4 = os.listdir(PATH_3)
            for folder in folder_4:
                PATH_4 = os.path.join(PATH_3, folder)
                os.mkdir(outDir + categ + '/' + f'{i}')
                out_path = outDir + categ + '/' + f'{i}'
                images = [os.path.basename(x) for x in glob.glob(PATH_4 + '/*.dcm')]
                for f in images:
                    nums = nums_from_string.get_nums(f)
                    idx = int(nums[4])
                    if idx > 30 and idx < 70:
#                     split = re.split('_', f)
#                     split_2 = int(split[13])
                    #split = int(f.split('_')[13])
                    #print(split)
                    #if split > 30 and split < 70:
                        ds = dicom.read_file(PATH_4 + '/' + f) # read dicom image
                        img = ds.pixel_array # get image array
                        img = np.flip(img, axis=0)
                        imageio.imwrite(out_path + '/' + f.replace('.dcm','.png'), img)
                i += 1


# In[7]:


categ = CATEGORIES[0]
i = 1
#os.mkdir(outDir + categ + '/' + f'{i}')
folders = os.listdir(inDir + categ + '/')
for folder in folders:
    PATH = os.path.join(inDir, categ, folder)
    folder_2 = os.listdir(PATH)
    for folder in folder_2:
        PATH_2 = os.path.join(PATH, folder)
        folder_3 = os.listdir(PATH_2)
        for folder in folder_3:
            PATH_3 = os.path.join(PATH_2, folder)
            folder_4 = os.listdir(PATH_3)
            for folder in folder_4:
                PATH_4 = os.path.join(PATH_3, folder)
                os.mkdir(outDir + categ + '/' + f'{i}')
                out_path = outDir + categ + '/' + f'{i}'
                images = [os.path.basename(x) for x in glob.glob(PATH_4 + '/*.dcm')]
                for f in images:
                    ds = dicom.read_file(PATH_4 + '/' + f) # read dicom image
                    img = ds.pixel_array # get image array
                    img = np.flip(img, axis=0)
                    imageio.imwrite(out_path + '/' + f.replace('.dcm','.png'), img)
                i += 1


# In[ ]:




