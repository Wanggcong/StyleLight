



import os
from PIL import Image
from envmap import EnvironmentMap
import numpy as np
from hdrio import imsave

#to_path_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018-256x512'
to_path_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018-512x1024'
if not os.path.exists(to_path_folder):
    os.mkdir(to_path_folder)


# train_txt = '/home/guangcongwang/projects/Boundless-in-Pytorch/train_Matterport3D.txt'
root_path = '/mnt/disks/data/datasets/IndoorHDRDataset2018'


# with open(train_txt, 'r') as f:
#     train_list = [line.strip() for line in f.readlines()]

train_list = sorted(os.listdir(root_path))


for file in train_list:
    file_path = os.path.join(root_path, file)   #dataset_name
    # list_sample.append(file_path)
    # im = Image.open(file_path)
    # size=(512,512)
    # im2=im.resize(size,Image.BILINEAR)    

    e = EnvironmentMap(file_path, 'latlong')
    # image = e.data

    if False:
        gamma=2.4
        image = np.clip(image,1e-10,1e8)
        image = np.power(image, 1 / gamma)*5.0#*255
        # image = np.clip(image,0,255)
        image = np.clip(image,0,1)
        im_ = Image.fromarray((image*255).astype(np.uint8))


        to_path = os.path.join(to_path_folder, file.split('.')[0]+'.png')

        im_.save(to_path)
    else:
        e.resize((512, 1024))
        imsave(os.path.join(to_path_folder,file), e.data)
        # imsave("crop.jpg", e, quality=90)



    # e = EnvironmentMap(os.path.join(self._path, fname), 'latlong')
    # image = e.data
    # print('image:', image)
    # lo, hi = image.min(),image.max()
    # print('############## dataset image lo, hi:',lo, hi) #

    # gamma=2.4
    # image = np.clip(image,1e-10,1e8)
    # image = np.power(image, 1 / gamma)*5.0#*255
    # # image = np.clip(image,0,255)
    # image = np.clip(image,0,1)

    # im_ = Image.fromarray((image*255).astype(np.uint8))
    # image=image*2-1


