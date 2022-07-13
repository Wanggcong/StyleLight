# import shutil
# import glob
import os
from PIL import Image

#panoramas_predict_folder = 'datasets/Matterport3D_512x512'
#panoramas_predict_folder = 'datasets/Matterport3D_128x128'
panoramas_predict_folder = 'datasets/Matterport3D_128x256'
if not os.path.exists(panoramas_predict_folder):
    os.mkdir(panoramas_predict_folder)


train_txt = '/home/guangcongwang/projects/Boundless-in-Pytorch/train_Matterport3D.txt'


root_path = '/home/guangcongwang/projects/Boundless-in-Pytorch/equirectangular/Matterport3D-central-png/'



with open(train_txt, 'r') as f:
    train_list = [line.strip() for line in f.readlines()]


for file in train_list:
    file_path = os.path.join(root_path, file+'.0.png')   #dataset_name
    # list_sample.append(file_path)
    im = Image.open(file_path)
    #size=(512,512)
    #size=(128,128)
    #size=(128,256)# wrong
    size=(256,128)
    im2=im.resize(size,Image.BILINEAR)    

    to_path = os.path.join(panoramas_predict_folder,file+'.0.png')

    im2.save(to_path)


