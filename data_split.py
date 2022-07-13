# import shutil
import glob
import os
from PIL import Image
# from shutil import copyfile
from shutil import copy2


def copy_files(from_file, to_folder, file_txt):
    with open(file_txt, 'r') as f:
        data_list = [line.strip() for line in f.readlines()]    
    
    for file in data_list:
        file_path = glob.glob(os.path.join(from_file,file))
        print('file_path:',file_path)
        if len(file_path)!=1:
        	print('##### wrong file name!')
        	assert(False)
        # copy
        # copyfile(file_path[0], to_folder)
        copy2(file_path[0], to_folder)


to_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018-128x256-data-splits/'

from_file = '/mnt/disks/data/datasets/IndoorHDRDataset2018-128x256'

train_txt = 'train.txt'
test_txt = 'test.txt'

if not os.path.exists(to_folder):
    os.mkdir(to_folder)
    os.mkdir(os.path.join(to_folder, 'train'))
    os.mkdir(os.path.join(to_folder, 'test'))



copy_files(from_file, os.path.join(to_folder, 'train'), train_txt)
copy_files(from_file, os.path.join(to_folder, 'test'), test_txt)