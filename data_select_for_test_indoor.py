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
        file_paths = glob.glob(os.path.join(from_file,file+'*'))
        print('len(file_paths), file:',len(file_paths), file)
        # if len(file_paths)!=1:
        # 	print('##### wrong file name!')
        # 	assert(False)
        # copy
        file_paths = sorted(file_paths)
        copy2(file_paths[0], to_folder)
        if False:
            for one_path in file_paths:
                # copy2(file_paths[0], to_folder)
                copy2(one_path, to_folder)
        
        # write to .txt
        if False:
            selected_name_str = file_paths[0].split('/')[-1].split('.')[0] 

            with open('test_selected.txt', 'a') as f:
                f.write(selected_name_str+'\n')





###########
# to_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018-256x512-data-splits2/'

# from_file = '/mnt/disks/data/datasets/IndoorHDRDataset2018-128x256'
# from_file = '/mnt/disks/data/datasets/IndoorHDRDataset2018-256x512'
###########

# to_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018_crop_from_origin_png_split2'
# from_file = '/mnt/disks/data/datasets/IndoorHDRDataset2018_crop_from_origin_png'


# to_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018_crop_from_origin_png_split2/test_select'
# from_file = '/mnt/disks/data/datasets/IndoorHDRDataset2018_crop_from_origin_png_split2/test'


to_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018-256x512-data-splits2/test_select'
from_file = '/mnt/disks/data/datasets/IndoorHDRDataset2018-256x512-data-splits2/test'



# train_txt = 'train_org.txt'
test_txt = 'test_org.txt'

if not os.path.exists(to_folder):
    os.mkdir(to_folder)
    # os.mkdir(os.path.join(to_folder, 'train'))
    # os.mkdir(os.path.join(to_folder, 'test'))



# copy_files(from_file, os.path.join(to_folder, 'train'), train_txt)
copy_files(from_file, to_folder, test_txt)



