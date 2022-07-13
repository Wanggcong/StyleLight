# import glob
import os
# from PIL import Image
# from shutil import copyfile
import random


root_path = '/mnt/disks/data/datasets/IndoorHDRDataset2018-128x256'
files = os.listdir(root_path)
random.shuffle(files)


test_txt = 'test.txt'



with open(test_txt, 'w') as f:
    for name in files[:100]:
        f.write(name + "\n")
f.close()


train_txt = 'train.txt'

with open(train_txt, 'w') as f:
    for name in files[100:]:
        f.write(name + "\n")
f.close()


# with open(file_txt, 'r') as f:
#     data_list = [line.strip() for line in f.readlines()] 