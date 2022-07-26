import glob
import os
from PIL import Image
import numpy as np
from skylibs.envmap import EnvironmentMap, rotation_matrix
from skylibs.hdrio import imread, imsave


pano_h=128
pano_w=256

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=99, max_mapping=0.99):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img.astype('float32'), alpha


def pre_process_data(src_file, to_folder, phase, file_txt):
    dst_folder = os.path.join(to_folder, phase)
    to_path_crop = to_folder+'test_crop'
    if not os.path.exists(to_path_crop):
      os.mkdir(to_path_crop)

    assert(phase in ['train', 'test'])
    with open(file_txt, 'r') as f:
        data_list = [line.strip() for line in f.readlines()]    
    k=0
    for file_ in data_list:
        file_paths = glob.glob(os.path.join(src_file,file_+'*'))
        print(f'k/total,files:{k}/{len(data_list)},{file_paths}')
        k = k+1
        # if k>2:
        #   break
        for one_path in file_paths:
            e = EnvironmentMap(one_path, 'latlong')
            if phase=='test':
                # tone mapping
                hdr2ldr = TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.9)
                ldr,_ = hdr2ldr(e.data)                
                ee = EnvironmentMap(ldr, 'latlong')
                # crop
                dcm = rotation_matrix(azimuth=0,elevation=0,roll=0)    
                crop = ee.project(vfov=60., # degrees
                                rotation_matrix=dcm,
                                ar=4./3.,
                                resolution=(256, 192),
                                projection="perspective",
                                mode="normal") #for cropping
                                # mode="mask")     # for mask
                im_ = Image.fromarray((crop*255).astype(np.uint8))
                basename = one_path.split('/')[-1].split('.')[0]+'.png'
                im_.save(os.path.join(to_path_crop,basename))

            e.resize((pano_h, pano_w))
            imsave(os.path.join(dst_folder,one_path.split('/')[-1]), e.data)

###########
to_folder = '/mnt/disks/data/datasets/IndoorHDRDataset2018-debug-'+str(pano_h)+'x'+str(pano_w)+'-data-splits2/'
from_file = '/mnt/disks/data/datasets/IndoorHDRDataset2018'

train_txt = 'train_org.txt'
test_txt = 'test_selected.txt'

if not os.path.exists(to_folder):
    os.mkdir(to_folder)
    os.mkdir(os.path.join(to_folder, 'train'))
    os.mkdir(os.path.join(to_folder, 'test'))

pre_process_data(from_file, to_folder, 'train', train_txt)  
pre_process_data(from_file, to_folder, 'test', test_txt)

