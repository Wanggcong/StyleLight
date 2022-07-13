# from skylibs.envmap import EnvironmentMap, rotation_matrix
from envmap import EnvironmentMap, rotation_matrix
import os
import numpy as np
import math
# from imageio import imread, imsave
from PIL import Image
import glob
import cv2

# from skylibs.hdrio import imread, imsave
from hdrio import imread, imsave

# image_path = os.path.join("/media/deep/HardDisk4T-new/datasets/laval-processed/hdrOutputs/","9C4A0010-others-00-1.80587-1.11567.exr")  
#image_path = os.path.join("/home/deep/datasets/3D60/processed/Matterport3D-central-png","0_0b217f59904d4bdf85d35da2cab963471_color_0_Left_Down_0.0.png")  
# image_paths =   glob.glob('/home/deep/projects/mini-stylegan2/Evaluation/data/ground_truth_ours_has_blacks/test_not_warp/*exr') 
image_paths =   glob.glob('/mnt/disks/data/datasets/IndoorHDRDataset2018/*exr') 

# to_path_warp = '/home/deep/projects/mini-stylegan2/Evaluation/data/ground_truth_ours/test_warp2/'
# to_path_crop = '/home/deep/projects/mini-stylegan2/Evaluation/data/ground_truth_ours/test_crop2'
to_path_crop = '/mnt/disks/data/datasets/IndoorHDRDataset2018_crop_from_origin'
# if not os.path.exists(to_path_warp):
#     os.mkdir(to_path_warp)

if not os.path.exists(to_path_crop):
    os.mkdir(to_path_crop)


for image_path in image_paths:
    env = EnvironmentMap(image_path, 'latlong')
    #(128, 256, 3)
    # print('image size:', e.data.shape)

    '''
    #######################################################################
    h, w, c = env.data.shape

    warp_image = np.zeros_like(env.data)
    beta = 0.3 # sin(beta)
    scale = 1


    theta_s, phi_s = [], []

    # compute 

    x=np.zeros((h,w))
    y=np.zeros((h,w))
    z=np.zeros((h,w))
    for ii in range(h):
        for jj in range(w):
            # direction of original coor
            x[ii,jj] = np.sin(ii*1.0/h*math.pi)*np.cos(jj*2.0/w*math.pi)#+beta-beta
            y[ii,jj] = np.sin(ii*1.0/h*math.pi)*np.sin(jj*2.0/w*math.pi)
            z[ii,jj] = np.cos(ii*1.0/h*math.pi) 


    a = x**2+y**2+z**2
    b = 2*x*beta
    c = beta**2-1  

    t = (-b+np.sqrt(b**2-4*a*c))/(2*a)      
    # print('t:',t)

    # # intersection
    x_ = x*t+beta #-beta
    y_ = y*t
    z_ = z*t 


    # # norm
    mag = np.sqrt(x_**2+y_**2+z_**2)
    x_norm = x_/mag
    y_norm = y_/mag
    z_norm = z_/mag 


    # angle (i.j)--> (x_norm, y_norm, z_norm)        

    theta = np.arccos(z_norm)
    phi = np.arctan2(y_norm,x_norm)#-math.pi/2

    theta_index = (theta/math.pi*h).astype(int) 
    phi_index = (phi/(2*math.pi)*w).astype(int) 

    
    # remove black pixels
    remove_black_pixels = True
    if remove_black_pixels:
        img_hdr = env.data[:107,:,:]
        img_hdr = cv2.resize(img_hdr, (256,128))

        # img_hdr = np.zeros_like(env.data)
        # img_hdr[:107,:,:] = env.data[:107,:,:]
        # img_hdr[107:,:,:] = env.data[86:107,:,:]

        # img_hdr = cv2.resize(img_hdr, (256,128))
    else:
        img_hdr = env.data
    

    for ii in range(h):
        for jj in range(w):
            warp_image[ii,jj,:] = img_hdr[theta_index[ii,jj]-1, phi_index[ii,jj]-1, :]

    image_warp_name = os.path.join(to_path_warp, image_path.split('/')[-1])
    imsave(image_warp_name, warp_image) 

    # target_new_name= image_path.split('/')[-1].split('.')[0]
    # imsave(f'{to_paths}/{target_new_name}_warp.exr', warp_image)
    '''
    #########################################################################
    #######################################################################

    # center crop
    dcm = rotation_matrix(azimuth=0,
                          # elevation=np.pi/8,
                          elevation=0,
                          # roll=np.pi/12)
                          roll=0)    

    # print('env.data size:', env.data.shape)
    crop = env.project(vfov=60., # degrees
                     rotation_matrix=dcm,
                     ar=4./3.,
                     # resolution=(640, 480),
                     # resolution=(256, 128),
                     resolution=(256, 192),
                     projection="perspective",
                     mode="normal") #for cropping
                     # mode="mask")     # for mask
    # crop = np.clip(255.*crop, 0, 255).astype('uint8')
    
    image_crop_name = os.path.join(to_path_crop, image_path.split('/')[-1])
    imsave(image_crop_name, crop) 