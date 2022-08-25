from skylibs.envmap import EnvironmentMap
import os
import numpy as np
import math
# from imageio import imread, imsave
from PIL import Image
import glob

from skylibs.hdrio import imread, imsave

# image_path = os.path.join("/media/deep/HardDisk4T-new/datasets/laval-processed/hdrOutputs/","9C4A0010-others-00-1.80587-1.11567.exr")  
#image_path = os.path.join("/home/deep/datasets/3D60/processed/Matterport3D-central-png","0_0b217f59904d4bdf85d35da2cab963471_color_0_Left_Down_0.0.png")  
image_paths =   glob.glob('/home/deep/projects/mini-stylegan2/Evaluation/data/tone/out_test8_hdr_split_00045_1000step/*exr') 

to_paths = '/home/deep/projects/mini-stylegan2/Evaluation/data/tone/out_test8_hdr_split_00045_1000step_warp/'
if not os.path.exists(to_paths):
    os.mkdir(to_paths)


for image_path in image_paths:
    e = EnvironmentMap(image_path, 'latlong')
    #(128, 256, 3)
    # print('image size:', e.data.shape)

    h, w, c = e.data.shape

    warp_image = np.zeros_like(e.data)
    beta = 0.7 # sin(beta)
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


    for ii in range(h):
        for jj in range(w):
            warp_image[ii,jj,:] = e.data[theta_index[ii,jj]-1, phi_index[ii,jj]-1, :]

    is_png = False
    if is_png:
        warp_image = np.clip(warp_image*255, 0, 255)
        warp_image = warp_image.astype(np.uint8)  
        print('warp_image:',warp_image.max())
        # imsave("warp.png", warp_image)

        # 0 degree is the largest, move it to the center
        warp_image = np.concatenate((warp_image[:,w//2:,:], warp_image[:,:w//2,:]), axis=1)

        # save
        warp_image_ = Image.fromarray(warp_image)
        warp_image_.save("warp_7.png")
        # print('theta, phi:', max(theta_s), max(phi_s), min(theta_s), min(phi_s)) 
    else:
        target_new_name= image_path.split('/')[-1].split('.')[0]
        imsave(f'{to_paths}/{target_new_name}_warp.exr', warp_image)
