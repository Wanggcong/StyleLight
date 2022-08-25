from envmap import EnvironmentMap
import os
import numpy as np
import math
# from imageio import imread, imsave
from PIL import Image

# image_path = os.path.join("/media/deep/HardDisk4T-new/datasets/laval-processed/hdrOutputs/","9C4A0010-others-00-1.80587-1.11567.exr")  
image_path = os.path.join("/home/deep/datasets/3D60/processed/Matterport3D-central-png","0_0b217f59904d4bdf85d35da2cab963471_color_0_Left_Down_0.0.png")  
e = EnvironmentMap(image_path, 'latlong')
# e_angular = e.copy().convertTo('angular')
# e_angular_sa = e_angular.solidAngles()


# e_angular = e.copy().convertTo('sphere')


# read
# import pdb
# pdb.set_trace()

#(128, 256, 3)
print('image size:', e.data.shape)

h, w, c = e.data.shape

import cv2
# image_resized = cv2.resize(e.data, )


# import pdb
# pdb.set_trace()


warp_image = np.zeros_like(e.data)

beta = 0.9 # sin(beta)

scale = 4
hh = scale*h
ww = scale*w


image_resized = cv2.resize(e.data, (ww, hh), interpolation = cv2.INTER_AREA)

# compute 
for ii in range(hh):
    for jj in range(ww):
        # i_int = ii//2
        # j_int = jj//2
        pixel = image_resized[ii,jj,:]
        
        # print('pixel:',pixel)
        # i=ii*1.0//2
        # j=jj*1.0//2
        x = np.sin(ii*1.0/hh*math.pi)*np.cos(jj*2.0/ww*math.pi)
        y = np.sin(ii*1.0/hh*math.pi)*np.sin(jj*2.0/ww*math.pi)
        z = np.cos(ii*1.0/hh*math.pi)        

        t = -x*beta+np.sqrt((x**2)*(beta**2)-(beta**2)+1)        

        # intersection
        x_ = x*t+beta
        y_ = y*t
        z_ = z*t        

        # norm
        mag = np.sqrt(x_**2+y_**2+z_**2)
        x_norm = x_/mag
        y_norm = y_/mag
        z_norm = z_/mag        

        # angle (i.j)--> (x_norm, y_norm, z_norm)        

        theta = np.arccos(z_norm)
        phi = np.arctan2(y_norm,x_norm)#+math.pi  

        # print('theta, phi:', theta, phi)      

        theta_index = (theta/math.pi*h).astype(int) 
        phi_index = (phi/(2*math.pi)*w).astype(int)         
        

        warp_image[theta_index,phi_index,:] = pixel
        # print(warp_image)


warp_image = np.clip(warp_image*255, 0, 255)
warp_image = warp_image.astype(np.uint8)  
print('warp_image:',warp_image.max())
# imsave("warp.png", warp_image)

warp_image_ = Image.fromarray(warp_image)

warp_image_.save("warp.png")

