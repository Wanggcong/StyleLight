

import numpy as np
from imageio import imread, imsave
from skylibs.envmap import EnvironmentMap, rotation_matrix
#from envmap import EnvironmentMap, rotation_matrix
import os
from PIL import Image

degrees = 60

def crop2pano(env, image_path):
    # e = EnvironmentMap(image_path, 'latlong')
    # image_path = os.path.join("/home/deep/datasets/3D60/processed/Matterport3D-central-png","0_0b217f59904d4bdf85d35da2cab963471_color_0_Left_Down_0.0.png")  
    # env = EnvironmentMap(image_path, 'latlong')
    # env.resize((128,256))
    
    # dcm = rotation_matrix(azimuth=np.pi/6,
    dcm = rotation_matrix(azimuth=0,
                          # elevation=np.pi/8,
                          elevation=0,
                          # roll=np.pi/12)
                          roll=0)
    projection = False
    #projection = True
    if projection:
        print('env.data size:', env.data.shape)
        crop = env.project(vfov=degrees, # degrees
                         rotation_matrix=dcm,
                         ar=4./3.,
                         # resolution=(640, 480),
                         # resolution=(256, 128),
                         resolution=(256, 192),
                         projection="perspective",
                         # mode="normal") #for cropping
                         mode="mask")     # for mask
        crop = np.clip(255.*crop, 0, 255).astype('uint8')
        imsave("crop60_256x512.jpg", crop, quality=90)
        return
    

    debug = False
    if debug: 
        cropped_img_ = Image.open('crop9.jpg')
        cropped_img = np.array(cropped_img_)
    else:
        cropped_img_ = Image.open(image_path)
        # cropped_img_ = cropped_img_.resize((128, 256), Image.LANCZOS)
        cropped_img = np.array(cropped_img_)
        print('crop image shape:',cropped_img.shape)
        h,w,_ = cropped_img.shape
        # cropped_img = cv2.resize()
    
    # print('cropped_img:',cropped_img)
    masked_pano = env.Fov2MaskedPano(cropped_img,
                     vfov=degrees, # degrees
                     rotation_matrix=dcm,
                     ar=4./3.,
                     # resolution=(640, 480),
                     resolution=(w, h),
                     # resolution=(512, 256),
                     projection="perspective",
                     mode="normal")
                     # mode="mask")

    # masked_pano = np.clip(255.*masked_pano, 0, 255).astype('uint8')
    masked_pano = masked_pano.astype('uint8')
    # imsave("masked_pano14_test.jpg", masked_pano, quality=90)
    # print('masked_pano:',masked_pano.shape)

    return  masked_pano

if __name__ == "__main__":
    crop_path = '/home/deep/Downloads/crop_ldr/9C4A2376-others-160-2.23104-1.10862.jpg'
    # crop_path = 'crop9.jpg'
    
    # e = EnvironmentMap(image_path, 'latlong')
    # image_path = os.path.join("/home/deep/datasets/3D60/processed/Matterport3D-central-png","0_0b217f59904d4bdf85d35da2cab963471_color_0_Left_Down_0.0.png")  
    # env = EnvironmentMap(image_path, 'latlong')
    env = EnvironmentMap(256, 'latlong')
    # env.resize((128,256))
    

    masked_pano = crop2pano(env,crop_path)
    print('masked_pano:',masked_pano.shape)
