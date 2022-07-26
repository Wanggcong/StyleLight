import cv2
import numpy as np
import glob
import imageio 
#img_array = []
#size = (256,512)


sequence_path='save_videos/v1_1_visual/*.png'
video_name='save_videos/project.mp4'

video = imageio.get_writer(f'{video_name}', mode='I', fps=10, codec='libx264', bitrate='16M')
img_fov = imageio.imread(filename)
for filename in sorted(glob.glob(f'{sequence_path}')):
    img = imageio.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    video.append_data(img)                                

video.close()
