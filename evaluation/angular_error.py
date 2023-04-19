# python angular_error.py --fake test5_tone/diffuse --real diffuse


import numpy as np
import hdrio
import os
import random
import argparse
import torch, torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--fake', help='test data. eg:test5_tone/mirror')
parser.add_argument('--real', help='mirror / matte_silver / diffuse')

def angular(pred, gt):
    h = pred.shape[0]
    w = pred.shape[1]
    mask = np.zeros((h, w))
    mask[pred[:,:,3]>0] = 1
    num_pixel = mask.sum()

    cos = np.sum(gt[:,:,:3]*pred[:,:,:3], axis=2) / (np.linalg.norm(gt[:,:,:3],axis=2) * np.linalg.norm(pred[:,:,:3],axis=2))
    cos = np.clip(cos, 0, 1)
    angular=np.nan_to_num(np.multiply(np.arccos(cos), mask))

    res=np.degrees(angular.sum()/num_pixel)

    return res

args = parser.parse_args()

fake_path = './data/render_results/' + args.fake + '/'
real_path = './data/ground_truth/test_tone_'+args.real+'/'
fake_imgs = os.listdir(fake_path)
fake_imgs.sort()

real_imgs = os.listdir(real_path)
random.shuffle(real_imgs)

error=0
cnt=0
for i, fake in enumerate(fake_imgs):

    pred = hdrio.imread(fake_path+fake)
    real = fake.replace("_fake_image","")
    gt = hdrio.imread(real_path+real)

    a=angular(pred,gt)
    error += a
    cnt += 1

print("Angular error:")
print(error/cnt)

