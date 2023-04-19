import util
import os
import hdrio
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--testdata', help='test data. eg:./data/original/out_test5_standard_exr/')
parser.add_argument('--out_dir', help='data dir after tonemap. eg: test5_tone')
# parser.add_argument('--flip', help='data dir after tonemap. eg: test5_tone')

args = parser.parse_args()

#test_path = '/home/ynyang/Downloads/blender-cli-rendering-master/emlight/'
# test_path = './data/'+args.testdata+'/'
test_path = args.testdata

test_imgs = os.listdir(test_path)
test_imgs.sort()
is_subset = False
if is_subset:
    test_imgs = test_imgs[:20]
tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
handle = util.PanoramaHandler()

is_toned = True
#is_toned = False
flip = False
for i in test_imgs:
    # exr = handle.read_hdr(test_path+i)
    exr = handle.read_hdr(os.path.join(test_path,i))
    if is_toned:
        img, alpha = tone(exr,clip=False)
    else:
        img, alpha = tone(exr,clip=False,alpha=1.0, gamma=False)
    print('alpha:',alpha)
    if flip:
        # print('img shape #$#:', img.shape) img shape #$#: (128, 256, 3)
        h, w, c = img.shape
        img = np.concatenate((img[:,w//2:,:],img[:,:w//2,:]), axis=1)

    outdir = "./data/tone/"+args.out_dir+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    hdrio.imsave(outdir+i,img)
    # crop.save(outdir + i.replace('exr','jpg'))
