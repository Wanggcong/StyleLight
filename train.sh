#python3 train.py --outdir=./training-runs-paper512-cyclic --data=datasets/Matterport3D_512x512.zip --gpus=8 --cfg=paper512 --aug=noaug --mirror=1 --resume training-runs-paper512/00003-Matterport3D_512x512-mirror-paper512-noaug/network-snapshot-011468.pkl 
#python3 train.py --outdir=./training-runs-paper512-cyclic-new-training --data=datasets/Matterport3D_512x512.zip --gpus=8 --cfg=paper512 --aug=noaug --mirror=1 #--resume training-runs-paper512/00003-Matterport3D_512x512-mirror-paper512-noaug/network-snapshot-011468.pkl 
#python3 train.py --outdir=./training-runs-paper512-cyclic-new-training --data=datasets/Matterport3D_512x512.zip --gpus=8 --cfg=paper512  --mirror=1 --augpipe=cyclic

#python3 train.py --outdir=./training-runs-paper512-cyclic-new-training --data=datasets/Matterport3D_512x512.zip --gpus=8 --cfg=paper512  --mirror=1 --augpipe=cyclic --resume=training-runs-paper512/00003-Matterport3D_512x512-mirror-paper512-noaug/network-snapshot-011468.pkl

python3 train.py --outdir=./training-runs-paper512-cyclic-new-training-full --data=datasets/Matterport3D_512x512.zip --gpus=8 --cfg=paper512  --mirror=1 --aug=noaug


CUDA_VISIBLE=6,7 python3 train.py --outdir=./training-run-paper256_2 --data=datasets/Matterport3D_128x256.zip --gpus=2 --cfg=paper256_2  --mirror=1 --aug=noaug
CUDA_VISIBLE=6,7 python3 train.py --outdir=./training-runs-paper512-cyclic-new-training-128x256-ws_plus_coor2-accepted --data=/mnt/disks/data/datasets/IndoorHDRDataset2018-128x256-data-splits/train --gpus=2 --cfg=paper256_2  --mirror=1 --aug=noaug