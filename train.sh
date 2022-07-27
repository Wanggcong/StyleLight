
# install
conda create -n StyleLight python=3.7 -y
conda activate StyleLight
pip install lpips
pip install wandb
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install matplotlib
pip install dlib
pip install imageio
pip install einops

sudo apt-get install openexr and libopenexr-dev
pip install OpenEXR

pip install imageio-ffmpeg
pip install ninja



# data pre-process
python data_prepare_laval.py

# train StyleLight
python train.py --outdir=./training-runs-paper512-cyclic-new-training-128x256-ws_plus_coor2-accepted --data=/mnt/disks/data/datasets/IndoorHDRDataset2018-128x256-data-splits/train --gpus=8 --cfg=paper256  --mirror=1 --aug=noaug


# lighting estimation and editing
python test_lighting.py


# evaluation