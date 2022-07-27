from random import choice
from string import ascii_uppercase
# from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from PTI_utils import global_config, paths_config, hyperparameters
import wandb
from training.coaches.my_coach import MyCoach
from training.coaches.my_editor import MyEditor
from PTI_utils.ImagesDataset import ImagesDataset

import glob




def run_PTI(run_name='', use_wandb=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    # embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    # os.makedirs(embedding_dir_path, exist_ok=True)
    os.makedirs(paths_config.save_image_path, exist_ok=True)


    # dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if not hyperparameters.edit:
        # root_path = '/home/deep/projects/mini-stylegan2/Evaluation/data/ground_truth_ours_neg0.6_60degree_HR/crop_test_high_resolution/*png'
        root_path = '/mnt/disks/data/datasets/IndoorHDRDataset2018-debug-128x256-data-splits2/test_crop/*png'
        dataloader = sorted(glob.glob(root_path))[0:1] # before rebbutal
        # dataloader = sorted(glob.glob(root_path)) 

        # root_path = 'assets/wild2/*jp*g'
        # dataloader = sorted(glob.glob(root_path))

        coach = MyCoach(dataloader, use_wandb)

    else:
        # root_path = 'assets/test_set_light_editing_new/*195*png'
        # root_path = 'assets/test_set_light_editing_new/*205*png'
        root_path = 'assets/test_set_light_editing_new/*279*png'
        dataloader = sorted(glob.glob(root_path))#[:10]
        coach = MyEditor(dataloader, use_wandb)


    coach.train()

    return global_config.run_name


if __name__ == '__main__':  
    run_PTI(run_name='', use_wandb=False)





