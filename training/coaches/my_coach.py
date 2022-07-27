import os
import torch
from tqdm import tqdm
from PTI_utils import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from PTI_utils.log_utils import log_images_from_w

from skylibs.envmap import EnvironmentMap
import numpy as np
import PIL.Image
from PIL import Image, ImageDraw
import imageio
import glob

# from skylibs.envmap import EnvironmentMap
from skylibs.demo_crop import crop2pano
from skylibs.hdrio import imread, imsave

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha,tonemapped_img



class MyCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):
        use_ball_holder = True
        is_128x256 = False

        # for fname, image in tqdm(self.data_loader):
        for image_name in tqdm(self.data_loader):
            env = EnvironmentMap(256, 'latlong')
            image_crop2pano = crop2pano(env, image_name)
            print('image max:', image_crop2pano.max())
            image = 2*(image_crop2pano/255.0)-1


            image = torch.tensor(image.transpose([2, 0, 1]), device=global_config.device)#/255.0     ################################### 0-1
            image = image.unsqueeze(0).to(torch.float32)


            self.restart_training()
            name = image_name.split('/')[-1].split('.')[0]

            # mask_fname = '/home/deep/projects/mini-stylegan2/crop10.jpg'
            # mask_fname = '/home/deep/projects/mini-stylegan2/crop60.jpg'
            if is_128x256:
                mask_fname = '/home/deep/projects/mini-stylegan2/crop60.jpg'
            else:
                mask_fname = 'crop60_256x512.jpg'
            mask_pil = PIL.Image.open(mask_fname).convert('RGB')
            use_debug = False
            if use_debug:
                print('###mask_pil size:', np.array(mask_pil).shape)
            
            if is_128x256:
                mask_pil = mask_pil.resize((256, 128), PIL.Image.LANCZOS)
            else:
                mask_pil = mask_pil.resize((512, 256), PIL.Image.LANCZOS)


            mask_pil_sum_c=np.sum(mask_pil,axis=2)
            mask_pil_sum_c_row = np.sum(mask_pil_sum_c,axis=1)
            mask_pil_sum_c_col = np.sum(mask_pil_sum_c,axis=0)
            row_min = np.argwhere(mask_pil_sum_c_row).min()+10 #128
            row_max = np.argwhere(mask_pil_sum_c_row).max()-5

            col_min = np.argwhere(mask_pil_sum_c_col).min()+10 #256
            col_max = np.argwhere(mask_pil_sum_c_col).max()-10  
            
            img1 = ImageDraw.Draw(mask_pil) 
            img1.rectangle([(col_min,row_min),(col_max, row_max)],fill=(255, 0, 0), outline ="red") 

            # im1 = mask_pil.crop((col_min, row_min, col_max, row_max))
            if use_debug:
                mask_pil.save(f'debug_bbox_{name}.png')

            bbox = [row_min, row_max, col_min, col_max] 

            w_pivot = None
            w_pivot = self.calc_inversions(image, name, bbox) ## here
            w_pivot = w_pivot.to(global_config.device)
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            real_images_batch = real_images_batch[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]]
            
            use_first_phase = False
            if use_first_phase:
                generated_images = self.forward(w_pivot)
                is_png = True
                if is_png:
                    generated_images = torch.clip(generated_images, -1, 1)
                    generated_images = (generated_images + 1) * (255/2)
                    generated_images = generated_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    PIL.Image.fromarray(generated_images, 'RGB').save(f'{paths_config.checkpoints_dir}/{name}.png')
                else:
                    gamma = 2.4
                    hdr = torch.clip(generated_images, -1, 1)
                    full = (hdr+1)/2
                    tone = True
                    if tone:
                        full_inv_tonemap = torch.pow(full/5, gamma)
                        img_hdr_np = full_inv_tonemap.permute(0, 2, 3, 1)[0].detach().cpu().numpy() 
                    else:
                        img_hdr_np = full.permute(0, 2, 3, 1)[0].cpu().numpy()


                    imsave(f'{paths_config.checkpoints_dir}/{name}_test.exr', img_hdr_np)


            do_save_image = False
            for i in tqdm(range(hyperparameters.max_pti_steps)):   #max_pti_steps = 350
                generated_images = self.forward(w_pivot)
                
                
                if do_save_image:
                    generated_images_ = (generated_images + 1) * (255/2)
                    generated_images_ = generated_images_.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    PIL.Image.fromarray(generated_images_, 'RGB').save(f'{paths_config.save_image_path}/{hyperparameters.first_inv_steps+i:04}.png')

                generated_images = torch.clip(generated_images, -1, 1)                     ######
                generated_images = generated_images[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]]

                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, name,
                                                               self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:     #LPIPS_value_threshold = 0.06 
                    break

                loss.backward()
                self.optimizer.step()
                
                #locality_regularization_interval = 1     #####
                #training_step = 1
                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0   ##
                
                # image_rec_result_log_snapshot = 100
                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [name])

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            generated_images = self.forward(w_pivot)
            is_png = False # in rebuttal, we use False
            if is_png:
                generated_images = (generated_images + 1) * (255/2)
                generated_images = generated_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                PIL.Image.fromarray(generated_images, 'RGB').save(f'{paths_config.checkpoints_dir}/{name}_test.png')
            else:
                gamma = 2.4
                limited = True

                if limited:
                    generated_images_singlemap = torch.mean(generated_images, dim=1, keepdim=True)                                                                             
                    r_percentile = torch.quantile(generated_images_singlemap,0.999)                                                                                     
                    light_mask = (generated_images_singlemap > r_percentile)*1.0                                                                                    
                    hdr = torch.clip(generated_images*(1-light_mask), -1, 1)+torch.clip(generated_images*light_mask, -1, 2)             

                else:
                    hdr = torch.clip(generated_images, -1, 1)
                
                full = (hdr+1)/2
                inv_tone = True
                if inv_tone:
                    full_inv_tonemap = torch.pow(full/5, gamma)
                    img_hdr_np = full_inv_tonemap.permute(0, 2, 3, 1)[0].detach().cpu().numpy() 
                else:
                    img_hdr_np = full.permute(0, 2, 3, 1)[0].detach().cpu().numpy()


                imsave(f'{paths_config.checkpoints_dir}/{name}_test.exr', img_hdr_np)

            # save video
            if do_save_image:
                sequence_path=f'{paths_config.save_image_path}/*.png' 
                sequences = sorted(glob.glob(f'{sequence_path}'))[150:]
                video_name=paths_config.save_video_path
                video = imageio.get_writer(f'{video_name}', mode='I', fps=25, codec='libx264', bitrate='16M')
                for filename in sequences: 
                    img = imageio.imread(filename) 
                    img_cat = np.concatenate([image_crop2pano, img], axis=1)      #img size (256, 512, 3)
                    video.append_data(img_cat) 

                video.close()






