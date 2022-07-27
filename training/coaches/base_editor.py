import abc
import os
import pickle
from argparse import Namespace
import wandb
import os.path
from PTI_utils.localitly_regulizer import Space_Regulizer
import torch
from torchvision import transforms
from lpips import LPIPS
from training.projectors import w_projector
from PTI_utils import global_config, paths_config, hyperparameters
from PTI_utils import l2_loss
from PTI_utils.e4e.psp import pSp                    #########
from PTI_utils.log_utils import log_image_from_w      ########
from PTI_utils.models_utils import toogle_grad, load_old_G

import numpy as np
import PIL.Image

class BaseEditor:
    def __init__(self, data_loader, use_wandb):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0

        if hyperparameters.first_inv_type == 'w+':   # first_inv_type = 'w'
            self.initilize_e4e()

        self.e4e_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G()
        toogle_grad(self.G, True)

        self.original_G = load_old_G()

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversion(self, w_path_dir, image_name, image):
        # embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        # os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            # torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        if hyperparameters.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}/0.pt'
        else:
            w_potential_path = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}/0.pt'
        if not os.path.isfile(w_potential_path):
            return None
        w = torch.load(w_potential_path).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image, image_name, bbox=None):
        if hyperparameters.first_inv_type == 'w+':
            w = self.get_e4e_inversion(image)

        else:
            id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            if not hyperparameters.edit:
                w = w_projector.project(self.G, bbox, id_image, device=torch.device(global_config.device), w_avg_samples=600,
                                        num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                        use_wandb=self.use_wandb)                                       ####first_inv_steps =450
            else:
                w = w_projector.edit(self.G, bbox, id_image, device=torch.device(global_config.device), w_avg_samples=600,
                                        num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                        use_wandb=self.use_wandb)                  
        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)

            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips




    def calc_loss_new(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch, temp_csv):
        loss = 0.0

        mask = torch.ones_like(generated_images)
        for obj in temp_csv:
            bbox = obj[1]
            mask[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]] = 0        

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images*mask, real_images*mask)

            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images*mask, real_images*mask)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val
        
        use_light_editing = True
        loss_edit = 0.0
        if use_light_editing:
            for obj in temp_csv:
                assert(obj[0] in ['light', 'non_light', 'strong_light'])
                bbox = obj[1]
                if obj[0]=='light':
                    loss_edit_ = self.calc_light_loss_remove_one_light(generated_images, real_images, log_name, new_G, use_ball_holder, w_batch, obj[1])
                else:# obj[0]=='non_light' or obj[0]=='strong_light':
                    loss_edit_ = self.calc_light_loss_add_one_light(generated_images, real_images, log_name, new_G, use_ball_holder, w_batch, obj[1])
                
                loss_edit += loss_edit_

            loss +=0.01*loss_edit/len(temp_csv)


        return loss, l2_loss_val, loss_lpips, mask


    def calc_loss_finetune(self, generated_images_, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch, mask):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            ratio = mask.sum()/(mask.shape[2]*mask.shape[3])
            l2_loss_val = l2_loss.l2_loss(generated_images_*mask, generated_images*mask)/ratio+l2_loss.l2_loss(generated_images_*(1-mask), real_images*(1-mask))/(1-ratio)

            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images_*mask, generated_images*mask)+self.lpips_loss(generated_images_*(1-mask), real_images*(1-mask))
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        return loss


    def calc_light_loss_remove_all_lights(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:

            r_percentile = torch.quantile(real_images, hyperparameters.percentile)
            light_mask = (real_images > r_percentile)*1.0
            l2_loss_val = torch.mean(generated_images*light_mask)

            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda

        
        if hyperparameters.pt_lpips_lambda > 0 and False:
            loss_lpips = self.lpips_loss(generated_images*(1-light_mask), real_images*(1-light_mask))
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += 0.01*loss_lpips * hyperparameters.pt_lpips_lambda
        
        if use_ball_holder and hyperparameters.use_locality_regularization and False:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss

    def calc_light_loss_remove_one_light(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch, bbox=[150,155,150,155]):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            mask = torch.zeros_like(generated_images)
            mask[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]] = 1
            mask_pixels = (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
            full_pixels = mask.shape[2]*mask.shape[3]#-mask_pixels

            l2_loss_val = l2_loss.l2_loss(generated_images*(1-mask), real_images*(1-mask))*full_pixels/(full_pixels-mask_pixels) #+torch.mean(torch.abs(generated_images*light_mask))#10*l2_loss.l2_loss(real_images*light_mask, generated_images*light_mask)
            loss += 5.0*l2_loss_val * hyperparameters.pt_l2_lambda    #########

            if bbox is not None:
                l2_loss_val2 = torch.sum(generated_images*mask)/mask_pixels

            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            
            if bbox is not None:
                loss += l2_loss_val2 * hyperparameters.pt_l2_lambda
        
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images*(1-mask), real_images*(1-mask))
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += 5.0*loss_lpips * hyperparameters.pt_lpips_lambda     ############
        
        if use_ball_holder and hyperparameters.use_locality_regularization and False:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss


    def calc_light_loss_add_one_light(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch, bbox=[150,155,150,155]):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            mask = torch.zeros_like(generated_images)
            mask[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]] = 1
            mask_pixels = (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
            full_pixels = mask.shape[2]*mask.shape[3]#-mask_pixels

            l2_loss_val = l2_loss.l2_loss(generated_images*(1-mask), real_images*(1-mask))*full_pixels/(full_pixels-mask_pixels) #+torch.mean(torch.abs(generated_images*light_mask))#10*l2_loss.l2_loss(real_images*light_mask, generated_images*light_mask)
            loss += 5.0*l2_loss_val * hyperparameters.pt_l2_lambda    #######3

            if bbox is not None:
                l2_loss_val2 = -torch.sum(generated_images*mask)/mask_pixels

            
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            
            if bbox is not None:
                loss += 1.0*l2_loss_val2 * hyperparameters.pt_l2_lambda
        
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images*(1-mask), real_images*(1-mask))
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += 5.0*loss_lpips * hyperparameters.pt_lpips_lambda ###########
        
        if use_ball_holder and hyperparameters.use_locality_regularization and False:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss




    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)

        return generated_images



    # do not use
    def initilize_e4e(self):
        ckpt = torch.load(paths_config.e4e, map_location='cpu')
        opts = ckpt['opts']
        opts['batch_size'] = hyperparameters.train_batch_size
        opts['checkpoint_path'] = paths_config.e4e
        opts = Namespace(**opts)
        self.e4e_inversion_net = pSp(opts)
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net = self.e4e_inversion_net.to(global_config.device)
        toogle_grad(self.e4e_inversion_net, False)
    
    # do not use
    def get_e4e_inversion(self, image):
        image = (image + 1) / 2
        new_image = self.e4e_image_transform(image[0]).to(global_config.device)
        _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        if self.use_wandb:
            log_image_from_w(w, self.G, 'First e4e inversion')
        return w



