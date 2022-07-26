import os
import torch
from tqdm import tqdm
from PTI_utils import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from training.coaches.base_editor import BaseEditor
from PTI_utils.log_utils import log_images_from_w

from skylibs.envmap import EnvironmentMap
import numpy as np
import PIL.Image
from PIL import Image, ImageDraw

# from skylibs.envmap import EnvironmentMap
from skylibs.demo_crop import crop2pano
from skylibs.hdrio import imread, imsave
import xml.etree.ElementTree as ET

import imageio
import glob

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

def xml2csv(file_whole_name):
    # To parse the xml files
    assert(file_whole_name.endswith(".xml"))

    # Open the xml name
    tree = ET.parse(file_whole_name)
    root = tree.getroot()

    # Get the width, height of images
    #  to normalize the bounding boxes
    size = root.find("size")
    width, height = float(size.find("width").text), float(size.find("height").text)
    # print('width, height:',width, height) #width, height: 512.0 256.0

    # ['light', 0.08984375, 0.234375, '', '', 0.2421875, 0.44140625, '', '', 'light', 0.576171875, 0.44140625, '', '', 0.623046875, 0.50390625, '', '', 'no_ligh
    # t', 0.337890625, 0.0703125, '', '', 0.419921875, 0.13671875, '', '']

    # Find all the bounding objects
    temp_csv=[]
    for label_object in root.findall("object"):
        # Class label
        # temp_csv.append(label_object.find("name").text)
        object_type = label_object.find("name").text

        # Bounding box coordinate
        bounding_box = label_object.find("bndbox")

        # Add the upper left coordinate
        x_min = float(bounding_box.find("xmin").text) #/ width
        y_min = float(bounding_box.find("ymin").text) #/ height
        # temp_csv.extend([x_min, y_min])
        # Add the lower right coordinate
        x_max = float(bounding_box.find("xmax").text) #/ width
        y_max = float(bounding_box.find("ymax").text) #/ height

        bbox = [int(y_min), int(y_max), int(x_min), int(x_max)]
        temp_csv.append(tuple([object_type, bbox]))

    return temp_csv


class MyEditor(BaseEditor):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):

        # w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        # os.makedirs(w_path_dir, exist_ok=True)
        # os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True
        is_128x256 = False

        # for fname, image in tqdm(self.data_loader):
        for image_name in tqdm(self.data_loader):
            # image_name = fname[0]
            print('image_name:',image_name)
            if False:
                # image = xxx
                tone = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
                e = EnvironmentMap(image_name, 'latlong')
                e.resize((256,512))
                if True:
                    e.data,_,_ = tone(e.data) 
                # image = (e.data*255.0).astype(np.uint8) 
                image = 2*e.data-1
            
            elif True:
                image = PIL.Image.open(image_name).convert('RGB')
                image = np.array(image)/127.5-1

            else:
                # env = EnvironmentMap(128, 'latlong')
                env = EnvironmentMap(256, 'latlong')
                image_crop2pano = crop2pano(env, image_name)
                print('image max:', image_crop2pano.max())
                image = 2*(image_crop2pano/255.0)-1
            
            # import pdb
            # pdb.set_trace()
            xml_path = image_name.split('.')[0]+'.xml'

            # example: [('light', [46.0, 60.0, 124.0, 113.0]), ('non_light', [173.0, 18.0, 215.0, 35.0])]
            # light, non_light, strong_light
            temp_csv = xml2csv(xml_path)

            image = torch.tensor(image.transpose([2, 0, 1]), device=global_config.device)#/255.0     ################################### 0-1
            image = image.unsqueeze(0).to(torch.float32)


            self.restart_training()

            # if self.image_counter >= hyperparameters.max_images_to_invert:   # max_images_to_invert=30, self.image_counter = 0
            #     break
            name = image_name.split('/')[-1].split('.')[0]
            # embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{name}'     ###############
            # os.makedirs(embedding_dir, exist_ok=True)


            w_pivot = None

            # if hyperparameters.use_last_w_pivots: # use_last_w_pivots = False, false for default
            #     w_pivot = self.load_inversions(w_path_dir, name)

            # elif not hyperparameters.use_last_w_pivots or w_pivot is None:
            #     # w_pivot = self.calc_inversions(image, name, bbox) ## here
            #     w_pivot = self.calc_inversions(image, name) ## here
            
            w_pivot = self.calc_inversions(image, name) ## here
            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)

            # torch.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            # real_images_batch = real_images_batch[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]]
            
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
                    # ldr = torch.clip(synth_image, -1, 1)
                    # hdr = torch.clip(synth_image-1, 0, 1e8)+1
                    # hdr = torch.clip(synth_image, -1, 1e8)
                    
                    # synth_image = synth_image*(1-mask)+(target_images/127.5-1)*mask
                    hdr = torch.clip(generated_images, -1, 1)
                    full = (hdr+1)/2
                    tone = True
                    if tone:
                        full_inv_tonemap = torch.pow(full/5, gamma)
                        img_hdr_np = full_inv_tonemap.permute(0, 2, 3, 1)[0].detach().cpu().numpy() 
                    else:
                        img_hdr_np = full.permute(0, 2, 3, 1)[0].cpu().numpy()


                    imsave(f'{paths_config.checkpoints_dir}/{name}_test.exr', img_hdr_np)

            
            
            for i in tqdm(range(hyperparameters.max_pti_steps)):   #max_pti_steps = 350
            # for i in tqdm(range(900)):   #max_pti_steps = 350
                generated_images = self.forward(w_pivot)
                generated_images = torch.clip(generated_images, -1, 1)                     ######

                # generated_images = generated_images[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]]
                combined_edit = False
                if combined_edit:
                    loss, l2_loss_val, loss_lpips, mask = self.calc_loss_new(generated_images, real_images_batch, name,
                                                                   self.G, use_ball_holder, w_pivot, temp_csv)
                else:
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
            

            ####################################################################################################
            # ops = ['light', 'non_light', 'strong_light', 'ignore'] # remove, add, add
            # select_ops = ['light']
            # select_ops = ['non_light', 'strong_light']
            # select_ops = ['light', 'non_light','strong_light']

            # 195: steps are 40, 400
            # 205, 10, 300
            # 279: 60, 400
            # 849?: 8, 400
            

            do_save_image = True

            if True:
                print('temp_csv:', temp_csv)
                for obj in temp_csv:
                    w_pivot = w_pivot.detach().requires_grad_()
                    noise_bufs = {name: buf for (name, buf) in self.G.synthesis.named_buffers() if 'noise_const' in name}
                    optimizer = torch.optim.Adam([w_pivot] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=hyperparameters.first_inv_lr)                
                    print('obj:', obj)
                    # assert(obj[0] in ['light', 'non_light', 'strong_light'])
                    if obj[0] not in ['light', 'strong_light']:
                        continue

                    edit_steps =8

                    for i in tqdm(range(edit_steps)):   #max_pti_steps = 350
                        generated_images_ = self.forward(w_pivot)
                        
                        if do_save_image:
                            generated_images_save = (generated_images_ + 1) * (255/2)
                            generated_images_save = generated_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                            PIL.Image.fromarray(generated_images_save, 'RGB').save(f'{paths_config.save_image_path}/{i:04}.png')



                        generated_images_ = torch.clip(generated_images_, -1, 1)                     ######
                        # print('bbox values:',generated_images_[:,:,obj[1][0]:obj[1][1],obj[1][2]:obj[1][3]])
                        loss = self.calc_light_loss_remove_one_light(generated_images_, generated_images.detach(), name, self.G, use_ball_holder, w_pivot, obj[1])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print('loss:', loss)

            if True:
                for obj in temp_csv:
                    w_pivot = w_pivot.detach().requires_grad_()
                    noise_bufs = {name: buf for (name, buf) in self.G.synthesis.named_buffers() if 'noise_const' in name}
                    optimizer = torch.optim.Adam([w_pivot] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=hyperparameters.first_inv_lr)                
                    print('obj:', obj)
                    # assert(obj[0] in ['light', 'non_light', 'strong_light'])
                    if obj[0] not in ['non_light']:
                        continue

                    edit_steps = 400

                    for i in tqdm(range(edit_steps)):   #max_pti_steps = 350
                        generated_images_2 = self.forward(w_pivot)

                        if do_save_image:
                            generated_images_save = (generated_images_2 + 1) * (255/2)
                            generated_images_save = generated_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                            PIL.Image.fromarray(generated_images_save, 'RGB').save(f'{paths_config.save_image_path}/{i:04}.png')



                        generated_images_2 = torch.clip(generated_images_2, -1, 1)                     ######
                        loss = self.calc_light_loss_add_one_light(generated_images_2, generated_images_.detach(), name, self.G, use_ball_holder, w_pivot, obj[1])
                        optimizer.zero_grad()
                        # print('loss:', loss)

                        loss.backward()
                        optimizer.step()




                
            # torch.save(self.G, f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
            ####################################################################################################
            
            generated_images = self.forward(w_pivot)
            is_png = True
            if is_png:
                generated_images = (generated_images + 1) * (255/2)
                generated_images = generated_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                PIL.Image.fromarray(generated_images, 'RGB').save(f'{paths_config.checkpoints_dir}/{name}_test.png')
            else:
                gamma = 2.4
                # ldr = torch.clip(synth_image, -1, 1)
                # hdr = torch.clip(synth_image-1, 0, 1e8)+1
                # hdr = torch.clip(synth_image, -1, 1e8)
                
                # synth_image = synth_image*(1-mask)+(target_images/127.5-1)*mask
                hdr = torch.clip(generated_images, -1, 1)
                full = (hdr+1)/2
                tone = False
                if tone:
                    full_inv_tonemap = torch.pow(full/5, gamma)
                    img_hdr_np = full_inv_tonemap.permute(0, 2, 3, 1)[0].detach().cpu().numpy() 
                else:
                    img_hdr_np = full.permute(0, 2, 3, 1)[0].detach().cpu().numpy()


                imsave(f'{paths_config.checkpoints_dir}/{name}_test.exr', img_hdr_np)


            # torch.save(self.G, f'{paths_config.checkpoints_dir}/{name}.pt')



            # save video
            if do_save_image:
                sequence_path=f'{paths_config.save_image_path}/*.png' 
                # sequences = sorted(glob.glob(f'{sequence_path}'))[::3]
                sequences = sorted(glob.glob(f'{sequence_path}'))[:]
                video_name=paths_config.save_video_path
                video = imageio.get_writer(f'{video_name}', mode='I', fps=25, codec='libx264', bitrate='16M')
                # img_fov = imageio.imread(filename) 
                for filename in sequences: 
                    img = imageio.imread(filename) 
                    # height, width, layers = img.shape
                    # size = (width,height)
                    # img_cat = np.concatenate([image_crop2pano, img], axis=1)      #img size (256, 512, 3)
                    # img_cat = np.concatenate([image_crop2pano, img], axis=1)      #img size (256, 512, 3)
                    video.append_data(img) 

                video.close()


            
