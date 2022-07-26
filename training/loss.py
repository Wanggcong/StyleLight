# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from PIL import Image
from einops import rearrange, reduce, repeat
import math
from . import tonemapping #import TonemapHDR
#----------------------------------------------------------------------------

def tonemapping(x, gamma=2.4, percentile=0.5, max_mapping=0.5, clip=True):
    x = torch.pow(x, 1/gamma)
    non_zero = x > 0
    if non_zero.any():
        r_percentile = torch.quantile(x[non_zero], percentile)
    else:
        r_percentile = torch.quantile(x, percentile)

    alpha = max_mapping / (r_percentile + 1e-10)
    x = x*alpha
    if clip:
        x =torch.clip(x, 0, 1)
    return x


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_ldr2hdr, G_mapping, G_synthesis, D, D_, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.G_ldr2hdr = G_ldr2hdr
        self.D = D
        self.D_ = D_
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)


        self.step_h = 4
        self.step_w = 4
        theta = torch.linspace(0, math.pi, steps=self.step_h+1)[:self.step_h] # 0-pi
        phi = torch.linspace(0, 2*math.pi, steps=self.step_w+1)[:self.step_w] # 0-2pi
        # grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing='ij')
        grid_theta, grid_phi = torch.meshgrid(theta, phi)

        # theta_sin = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)
        theta_sin = torch.cos(grid_theta).view(1, self.step_h, self.step_w,1)
        phi_cos = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)*torch.cos(grid_phi).view(1, self.step_h, self.step_w,1)   # 0-2pi
        phi_sin = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)*torch.sin(grid_phi).view(1, self.step_h, self.step_w,1)

        self.spherical_positions = torch.cat((theta_sin, phi_cos, phi_sin), dim=3).to(device)  # 4, 4, 3  ->b, 4, 4, 3
        
        # model positions as bbox
        self.spherical_positions_ = torch.roll(self.spherical_positions, shifts=(1, 1), dims=(1, 2))  # 4, 4, 3  ->b, 4, 4, 3
        self.spherical_positions = torch.cat((self.spherical_positions,self.spherical_positions_), dim=3)
        self.spherical_positions_new = self.spherical_positions.reshape(1, self.step_h*self.step_w, 1, -1)

    # original
    def run_G_org(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        # return img, ws
        return img, img, ws

    # original
    def run_G_(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img_ldr_ = self.G_synthesis(ws)  # predict ldr
            img_ldr = torch.clip(img_ldr_, -1, 1)
            img_ldr_ = (img_ldr_+1)/2
            gamma = 2.4
            img_hdr = torch.clip(img_ldr_, 0.0, 1e8)
            img_hdr = torch.pow(img_hdr/5, gamma)

        return img_ldr, img_hdr, ws


    # modified
    def run_G_z_position(self, z, c, sync):
        b,dim = z.size()
        z = z.view(b,1, dim).expand(-1,self.step_h*self.step_w,-1)   # (N, Patches, dims)
        position_codes = self.spherical_positions.expand(b, -1,-1,-1).reshape(b*self.step_h*self.step_w,1,-1) #(N, Patches, 3)
        c = self.spherical_positions.expand(b, -1,-1,-1).reshape(b*self.step_h*self.step_w,-1) #(N, Patches, 3)
        z = z.reshape(-1, z.size(2))
        
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        
        # reshape patches into an image
        # patches = rearrange(x, '(b h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_height//self.patch_height, w=self.image_width//self.patch_width, p1=self.patch_height, p2=self.patch_width)
        img_ldr_ = rearrange(img, '(b h2 w2) c h w -> b c (h2 h) (w2 w)', h2=self.step_h, w2=self.step_w)   ####
        
        img_ldr = torch.clip(img_ldr_, -1, 1)
        
        img_ldr_ = (img_ldr_+1)/2

        gamma = 2.4
        img_hdr = torch.clip(img_ldr_, 0.0, 1e8)
        img_hdr = torch.pow(img_hdr/5, gamma)
        return img_ldr, img_hdr, ws, img

    # modified, ws+ position
    def run_G(self, z, c, sync):
     
        with misc.ddp_sync(self.G_mapping, sync):
            # ws = self.G_mapping(z, c)
            ws = self.G_mapping(z, None)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        has_positional_coding = False
        # has_positional_coding = True
        if has_positional_coding:
            b, num_ws, dim = ws.size()
            ws = ws.view(b, 1, num_ws, dim).expand(-1,self.step_h*self.step_w, -1, -1).reshape(b*self.step_h*self.step_w, num_ws, dim)   # (N, Patches, dims)
            c = self.spherical_positions_new.expand(b, -1, num_ws,-1).reshape(b*self.step_h*self.step_w, num_ws, -1) #(N, Patches, 3)
            ws = torch.cat((ws,c), dim=2)


        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        
        # reshape patches into an image
        # patches = rearrange(x, '(b h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_height//self.patch_height, w=self.image_width//self.patch_width, p1=self.patch_height, p2=self.patch_width)
        if has_positional_coding:
            img_shared = rearrange(img, '(b h2 w2) c h w -> b c (h2 h) (w2 w)', h2=self.step_h, w2=self.step_w)   ####
            img_ldr = torch.clip(img_shared, -1, 1)
            # img_hdr = torch.clip(img_shared, -1, 100)
            img_hdr = torch.clip(img_shared, -1, 10)
        else:
            img_shared = img
            img_ldr = torch.clip(img_shared, -1, 1)
            # img_hdr = torch.clip(img_shared, -1, 100)
            img_hdr = torch.clip(img_shared, -1, 10)

        img_hdr = torch.clip(img_hdr, -1, 1e8)
        # return img, img_, ws
        return img_ldr, img_hdr, ws, img

    # modified, ws+ position
    def run_G_hdr2ldr(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            # ws = self.G_mapping(z, c)
            ws = self.G_mapping(z, None)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        
        img_shared = img
        img_hdr = torch.clip(img_shared, -1, 1e8)

        return img_ldr, img_hdr, ws, img

    # for ldr
    def run_D(self, img, c, sync, isRealImage=False):
        if isRealImage:
            img = img[:,:3,:,:]    ##(ldr,hdr)
            # img = img[:,3:,:,:]    ##(ldr,hdr)

        if isRealImage:
            image =(img[0,:,:,:]+1)/2
            images_np = image.cpu().detach().numpy().transpose(1,2,0)#*5
            # images_np = np.clip((images_np+1)*0.5*255, 0, 255)
            images_np = np.clip(images_np*255, 0, 255)
            im_ = Image.fromarray((images_np).astype(np.uint8))
            # im_.save('AugmentPipe_xxx_yyy_zzz_before.png')


        if isRealImage:
            # image =img[0,:,:,:]
            image =(img[0,:,:,:]+1)/2
            images_np = image.cpu().detach().numpy().transpose(1,2,0)#*5
            # images_np = np.clip((images_np+1)*0.5*255, 0, 255)
            images_np = np.clip(images_np*255, 0, 255)
            im_ = Image.fromarray((images_np).astype(np.uint8))
            # im_.save('AugmentPipe_xxx_yyy_zzz.png')

        
        if self.augment_pipe is not None:
            img = self.augment_pipe(img, isRealImage=isRealImage)
        
        if isRealImage:
            # image =img[0,:,:,:]
            image =(img[0,:,:,:]+1)/2
            images_np = image.cpu().detach().numpy().transpose(1,2,0)#*5
            # images_np = np.clip((images_np+1)*0.5*255, 0, 255)
            images_np = np.clip(images_np*255, 0, 255)
            im_ = Image.fromarray((images_np).astype(np.uint8))
            # im_.save('AugmentPipe_xxx_yyy_zzz_aug.png')

        with misc.ddp_sync(self.D, sync):
            # print('img shape:',img.shape)
            logits = self.D(img, c)
        return logits

    # for hdr
    def run_D_hdr(self, img, c, sync, isRealImage=False):

        if isRealImage:
            # img = img[:,:3,:,:]  #(ldr,hdr), for debugging
            img = img[:,3:,:,:]  #(ldr,hdr)                      ######################## diff from run_D

        if self.augment_pipe is not None:
            img = self.augment_pipe(img, isRealImage=isRealImage)
    
        with misc.ddp_sync(self.D_, sync):
            logits = self.D_(img, c)
        return logits


    def run_D_old(self, img, c, sync, isRealImage=False):
        gamma = 2.4
        # percentile=50
        percentile=0.99
        max_mapping=0.99
        # alpha = None

        self.tonemap = tonemapping.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)


        if self.augment_pipe is not None:
            img = self.augment_pipe(img, isRealImage=isRealImage)
        with misc.ddp_sync(self.D_, sync):
            # img: from hdr->ldr
            img = torch.clip(img, 1e-10,1e8)
            img_power = torch.pow(img, 1 / gamma)

            alpha = 5.0

            # tonemapped_img = np.multiply(alpha, img_power)
            tonemapped_img = alpha*img_power
            tonemapped_img = torch.clip(tonemapped_img, 0, 1)

            if isRealImage:
                image =tonemapped_img[0,:,:,:]
                images_np = image.cpu().detach().numpy().transpose(1,2,0)#*5
                # images_np = np.clip((images_np+1)*0.5*255, 0, 255)
                images_np = np.clip(images_np*255, 0, 255)
                im_ = Image.fromarray((images_np).astype(np.uint8))
                # im_.save('AugmentPipe_xxx_yyy.png')
                
            # logits = self.D_(tonemapped_img, c)
            logits = self.D(tonemapped_img, c)
        return logits

    def accumulate_gradients_hdr_only(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        # assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D_main', 'D_reg', 'D_both']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dmain_ = (phase in ['D_main', 'D_both'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        do_Dr1_   = (phase in ['D_reg', 'D_both']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws, _ = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                # gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False,isRealImage=False)    ########  add isRealImage=False
                gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False,isRealImage=False)    ########  add isRealImage=False
                training_stats.report('Loss/scores/fake', gen_logits_)
                training_stats.report('Loss/signs/fake', gen_logits_.sign())
                # loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gmain_ = torch.nn.functional.softplus(-gen_logits_) # -log(sigmoid(gen_logits))
                # training_stats.report('Loss/G/loss', loss_Gmain+loss_Gmain_)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain_.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img_ldr, gen_img_hdr, gen_ws, img = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(img) / np.sqrt(img.shape[2] * img.shape[3])

                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                if len(gen_ws.shape)==2:
                    pl_lengths = pl_grads.square().sum(1).sqrt()
                else:
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if False:
        # if False:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws,_ = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                # gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                # loss_Dgen_ = torch.nn.functional.softplus(gen_logits_) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()
                # loss_Dgen_.mean().mul(gain).backward()
                # (0*loss_Dgen+loss_Dgen_).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen_ = 0
        if do_Dmain_:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws,_ = self.run_G(gen_z, gen_c, sync=False)
                # gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                training_stats.report('Loss/scores/fake', gen_logits_)
                training_stats.report('Loss/signs/fake', gen_logits_.sign())
                # loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dgen_ = torch.nn.functional.softplus(gen_logits_) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward_'):
                # loss_Dgen.mean().mul(gain).backward()
                loss_Dgen_.mean().mul(gain).backward()
                # (0*loss_Dgen+loss_Dgen_).mean().mul(gain).backward()


        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        # if do_Dmain or do_Dr1:
        if False:
        # if False:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                # real_logits_ = self.run_D_hdr(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    # loss_Dreal_ = torch.nn.functional.softplus(-real_logits_) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        # r1_grads_ = torch.autograd.grad(outputs=[real_logits_.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    # r1_penalty_ = r1_grads_.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    # loss_Dr1_ = r1_penalty_ * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain_ or do_Dr1_:
            name = 'Dreal_Dr1_' if do_Dmain_ and do_Dr1_ else 'Dreal_' if do_Dmain_ else 'Dr1_'
            with torch.autograd.profiler.record_function(name + '_forward_'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1_)
                # real_logits = self.run_D(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                real_logits_ = self.run_D_hdr(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                training_stats.report('Loss/scores/real_', real_logits_)
                training_stats.report('Loss/signs/real_', real_logits_.sign())

                # loss_Dreal = 0
                loss_Dreal_ = 0
                if do_Dmain_:
                    # loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dreal_ = torch.nn.functional.softplus(-real_logits_) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_', loss_Dgen_ + loss_Dreal_)

                # loss_Dr1 = 0
                loss_Dr1_ = 0
                if do_Dr1_:
                    with torch.autograd.profiler.record_function('r1_grads_'), conv2d_gradfix.no_weight_gradients():
                        # r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_grads_ = torch.autograd.grad(outputs=[real_logits_.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    # r1_penalty = r1_grads.square().sum([1,2,3])
                    r1_penalty_ = r1_grads_.square().sum([1,2,3])
                    # loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    loss_Dr1_ = r1_penalty_ * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_', r1_penalty_)
                    training_stats.report('Loss/D/reg_', loss_Dr1_)

            with torch.autograd.profiler.record_function(name + '_backward_'):
                (real_logits_ * 0 + loss_Dreal_ + loss_Dr1_).mean().mul(gain).backward()

    # could be wrong, self.run_G returns hdr, ldr---> modifed into ldr, hdr, not sure if the order is changed 
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        # assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D_main', 'D_reg', 'D_both']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dmain_ = (phase in ['D_main', 'D_both'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        do_Dr1_   = (phase in ['D_reg', 'D_both']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws, _ = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False,isRealImage=False)    ########  add isRealImage=False
                gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False,isRealImage=False)    ########  add isRealImage=False
                training_stats.report('Loss/scores/fake', gen_logits_)
                training_stats.report('Loss/signs/fake', gen_logits_.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gmain_ = torch.nn.functional.softplus(-gen_logits_) # -log(sigmoid(gen_logits))
                # training_stats.report('Loss/G/loss', loss_Gmain+loss_Gmain_)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain+loss_Gmain_).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img_ldr, gen_img_hdr, gen_ws, img = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(img) / np.sqrt(img.shape[2] * img.shape[3])

                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    # pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                    pl_grads = torch.autograd.grad(outputs=[(img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                # pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                # print('pl_grads shape:',pl_grads.shape)
                if len(gen_ws.shape)==2:
                    pl_lengths = pl_grads.square().sum(1).sqrt()
                else:
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
        # if False:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws,_ = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                # gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                # loss_Dgen_ = torch.nn.functional.softplus(gen_logits_) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen_ = 0
        if do_Dmain_:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws,_ = self.run_G(gen_z, gen_c, sync=False)
                # gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                training_stats.report('Loss/scores/fake', gen_logits_)
                training_stats.report('Loss/signs/fake', gen_logits_.sign())
                # loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dgen_ = torch.nn.functional.softplus(gen_logits_) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward_'):
                # loss_Dgen.mean().mul(gain).backward()
                loss_Dgen_.mean().mul(gain).backward()
                # (0*loss_Dgen+loss_Dgen_).mean().mul(gain).backward()


        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
        # if False:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                # real_logits_ = self.run_D_hdr(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                # loss_Dreal_ = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    # loss_Dreal_ = torch.nn.functional.softplus(-real_logits_) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        # r1_grads_ = torch.autograd.grad(outputs=[real_logits_.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    # r1_penalty_ = r1_grads_.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    # loss_Dr1_ = r1_penalty_ * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain_ or do_Dr1_:
            name = 'Dreal_Dr1_' if do_Dmain_ and do_Dr1_ else 'Dreal_' if do_Dmain_ else 'Dr1_'
            with torch.autograd.profiler.record_function(name + '_forward_'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1_)
                # real_logits = self.run_D(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                real_logits_ = self.run_D_hdr(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                training_stats.report('Loss/scores/real_', real_logits_)
                training_stats.report('Loss/signs/real_', real_logits_.sign())

                # loss_Dreal = 0
                loss_Dreal_ = 0
                if do_Dmain_:
                    # loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dreal_ = torch.nn.functional.softplus(-real_logits_) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_', loss_Dgen_ + loss_Dreal_)

                # loss_Dr1 = 0
                loss_Dr1_ = 0
                if do_Dr1_:
                    with torch.autograd.profiler.record_function('r1_grads_'), conv2d_gradfix.no_weight_gradients():
                        # r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_grads_ = torch.autograd.grad(outputs=[real_logits_.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    # r1_penalty = r1_grads.square().sum([1,2,3])
                    r1_penalty_ = r1_grads_.square().sum([1,2,3])
                    # loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    loss_Dr1_ = r1_penalty_ * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_', r1_penalty_)
                    training_stats.report('Loss/D/reg_', loss_Dr1_)

            with torch.autograd.profiler.record_function(name + '_backward_'):
                (real_logits_ * 0 + loss_Dreal_ + loss_Dr1_).mean().mul(gain).backward()

    # could be wrong, self.run_G returns hdr, ldr---> modifed into ldr, hdr, not sure if the order is changed 
    def accumulate_gradients_ldr_only(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        # assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D_main', 'D_reg', 'D_both']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dmain_ = (phase in ['D_main', 'D_both'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        do_Dr1_   = (phase in ['D_reg', 'D_both']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws, _ = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False,isRealImage=False)    ########  add isRealImage=False
                # gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False,isRealImage=False)    ########  add isRealImage=False
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()


        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img_ldr, gen_img_hdr, gen_ws, img = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(img) / np.sqrt(img.shape[2] * img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    # pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                    pl_grads = torch.autograd.grad(outputs=[(img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                # pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                # print('pl_grads shape:',pl_grads.shape)
                if len(gen_ws.shape)==2:
                    pl_lengths = pl_grads.square().sum(1).sqrt()
                else:
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
        # if False:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws,_ = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                # gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                # loss_Dgen_ = torch.nn.functional.softplus(gen_logits_) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()


        # Dmain: Minimize logits for generated images.
        loss_Dgen_ = 0
        # if do_Dmain_:
        if False:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img_ldr, gen_img_hdr, _gen_ws,_ = self.run_G(gen_z, gen_c, sync=False)
                # gen_logits = self.run_D(gen_img_ldr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                gen_logits_ = self.run_D_hdr(gen_img_hdr, gen_c, sync=False, isRealImage=False) # Gets synced by loss_Dreal.   ########
                training_stats.report('Loss/scores/fake', gen_logits_)
                training_stats.report('Loss/signs/fake', gen_logits_.sign())
                # loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dgen_ = torch.nn.functional.softplus(gen_logits_) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward_'):
                # loss_Dgen.mean().mul(gain).backward()
                loss_Dgen_.mean().mul(gain).backward()
                # (0*loss_Dgen+loss_Dgen_).mean().mul(gain).backward()


        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
        # if False:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                # real_logits_ = self.run_D_hdr(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                # loss_Dreal_ = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    # loss_Dreal_ = torch.nn.functional.softplus(-real_logits_) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                # loss_Dr1_ = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        # r1_grads_ = torch.autograd.grad(outputs=[real_logits_.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    # r1_penalty_ = r1_grads_.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    # loss_Dr1_ = r1_penalty_ * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
                # (real_logits * 0 +real_logits_ * 0 + loss_Dreal + loss_Dreal_+loss_Dr1+loss_Dr1_).mean().mul(gain).backward()
                # (real_logits * 0 +real_logits_ * 0 + 0*loss_Dreal + loss_Dreal_+0*loss_Dr1+loss_Dr1_).mean().mul(gain).backward()


        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        # if do_Dmain_ or do_Dr1_:
        if False:
            name = 'Dreal_Dr1_' if do_Dmain_ and do_Dr1_ else 'Dreal_' if do_Dmain_ else 'Dr1_'
            with torch.autograd.profiler.record_function(name + '_forward_'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1_)
                # real_logits = self.run_D(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                real_logits_ = self.run_D_hdr(real_img_tmp, real_c, sync=sync, isRealImage=True)    ########
                training_stats.report('Loss/scores/real_', real_logits_)
                training_stats.report('Loss/signs/real_', real_logits_.sign())

                # loss_Dreal = 0
                loss_Dreal_ = 0
                if do_Dmain_:
                    # loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dreal_ = torch.nn.functional.softplus(-real_logits_) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_', loss_Dgen_ + loss_Dreal_)

                # loss_Dr1 = 0
                loss_Dr1_ = 0
                if do_Dr1_:
                    with torch.autograd.profiler.record_function('r1_grads_'), conv2d_gradfix.no_weight_gradients():
                        # r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_grads_ = torch.autograd.grad(outputs=[real_logits_.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    # r1_penalty = r1_grads.square().sum([1,2,3])
                    r1_penalty_ = r1_grads_.square().sum([1,2,3])
                    # loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    loss_Dr1_ = r1_penalty_ * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_', r1_penalty_)
                    training_stats.report('Loss/D/reg_', loss_Dr1_)

            with torch.autograd.profiler.record_function(name + '_backward_'):
                (real_logits_ * 0 + loss_Dreal_ + loss_Dr1_).mean().mul(gain).backward()

