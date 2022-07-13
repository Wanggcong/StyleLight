# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from einops import rearrange, reduce, repeat
import math
#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x



#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        
        self.z_dim = z_dim     ##
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                # misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            # print('x shape and idx:', x.shape, idx)
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x



#LDR2HDR(img_resolution=img_resolution, img_channels=img_channels)
@persistence.persistent_class
class LDR2HDR(torch.nn.Module):
    def __init__(self,
        in_channels,                      # Input latent (Z) dimensionality, 0 = no latent.
        out_channels,                      # Conditioning label (C) dimensionality, 0 = no label.
    ):
        super().__init__()
        # self.in_channels = in_channels     ##
        # self.out_channels = out_channels
        # networks
        self.conv1 = Conv2dLayer(in_channels, out_channels, kernel_size=1,  bias = True, activation = 'lrelu')
        # self.conv2 = Conv2dLayer(out_channels, in_channels, kernel_size=1,  bias = True, activation = 'lrelu')
    def forward(self, img):

        img = self.conv1(img)
        # img = self.conv2(img)

        return img





#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, 2*resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, 2*in_resolution])

        # ### 
        # batch_size, num_channels, height, width = x.shape
        # random_index = np.random.randint(width)
        # x = torch.cat((x[:,:,:,random_index:],x[:,:,:,:random_index]), dim=3)

        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, 2*self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x







#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x






#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
 
        if resolution>=32:
            #out_channels_=out_channels//2
            out_channels_=out_channels
        else:
            out_channels_=out_channels

     
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, 2*resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels_, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1
        #self.conv1 = SynthesisLayerMobile(out_channels_, out_channels, w_dim=w_dim, resolution=resolution,
        self.conv1 = SynthesisLayer(out_channels_, out_channels, w_dim=w_dim, resolution=resolution,
        #self.conv1 = SynthesisLayerEfficient(out_channels_, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)
        #if self.is_last or self.architecture == 'skip':
        #if resolution >4:
        #    self.weight = torch.nn.parameter.Parameter(torch.randn(2))


    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        # misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            # misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        #import pdb
        #pdb.set_trace()
        #weight = self.weight.to(x.dtype)
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y
            #if img is not None:
            #    weight = self.weight.to(x.dtype)
            #    weight = torch.nn.functional.softmax(weight,dim=0)
            #    img = weight[0]*img+weight[1]*y
            #else:
            #    img = y
               
            #weight = self.weight.to(x.dtype)
            #weight = torch.nn.functional.softmax(weight,dim=0)
            #img = weight[0]*img+weight[1]*y if img is not None else y
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channels_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64},  # original 
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        #channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions} ### removed
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        #import pdb
        #pdb.set_trace()
        # orig: {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64}
        # channels_dict={4: 512, 8: 512, 16: 256, 32: 256, 64: 128, 128: 64, 256: 64}
        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            # misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


#----------------------------------------------------------------------------

# @persistence.persistent_class
# class SynthesisLayer(torch.nn.Module):
#     def __init__(self,
#         in_channels,                    # Number of input channels.
#         out_channels,                   # Number of output channels.
#         w_dim,                          # Intermediate latent (W) dimensionality.
#         resolution,                     # Resolution of this layer.
#         kernel_size     = 3,            # Convolution kernel size.
#         up              = 1,            # Integer upsampling factor.
#         use_noise       = True,         # Enable noise input?
#         activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
#         resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
#         conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
#         channels_last   = False,        # Use channels_last format for the weights?
#     ):
#         super().__init__()
#         self.resolution = resolution
#         self.up = up
#         self.use_noise = use_noise
#         self.activation = activation
#         self.conv_clamp = conv_clamp
#         self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
#         self.padding = kernel_size // 2
#         self.act_gain = bias_act.activation_funcs[activation].def_gain

#         self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
#         memory_format = torch.channels_last if channels_last else torch.contiguous_format
#         self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
#         if use_noise:
#             self.register_buffer('noise_const', torch.randn([resolution, 2*resolution]))
#             self.noise_strength = torch.nn.Parameter(torch.zeros([]))
#         self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

#     def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
#         assert noise_mode in ['random', 'const', 'none']
#         in_resolution = self.resolution // self.up
#         misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, 2*in_resolution])


#         styles = self.affine(w)

#         noise = None


#         flip_weight = (self.up == 1) # slightly faster
#         x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
#             padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

#         act_gain = self.act_gain * gain
#         act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
#         x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
#         return x

# @persistence.persistent_class
# class Synthesis_LDR2HDR(torch.nn.Module):
#     def __init__(self,
#         w_dim,                      # Intermediate latent (W) dimensionality.
#         img_resolution,             # Output image resolution.
#         img_channels,               # Number of color channels.
#         channels_dict = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64},  # original 
#         channel_base    = 32768,    # Overall multiplier for the number of channels.
#         channel_max     = 512,      # Maximum number of channels in any layer.
#         num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
#         **block_kwargs,             # Arguments for SynthesisBlock.
#     ):


#     def forward(self, ws, **block_kwargs):
#         pass

#         return img



@persistence.persistent_class
# class Generator_org(torch.nn.Module):
class Generator_(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        rank = 'cuda:0'
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        #############################
        # self.step_h = 4
        # self.step_w = 4
        # theta = torch.linspace(0, math.pi, steps=self.step_h) # 0-pi
        # phi = torch.linspace(0, 2*math.pi, steps=self.step_w) # 0-2pi
        # # grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing='ij')
        # grid_theta, grid_phi = torch.meshgrid(theta, phi)

        # theta_sin = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)
        # phi_cos = torch.cos(grid_phi).view(1, self.step_h, self.step_w,1)   # 0-2pi
        # phi_sin = torch.sin(grid_phi).view(1, self.step_h, self.step_w,1)

        # self.spherical_positions = torch.cat((theta_sin, phi_cos, phi_sin), dim=3).to(rank)  # 4, 4, 3


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        # b,dim = z.size()
        # # print('----z size0:', z.shape)

        # z = z.view(b,1, dim).expand(-1,self.step_h*self.step_w,-1)   # (N, Patches, dims)
        # position_codes = self.spherical_positions.expand(b, -1,-1,-1).view(b,self.step_h*self.step_w,-1) #(N, Patches, 3)
        # # position_codes = position_codes.view(-1, )
        # # print('----z size1:', z.shape)

        # z = torch.cat((z, position_codes), dim=2)
        # z = z.view(-1, z.size(2))
        # # print('----z size2:', z.shape)


        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)

        # img_ = rearrange(img, '(b h2 w2) c h w -> b c (h2 h) (w2 w)', h2=self.step_h, w2=self.step_w)   ####
        return img




@persistence.persistent_class
# class Generator_org(torch.nn.Module):
class Generator_org(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        rank = 'cuda:0'
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        #############################
        # self.step_h = 4
        # self.step_w = 4
        # theta = torch.linspace(0, math.pi, steps=self.step_h) # 0-pi
        # phi = torch.linspace(0, 2*math.pi, steps=self.step_w) # 0-2pi
        # # grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing='ij')
        # grid_theta, grid_phi = torch.meshgrid(theta, phi)

        # theta_sin = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)
        # phi_cos = torch.cos(grid_phi).view(1, self.step_h, self.step_w,1)   # 0-2pi
        # phi_sin = torch.sin(grid_phi).view(1, self.step_h, self.step_w,1)

        # self.spherical_positions = torch.cat((theta_sin, phi_cos, phi_sin), dim=3).to(rank)  # 4, 4, 3


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        # b,dim = z.size()
        # # print('----z size0:', z.shape)

        # z = z.view(b,1, dim).expand(-1,self.step_h*self.step_w,-1)   # (N, Patches, dims)
        # position_codes = self.spherical_positions.expand(b, -1,-1,-1).view(b,self.step_h*self.step_w,-1) #(N, Patches, 3)
        # # position_codes = position_codes.view(-1, )
        # # print('----z size1:', z.shape)

        # z = torch.cat((z, position_codes), dim=2)
        # z = z.view(-1, z.size(2))
        # # print('----z size2:', z.shape)


        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)

        # img_ = rearrange(img, '(b h2 w2) c h w -> b c (h2 h) (w2 w)', h2=self.step_h, w2=self.step_w)   ####
        return img



#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator_position(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        rank = 'cuda:0'
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        #############################
        self.step_h = 4
        self.step_w = 4
        # theta = torch.linspace(0, math.pi, steps=self.step_h) # 0-pi
        # phi = torch.linspace(0, 2*math.pi, steps=self.step_w) # 0-2pi

        theta = torch.linspace(0, math.pi, steps=self.step_h+1)[:self.step_h] # 0-pi
        phi = torch.linspace(0, 2*math.pi, steps=self.step_w+1)[:self.step_w] # 0-2pi

        # grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing='ij')
        grid_theta, grid_phi = torch.meshgrid(theta, phi)

        # theta_sin = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)
        theta_sin = torch.cos(grid_theta).view(1, self.step_h, self.step_w,1)
        phi_cos = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)*torch.cos(grid_phi).view(1, self.step_h, self.step_w,1)   # 0-2pi
        phi_sin = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)*torch.sin(grid_phi).view(1, self.step_h, self.step_w,1)

        self.spherical_positions = torch.cat((theta_sin, phi_cos, phi_sin), dim=3).to(rank)  # 4, 4, 3

        # # model positions as bbox
        self.spherical_positions_ = torch.roll(self.spherical_positions, shifts=(1, 1), dims=(1, 2))  # 4, 4, 3  ->b, 4, 4, 3
        self.spherical_positions = torch.cat((self.spherical_positions,self.spherical_positions_), dim=3)


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        b, dim = z.size()
        # print('----z size0:', z.shape)

        z = z.view(b,1, dim).expand(-1,self.step_h*self.step_w,-1)   # (N, Patches, dims)
        # position_codes = self.spherical_positions.expand(b, -1,-1,-1).reshape(b*self.step_h*self.step_w,1,-1) #(N, Patches, 3)
        # c = self.spherical_positions.expand(b, -1,-1,-1).reshape(b*self.step_h*self.step_w,1,-1) #(N, Patches, 3)
        c = self.spherical_positions.expand(b, -1,-1,-1).reshape(b*self.step_h*self.step_w,-1) #(N, Patches, 3)
        # position_codes = position_codes.view(-1, )
        # print('----z size1:', z.shape)

        # z = torch.cat((z, position_codes), dim=2)
        z = z.reshape(-1, z.size(2))
        # print('----z size2:', z.shape)


        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        # b, dim = ws.size()
        # ws = ws.reshape(b,1, dim).expand(-1,self.step_h*self.step_w,-1).reshape(b*self.step_h*self.step_w,-1)

        # ws = torch.cat((ws, position_codes), dim=1)  ####

        # b, num_ws, dim = ws.size()
        # ws = ws.reshape(b,1, num_ws, dim).expand(-1,self.step_h*self.step_w,-1, -1).reshape(b*self.step_h*self.step_w, num_ws, -1)
        # position_codes = position_codes.expand(-1,num_ws,-1)
        # ws = torch.cat((ws, position_codes), dim=2)  ####


        img = self.synthesis(ws, **synthesis_kwargs)

        img_ = rearrange(img, '(b h2 w2) c h w -> b c (h2 h) (w2 w)', h2=self.step_h, w2=self.step_w)   ####
        return img_


#################################################

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        rank = 'cuda:0'
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.has_positional_coding = False
        # self.has_positional_coding = True
        if self.has_positional_coding:
             self.synthesis = SynthesisNetwork(w_dim=w_dim+6, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        else:
            self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        self.ldr2hdr = LDR2HDR(in_channels=3, out_channels=3)

        #############################
        self.step_h = 4
        self.step_w = 4
        # theta = torch.linspace(0, math.pi, steps=self.step_h) # 0-pi
        # phi = torch.linspace(0, 2*math.pi, steps=self.step_w) # 0-2pi

        theta = torch.linspace(0, math.pi, steps=self.step_h+1)[:self.step_h] # 0-pi
        phi = torch.linspace(0, 2*math.pi, steps=self.step_w+1)[:self.step_w] # 0-2pi

        # grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing='ij')
        grid_theta, grid_phi = torch.meshgrid(theta, phi)

        # theta_sin = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)
        theta_sin = torch.cos(grid_theta).view(1, self.step_h, self.step_w,1)
        phi_cos = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)*torch.cos(grid_phi).view(1, self.step_h, self.step_w,1)   # 0-2pi
        phi_sin = torch.sin(grid_theta).view(1, self.step_h, self.step_w,1)*torch.sin(grid_phi).view(1, self.step_h, self.step_w,1)

        self.spherical_positions = torch.cat((theta_sin, phi_cos, phi_sin), dim=3).to(rank)  # 4, 4, 3

        # # model positions as bbox
        self.spherical_positions_ = torch.roll(self.spherical_positions, shifts=(1, 1), dims=(1, 2))  # 4, 4, 3  ->b, 4, 4, 3
        self.spherical_positions = torch.cat((self.spherical_positions, self.spherical_positions_), dim=3)

        self.spherical_positions = self.spherical_positions.reshape(1, self.step_h*self.step_w, 1, -1)


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        b, num_ws, dim = ws.size()

        if self.has_positional_coding:
            ws = ws.view(b, 1, num_ws, dim).expand(-1,self.step_h*self.step_w, -1, -1).reshape(b*self.step_h*self.step_w, num_ws, dim)   # (N, Patches, dims)
            c = self.spherical_positions.expand(b, -1, num_ws,-1).reshape(b*self.step_h*self.step_w, num_ws,-1) #(N, Patches, 3)            
            ws = torch.cat((ws,c), dim=2)


        img_ = self.synthesis(ws, **synthesis_kwargs)
        if self.has_positional_coding:
            img_ = rearrange(img_, '(b h2 w2) c h w -> b c (h2 h) (w2 w)', h2=self.step_h, w2=self.step_w)   ####
        return img_


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x





#################################################
@persistence.persistent_class
class ViT(nn.Module):
    def __init__(self, img_resolution=512):
        super().__init__()
        # self.z_dim = z_dim
        # self.c_dim = c_dim
        # self.w_dim = w_dim
        # self.img_resolution = img_resolution
        # self.img_channels = img_channels
        # self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        # self.num_ws = self.synthesis.num_ws
        # self.num_ws = 6
        # self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    # def __init__(self, *, image_size, patch_size, out_dim, dim, depth, heads, mlp_dim, pool = 'all', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
    #     super().__init__()
        image_size = (img_resolution, img_resolution) #512
        patch_size = (16, 16)
        out_dim = 16*16*3
        dim = 1024
        depth = 6
        heads = 16
        # mlp_dim = 2048
        mlp_dim = 1024
        pool = 'all' 
        channels = 3 
        dim_head = 64 
        dropout = 0.
        emb_dropout = 0.1
        # emb_dropout = 0.1

        if len(image_size)==2:
            image_height, image_width = image_size  # eg., 100, 400
        else:
            image_height, image_width = pair(image_size)
        
        if len(patch_size) == 2:
            patch_height, patch_width = patch_size   # 100, 25
        else:
            patch_height, patch_width = pair(patch_size)


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        print('################ num_patches:',num_patches)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean', 'pred', 'all'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # self.downsampling = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
        # print('patch_height, patch_width, patch_dim, dim:',patch_height, patch_width, patch_dim, dim)
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     # Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        # )

        # noise_dim = 512
        # self.mapping = nn.Linear(noise_dim, dim)     # synthesis


        self.patch_height= patch_height
        self.patch_width= patch_width

        self.image_height= image_height
        self.image_width= image_width
 
        # self.synthesis = nn.Sequential(

        # )

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) 



        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.transformer = OutPaintingTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
    
    # def forward(self, noises):          # noise size: b (h w) (p1 p2 c)
    def forward(self, ws, **block_kwargs):
    # def forward(self, ws):
        # n,_ = noises.shape
        # noises_ = self.mapping(noises)     #  torch.Size([25, 24, 1024]) torch.Size([25, 3, 256, 384])
        # x += self.pos_embedding[:, :(n + 1)]
        n = self.num_patches

        b,_ = ws.shape         ## b is batchsize
        ws=ws.view(b,1,-1)
        
        # copied to n patches
        x = ws.expand(b, n, -1).clone()  #x.expand(-1, 4)

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :n]    # x: torch.Size([8, 1025, 1024])
        # print('-##################################-xx   x:',x.shape)

        # print('x14 size:', x.size())

        x = self.dropout(x)
        # print('x2 size:', x.size())
        # x = self.transformer(x, mask)  # add mask
        x = self.transformer(x)  # add mask
        # print('x3 size:', x.size())

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]   ####
        if self.pool == 'mean':
            x = x.mean(dim = 1)
        elif self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'all':
            # x = x[:,:n]
            x =  x.reshape(n*b,-1)
        else:
            x = x[:,-1]


        x = self.to_latent(x)

        x = self.mlp_head(x)
        

        if self.pool == 'predict':
            patch = x.view(b, 3, 128, 32)
            return patch
        elif self.pool == 'mean' or self.pool == 'cls':
            return x
        else:  # for all patches
            # patches = x.view(n, b, 3, 128, 32)
            # print('x.size:', x.size())
            # y = rearrange(x, 'b c h w -> b (c h w)')
            patches = rearrange(x, '(b h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_height//self.patch_height, w=self.image_width//self.patch_width, p1=self.patch_height, p2=self.patch_width)
            # Rearrange('(b h w) (p1 p2 c) -> b c (h p1) (w p2)', h =image_height // patch_height, w=image_width // patch_width, p1 = patch_height, p2 = patch_width),
            # patches = self.to_patches(x)
            return patches


@persistence.persistent_class
class MLP_Mapping(nn.Module):
    def __init__(self,noise_dim=512, dim=1024):
        super().__init__()
        self.mapping = nn.Linear(noise_dim, dim) 

    # def forward(self, x, c):sss
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        x = self.mapping(z)

        return  x



@persistence.persistent_class
class Generator_vit(nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        # self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        # self.num_ws = self.synthesis.num_ws
        self.num_ws = 6

        self.mapping = MLP_Mapping(512,1024)
        self.synthesis = ViT()

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        x =  self.synthesis(ws, **synthesis_kwargs) 

        return x



        # ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        # img = self.synthesis(ws, **synthesis_kwargs)


#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            # print('x.shape:', x.shape)
            # print('self.resolution:', self.resolution)
            misc.assert_shape(x, [None, self.in_channels, self.resolution, 2*self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            # misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F
        n = N//G
        # print: 8 1 tensor(8) 1 512 32
        # print('self.group_size,self.num_channels,G,F,c,N:',self.group_size, self.num_channels, G,F,c,N)
        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.

        #y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        #y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.

        y = y.reshape(n,1, F, 1, 1)
        y = y.expand(n,G,F, 1, 1).reshape(-1,F,1,1)
        y = y.repeat(1, 1, H, W)

        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (2*resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, 2*self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, 2*self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        channels_dict= {256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512},
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        # channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        #######################################################
        batch_size, num_channels, height, width = img.shape
        # print('batch_size, num_channels, height, width:',batch_size, num_channels, height, width)
        random_index = np.random.randint(width)
        img = torch.cat((img[:,:,:,random_index:],img[:,:,:,:random_index]), dim=3)
        #######################################################

        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------
