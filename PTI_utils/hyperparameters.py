## Architechture
lpips_type = 'alex'
first_inv_type = 'w'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1     #####
use_locality_regularization = False

regulizer_alpha = 30

## Loss
# pt_l2_lambda = 1 # org
# pt_l2_lambda = 10
# pt_lpips_lambda = 1

## Steps
LPIPS_value_threshold = 0.06    #####
max_pti_steps = 350           ######
first_inv_steps = 450                    ######
max_images_to_invert = 30       ######

## Optimization
pti_learning_rate = 3e-4     ######
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = False    ######

###################for fov ldr-> panorama hdr###########################
# # for fov ldr-> panorama hdr

edit = False
pt_l2_lambda = 10
pt_lpips_lambda = 1
regulizer_l2_lambda = 0.01 #org 0.1
regulizer_lpips_lambda = 0.01 # org 0.1

################### lighting editing ###########################
# # for lighting editing

# edit = True
# pt_l2_lambda = 1
# pt_lpips_lambda = 1
# regulizer_l2_lambda = 0.01 #org 0.1
# regulizer_lpips_lambda = 0.01 # org 0.1
# edit = True
# edit_steps = 20
# percentile = 0.9