## Pretrained models paths
e4e = './pretrained_models/e4e_ffhq_encode.pt'             ##########
# stylegan2_ada_ffhq = '../pretrained_models/ffhq.pkl'             #######
# stylegan2_ada_ffhq = '/home/deep/projects/mini-stylegan2/checkpoints/coor2-00061-large-model/network-snapshot-002016.pkl'             #######
# stylegan2_ada_ffhq = '/home/deep/projects/mini-stylegan2/checkpoints/256x512-coor2-00001/network-snapshot-002000.pkl'             #######
stylegan2_ada_ffhq = 'training-runs-paper512-cyclic-new-training-128x256-ws_plus_coor2-accepted/00003--mirror-paper256-noaug/network-snapshot-002000.pkl'             #######
 

style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'

## Dirs for output files
# checkpoints_dir = './checkpoints_edit'           ######
#checkpoints_dir = './checkpoints_edit_new_wild_images'           ######
# checkpoints_dir = './checkpoints_without_fov_loss'           ###### before rebuttal
# checkpoints_dir = './checkpoints_without_light_mask'           ###### rebuttal
checkpoints_dir = './assets/checkpoints_without_light_mask_both_finetuned'           ###### rebuttal
embedding_base_dir = './embeddings'                   #########
styleclip_output_dir = './StyleCLIP_results'
experiments_output_dir = './output'

## Input info
### Input dir, where the images reside
input_data_path = ''
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'barcelona'                ###########
save_image_path = './assets/save_videos/v5_2_visual'
save_video_path='assets/save_videos/project5_2.mp4'

## Keywords
pti_results_keyword = 'PTI'          ########
e4e_results_keyword = 'e4e'          #########
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'
