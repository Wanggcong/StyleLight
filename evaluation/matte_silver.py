# blender --background --python 04_principled_bsdf.py --render-frame 1 -- </path/to/output/image> <resolution_percentage> <num_samples>

import bpy
import sys
import math
import os

working_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir_path)

import utils


def set_principled_node_as_mirror_silver(principled_node: bpy.types.Node) -> None:
    utils.set_principled_node(
        principled_node=principled_node,
        base_color=(0.8, 0.8, 0.8, 1.0),
        metallic=1.0,
        specular=0.0,
        roughness=0.3271,
    )


def set_scene_objects() -> bpy.types.Object:
    left_object = utils.create_smooth_sphere()

    mat = utils.add_material("sphere_mirror_silver", use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node_as_mirror_silver(principled_node)

    glossy_node = nodes.new(type="ShaderNodeBsdfGlossy")
    # set glossy node
    roughness = 0.369
    glossy_node.inputs['Roughness'].default_value = roughness

    mix_node = nodes.new(type='ShaderNodeMixShader')
    fresnel_node = nodes.new(type='ShaderNodeFresnel')
    fresnel_node.inputs['IOR'].default_value = 6.550

    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    links.new(principled_node.outputs['BSDF'], mix_node.inputs[1])
    links.new(glossy_node.outputs['BSDF'], mix_node.inputs[2])
    links.new(fresnel_node.outputs['Fac'], mix_node.inputs['Fac'])

    left_object.data.materials.append(mat)
    focus_target = bpy.context.object
    return focus_target


# Args
resolution_percentage = int(sys.argv[sys.argv.index('--') + 1])
num_samples = int(sys.argv[sys.argv.index('--') + 2])
hdri = sys.argv[sys.argv.index('--') + 3]
#save = sys.argv[sys.argv.index('--') + 4]
start = int(sys.argv[sys.argv.index('--') + 4])
end = int(sys.argv[sys.argv.index('--') + 5])



# Scene Building
scene = bpy.data.scenes["Scene"]
world = scene.world

## Reset
utils.clean_objects()

## Suzannes
focus_target = set_scene_objects()

## Camera
bpy.ops.object.camera_add(location=(0.0, -16.0, 2.0))
camera_object = bpy.context.object

utils.add_track_to_constraint(camera_object, focus_target)
utils.set_camera_params(camera_object.data, focus_target, lens=85, fstop=0.5)

scene.render.resolution_percentage = resolution_percentage
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
utils.set_cycles_renderer(scene, camera_object, num_samples, use_transparent_bg=True)
hdr_path = "./data/tone/" + hdri
save_path = "./data/render_results/" + hdri+ "/matte_silver/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
hdr_img=[]
for i in os.listdir(hdr_path):
    if 'exr' in i:
        hdr_img.append(i)
hdr_img.sort()

for d in hdr_img[start: end]:
    hdri_path = hdr_path+"/"+d
    utils.build_environment_texture_background(world, hdri_path)
    if True:
        scene.render.filepath = save_path+d.split('-')[0]+'_fake_image'
    else:
        scene.render.filepath = save_path+d.split('_')[0]+'_fake_image'
    bpy.ops.render.render(write_still=True)


