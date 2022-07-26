bl_info = {
    "name": "Generate Transport Matrix",
    "author": "Yannick Hold-Geoffroy",
    "version": (0, 2, 0),
    "blender": (2, 92, 0),
    "category": "Import-Export",
    "location": "File > Export > Generate Transport Matrix",
    "description": "Export the current camera viewport to a transport matrix.",
}

import bpy
from mathutils import Vector
import bpy_extras
import numpy as np
import time


# TODOs:
# 1. Generate a mask to shrink the matrix size (remove pixels that doesn't intersect the scene)
# 2. Add support for (lambertian) albedo from blender's UI
# 3. Add anti-aliasing


# For each pixel, check intersection
# Inspirations:
# https://blender.stackexchange.com/questions/90698/projecting-onto-a-mirror-and-back/91019#91019
# https://blender.stackexchange.com/questions/13510/get-position-of-the-first-solid-crossing-limits-line-of-a-camera
def getClosestIntersection(clip_end, ray_begin, ray_direction):
    min_normal = None
    min_location = None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    success, location, normal, index, object, matrix = bpy.context.scene.ray_cast(depsgraph, ray_begin, ray_direction - ray_begin)

    if success:
        min_normal = matrix.to_3x3().normalized() @ normal.copy()
        min_location = location.copy()
    return min_normal, min_location


# Taken from https://github.com/soravux/skylibs/blob/master/envmap/projections.py
def latlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def skylatlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v / 2

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def getEnvmapDirections(envmap_size, envmap_type):
    """envmap_size = (rows, columns)
    envmap_type in ["latlong", "skylatlong"]"""
    cols = np.linspace(0, 1, envmap_size[1] * 2 + 1)
    rows = np.linspace(0, 1, envmap_size[0] * 2 + 1)

    cols = cols[1::2]
    rows = rows[1::2]

    u, v = [d.astype('float32') for d in np.meshgrid(cols, rows)]
    if envmap_type.lower() == "latlong":
        x, y, z, valid = latlong2world(u, v)
    elif envmap_type.lower() == "skylatlong":
        x, y, z, valid = skylatlong2world(u, v)
    else:
        raise Exception("Unknown format: {}. Should be either \"latlong\" or "
                        "\"skylatlong\".".format(envmap_type))

    return np.asarray([x, y, z]).reshape((3, -1)).T


class GenerateTransportMatrix(bpy.types.Operator):
    bl_idname = "export.generate_transport_matrix"
    bl_label = "Generate Transport Matrix"
    bl_options = {'REGISTER'}

    filepath = bpy.props.StringProperty(default="scene.msgpack", subtype="FILE_PATH")
    envmap_type = bpy.props.EnumProperty(name="Environment Map Type",
                                         items=[("SKYLATLONG", "skylatlong", "width should be 4x the height", "", 1),
                                                ("LATLONG", "latlong", "width should be 2x the height", "", 2)])
    envmap_height = bpy.props.IntProperty(name="Environment Map Height (px)", default=64)
    only_surface_normals = bpy.props.BoolProperty(name="Output only surface normals", default=False)

    def execute(self, context):

        T, normals, resolution = self.compute_transport(self.envmap_height, self.envmap_type, self.only_surface_normals)
        print("Saving to ", self.filepath)
        with open(self.filepath, "wb") as fhdl:
            np.savez(fhdl, T, normals, resolution)

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    @staticmethod
    def compute_occlusion(location, normal, envmap_directions, cam, envmap_directions_blender=None):
        if envmap_directions_blender is None:
            envmap_directions_blender = envmap_directions[:, [0, 2, 1]].copy()
            envmap_directions_blender[:, 1] = -envmap_directions_blender[:, 1]

        if normal.any():
            # For every envmap pixel
            # normal_np = np.array([normal[0], normal[2], -normal[1]])
            intensity = envmap_directions.dot(normal)
            # TODO: Add albedo here

            # Handle occlusions (single bounce)
            for idx in range(envmap_directions.shape[0]):
                # if the normal opposes the light direction, no need to raytrace.
                if intensity[idx] < 0:
                    intensity[idx] = 0
                    continue

                target_vec = Vector(envmap_directions_blender[idx, :])
                # Check for occlusions. The 1e-3 is just to be sure the
                # ray casting does not start right from the surface and
                # find itself as occlusion...
                normal_occ, _ = getClosestIntersection(cam.data.clip_end,
                                                       location + (1 / cam.data.clip_end) * 1e-3 * target_vec,
                                                       target_vec)

                if normal_occ:
                    intensity[idx] = 0
        else:
            intensity = np.zeros(envmap_directions.shape[0])
        return intensity

    @staticmethod
    def compute_transport(envmap_height, envmap_type, only_surface_normals):
        start_time = time.time()

        cam = bpy.data.objects['Camera']

        envmap_size = (
            envmap_height,
            2 * envmap_height if envmap_type.lower() == "latlong" else 4 * envmap_height)
        envmap_coords = cam.data.clip_end * getEnvmapDirections(envmap_size, envmap_type)
        envmap_coords_blender = envmap_coords[:, [0, 2, 1]].copy()
        envmap_coords_blender[:, 1] = -envmap_coords_blender[:, 1]

        # Get the render resolution
        resp = bpy.data.scenes["Scene"].render.resolution_percentage / 100.
        resx = int(bpy.data.scenes["Scene"].render.resolution_x * resp)
        resy = int(bpy.data.scenes["Scene"].render.resolution_y * resp)

        # Get the camera viewport corner 3D coordinates
        frame = cam.data.view_frame()
        tr, br, bl, tl = [cam.matrix_world @ corner for corner in frame]
        x = br - bl
        dx = x.normalized()
        y = tl - bl
        dy = y.normalized()

        whole_normals = []
        whole_positions = []
        print("Raycast scene...")
        # TODO: Shouldn't we use the center of the pixel instead of the corner?
        for s in range(resy):
            py = tl - s * y.length / float(resy) * dy
            normals_row = []
            position_row = []
            for b in range(resx):
                ray_direction = py + b * x.length / float(resx) * dx
                normal, location = getClosestIntersection(cam.data.clip_end, cam.location, ray_direction)
                # This coordinates system converts from blender's z=up to skylibs y=up
                normals_row.append([normal[0], normal[2], -normal[1]] if normal else [0, 0, 0])
                position_row.append(location)

            whole_normals.append(normals_row)
            whole_positions.append(position_row)

        whole_normals = np.asarray(whole_normals)

        print("Raycast light contribution")
        pixels = np.zeros((resx * resy, envmap_coords.shape[0]))
        if not only_surface_normals:
            for s in range(resy):
                for b in range(resx):
                    index = s * resx + b
                    intensity = GenerateTransportMatrix.compute_occlusion(whole_positions[s][b],
                                                                          whole_normals[s, b, :],
                                                                          envmap_coords,
                                                                          cam,
                                                                          envmap_coords_blender)
                    pixels[index, :] = intensity
                print("{}/{}".format(s, resy))

        print("Total time : {}".format(time.time() - start_time))
        return pixels, whole_normals, np.array([resy, resx])


def menu_export(self, context):
    # import os
    # default_path = os.path.splitext(bpy.data.filepath)[0] + ".npz"
    self.layout.operator(GenerateTransportMatrix.bl_idname, text="Transport Matrix (.npz)")  # .filepath = default_path


def register():
    bpy.types.TOPBAR_MT_file_export.append(menu_export)
    bpy.utils.register_class(GenerateTransportMatrix)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.utils.unregister_class(GenerateTransportMatrix)


if __name__ == "__main__":
    register()

    # Debug code
    #import time
    #time_start = time.time()
    #T, normals, res = GenerateTransportMatrix.compute_transport(12, "latlong", False)
    #print("Total time = {}".format(time.time() - time_start))
