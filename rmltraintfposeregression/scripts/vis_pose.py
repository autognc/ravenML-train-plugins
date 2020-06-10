"""
Use Blender to to visualize pose results.

Run using: blender -P vis_pose.py -- [ARGS]

This script takes a directory of images, and a JSON file of results. The JSON file must be
a list of objects, one corresponding to each image (in sorted order), that each have the entries
"detected_pose" (quaternion format) and "centroid" (y, x in pixels).

It then uses Blender to render a set of X, Y, and Z axes oriented the same as the detected pose of
each image and overlays them on top of the image.
"""

import argparse
import os
import json
import glob
import sys
import tempfile
import bpy
import time
import numpy as np
import cv2
import shutil
from mathutils import Quaternion

DOWNSCALE_FACTOR = 1
OBJ_DISTANCE_FACTOR = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help="Path to directory of images", required=True)
    parser.add_argument('-r', '--results', type=str, help="Path to JSON file of results", required=True)
    parser.add_argument('-o', '--output', type=str, help="Path to output directory", required=True)
    parser.add_argument('--object', type=str, help="Name of the object to display as a wireframe (optional)", required=False)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

    if os.path.exists(args.output):
        if input('Artifact storage location contains old data. Overwrite?') in ['y', 'Y']:
            shutil.rmtree(args.output)
        else:
            return
    os.makedirs(args.output)

    # set the background color to pure black
    bpy.context.preferences.themes[0].view_3d.space.gradients.high_gradient = (0, 0, 0)
    # set view area so that only axes are visible
    space = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D').spaces[0]

    obj = None
    if args.object:
        obj = bpy.data.objects[args.object]

    if not obj:
        space.show_object_viewport_mesh = False
        space.overlay.show_axis_z = True
    else:
        space.overlay.show_axis_x = False
        space.overlay.show_axis_y = False
    space.show_object_viewport_camera = False
    space.show_object_viewport_light = False
    space.overlay.show_cursor = False
    space.overlay.show_floor = False

    # set up object, if requested
    if obj:
        obj.display_type = 'WIRE'
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = Quaternion([1, 0, 0, 0])
        bpy.ops.object.select_all(action='SELECT')

    # create a temp file for render output
    _, temp_path = tempfile.mkstemp(suffix='.png')
    bpy.context.scene.render.filepath = temp_path

    with open(args.results, 'r') as f:
        results = json.load(f)
    img_paths = sorted(glob.glob(os.path.join(args.directory, 'image_*')))

    for img_path, result in zip(img_paths, results):
        start_time = time.time()
        img = cv2.imread(img_path)
        ydim, xdim = img.shape[:2]
        bpy.context.scene.render.resolution_y = ydim * 2 // DOWNSCALE_FACTOR
        bpy.context.scene.render.resolution_x = xdim * 2 // DOWNSCALE_FACTOR

        region_3d = space.region_3d
        region_3d.view_rotation = Quaternion(result['detected_pose']).conjugated()
        if obj:
            region_3d.view_location = obj.location
            region_3d.view_distance = max(obj.dimensions * OBJ_DISTANCE_FACTOR)

        start_render_time = time.time()
        bpy.ops.render.opengl(write_still=True)
        render_time = time.time() - start_render_time

        blender_img = cv2.imread(temp_path)
        # upscale
        blender_img = cv2.resize(blender_img, (xdim * 2, ydim * 2))
        # crop so that center is in same spot
        cy, cx = result['centroid']
        blender_img = blender_img[ydim - cy:2 * ydim - cy, xdim - cx:2 * xdim - cx, :]
        # create mask of background pixels
        mask = np.sum(blender_img, axis=-1) < 6

        # overlay images
        combined = np.where(mask[..., None], img, blender_img)

        # save
        img_name = os.path.split(img_path)[-1]
        cv2.imwrite(os.path.join(args.output, img_name), combined)

        print(f'{img_name}  Render Time: {render_time}, Total Time: {time.time() - start_time}')
    os.remove(temp_path)


if __name__ == "__main__":
    try:
        main()
    finally:
        bpy.ops.wm.quit_blender()
