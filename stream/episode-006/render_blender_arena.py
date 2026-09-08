#!/usr/bin/env python3
"""Render an environment-only Blender opener for Episode 006."""

from pathlib import Path
import math
import subprocess
import bpy

HERE = Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)


def material(name, color, metallic=0.0, roughness=0.5, emission=None):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1)
        bsdf.inputs["Emission Strength"].default_value = 5.0
    return mat


floor_mat = material("Midnight floor", (0.012, 0.026, 0.04), 0.22, 0.3)
white = material("Goal", (0.72, 0.82, 0.88), 0.65, 0.18)
orange = material("Ball", (1.0, 0.12, 0.025), 0.15, 0.24)
cyan = material("Prediction", (0.02, 0.65, 0.9), 0.1, 0.2, (0.02, 0.65, 0.9))

bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.object
floor.data.materials.append(floor_mat)


def cube(name, location, scale, mat, bevel=0.04):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    mod = obj.modifiers.new("soft edges", "BEVEL")
    mod.width, mod.segments = bevel, 4
    return obj


cube("left post", (1.15, -1.5, 0.92), (0.07, 0.07, 0.92), white)
cube("right post", (1.15, 1.5, 0.92), (0.07, 0.07, 0.92), white)
cube("crossbar", (1.15, 0, 1.84), (0.07, 1.5, 0.07), white)
cube("goal line", (0.85, 0, 0.014), (0.025, 1.5, 0.014), cyan, 0.01)

bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.24,
                                     location=(-3.6, -0.65, 0.24))
ball = bpy.context.object
ball.data.materials.append(orange)
ball.keyframe_insert(data_path="location", frame=1)
ball.location = (0.6, 0.92, 0.24)
ball.rotation_euler[1] = math.radians(720)
ball.keyframe_insert(data_path="location", frame=210)
ball.keyframe_insert(data_path="rotation_euler", frame=210)

curve = bpy.data.curves.new("forecast line", "CURVE")
curve.dimensions, curve.bevel_depth, curve.bevel_resolution = "3D", 0.025, 5
spline = curve.splines.new("BEZIER")
spline.bezier_points.add(2)
for point, co in zip(spline.bezier_points,
                     [(-3.6, -0.65, 0.03), (-1.5, 0.1, 0.03), (0.85, 0.92, 0.03)]):
    point.co, point.handle_left_type, point.handle_right_type = co, "AUTO", "AUTO"
line = bpy.data.objects.new("predicted crossing", curve)
bpy.context.collection.objects.link(line)
curve.materials.append(cyan)

bpy.ops.object.light_add(type="AREA", location=(-1.8, -3.8, 5.2))
key = bpy.context.object
key.data.energy, key.data.shape, key.data.size = 1300, "DISK", 4.0
key.data.color = (0.72, 0.86, 1.0)
bpy.ops.object.light_add(type="AREA", location=(2.3, 2.8, 3.0))
rim = bpy.context.object
rim.data.energy, rim.data.size, rim.data.color = 950, 3.0, (0.15, 0.75, 1.0)

bpy.ops.object.empty_add(location=(-0.5, 0.15, 0.65))
target = bpy.context.object
bpy.ops.object.camera_add(location=(-5.4, -4.4, 2.8))
camera = bpy.context.object
camera.data.lens = 47
constraint = camera.constraints.new("TRACK_TO")
constraint.target, constraint.track_axis, constraint.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
camera.keyframe_insert(data_path="location", frame=1)
camera.location = (-4.3, 3.8, 2.15)
camera.keyframe_insert(data_path="location", frame=240)
bpy.context.scene.camera = camera

world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.002, 0.007, 0.015, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.18

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
scene.display.shading.cavity_type = "WORLD"
scene.display.shading.background_type = "WORLD"
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 2560, 1440, 100
scene.render.fps, scene.frame_start, scene.frame_end = 30, 1, 240
frames = HERE / "blender-opener-frames"
frames.mkdir(exist_ok=True)
scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.quality = 92
scene.render.filepath = str(frames / "frame_")
scene.render.film_transparent = False
scene.view_settings.look = "AgX - Medium High Contrast"
bpy.ops.wm.save_as_mainfile(filepath=str(HERE / "goalkeeper-arena.blend"))
bpy.ops.render.render(animation=True)
subprocess.run([
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", "30",
    "-i", str(frames / "frame_%04d.jpg"), "-vf", "fps=60", "-c:v",
    "h264_videotoolbox", "-b:v", "20M", "-pix_fmt", "yuv420p", "-movflags",
    "+faststart", str(HERE / "blender-arena-opener.mp4")], check=True)
